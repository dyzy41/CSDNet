from mmseg.registry import MODELS

import timm
import torch.nn as nn
import torch.nn.functional as F
from mmseg.registry import MODELS
from change_detection.utils.decode_block import *
from .StCoBlock import StyleStrip, StyleContextModule, StyleContextModuleLite
from .ccr_block import ContextualContentRefiner
from change_detection.utils.backbone import build_backbone
from change_detection.utils.exchange import FeatureExchanger, ExchangeType
import torch

from typing import Tuple, Union
import os

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out

class SE(nn.Module):

    def __init__(self, in_chnls, ratio=8):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls // ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls // ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return x*F.sigmoid(out)

class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class CSDNet_NL(nn.Module):
    def __init__(self, model_name='hrnet_w64'):
        super(CSDNet_NL, self).__init__()
        if model_name is None:
            model_name = 'hrnet_w64'
        self.model, self.num_stages, FPN_DICT, self.dim_change = build_backbone(model_name=model_name)
        
        self.fpn = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels'])
        )
        self.sigmoid = nn.Sigmoid()

        self.ss_block = nn.Sequential(
            StyleStrip(in_channels=FPN_DICT['in_channels'][0]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][1]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][2]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][3])
        )

        self.ex_func = FeatureExchanger(training=self.training)

        self.decode_conv = nn.Sequential(
            NonLocalBlock(FPN_DICT['out_channels']),
            nn.Conv2d(FPN_DICT['out_channels'], FPN_DICT['out_channels'], kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(FPN_DICT['out_channels'], 2, kernel_size=1)


    def decode_stage(self, feature_list):
        x1, x2, x3, x4 = feature_list
        x4 = self.decode_layersA[4](x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = x3 + x4
        x3 = self.decode_layersA[3](x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = x2 + x3
        x2 = self.decode_layersA[2](x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = x1 + x2
        x1 = self.decode_layersA[1](x1)
        return x1

    def decode_head(self, x):
        x = self.decode_conv(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv_seg(x)
        return x

    def forward(self, xA, xB):
        xA_list = self.model(xA)[self.num_stages-4:]
        xB_list = self.model(xB)[self.num_stages-4:]

        if self.dim_change:
            xA_list = [xA_list[i].permute(0, 3, 1, 2) for i in range(len(xA_list))]
            xB_list = [xB_list[i].permute(0, 3, 1, 2) for i in range(len(xB_list))]

        xA_list = [self.ss_block[i](xA_list[i]) for i in range(len(xA_list))]
        xB_list = [self.ss_block[i](xB_list[i]) for i in range(len(xB_list))]

        xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=ExchangeType.LAYER, thresh=0.5)

        xA_list = self.fpn(xA_list)
        xB_list = self.fpn(xB_list)

        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)
        outA = self.decode_head(outA)
        outB = self.decode_head(outB)
        change_maps = [outA, outB]
        return change_maps


class CSDNet_SE(nn.Module):
    def __init__(self, model_name='hrnet_w64'):
        super(CSDNet_SE, self).__init__()
        if model_name is None:
            model_name = 'hrnet_w64'
        self.model, self.num_stages, FPN_DICT, self.dim_change = build_backbone(model_name=model_name)
        
        self.fpn = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels'])
        )
        self.sigmoid = nn.Sigmoid()

        self.ss_block = nn.Sequential(
            StyleStrip(in_channels=FPN_DICT['in_channels'][0]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][1]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][2]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][3])
        )

        self.ex_func = FeatureExchanger(training=self.training)

        self.decode_conv = nn.Sequential(
            SE(FPN_DICT['out_channels']),
            nn.Conv2d(FPN_DICT['out_channels'], FPN_DICT['out_channels'], kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(FPN_DICT['out_channels'], 2, kernel_size=1)


    def decode_stage(self, feature_list):
        x1, x2, x3, x4 = feature_list
        x4 = self.decode_layersA[4](x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = x3 + x4
        x3 = self.decode_layersA[3](x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = x2 + x3
        x2 = self.decode_layersA[2](x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = x1 + x2
        x1 = self.decode_layersA[1](x1)
        return x1

    def decode_head(self, x):
        x = self.decode_conv(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv_seg(x)
        return x

    def forward(self, xA, xB):
        xA_list = self.model(xA)[self.num_stages-4:]
        xB_list = self.model(xB)[self.num_stages-4:]

        if self.dim_change:
            xA_list = [xA_list[i].permute(0, 3, 1, 2) for i in range(len(xA_list))]
            xB_list = [xB_list[i].permute(0, 3, 1, 2) for i in range(len(xB_list))]

        xA_list = [self.ss_block[i](xA_list[i]) for i in range(len(xA_list))]
        xB_list = [self.ss_block[i](xB_list[i]) for i in range(len(xB_list))]

        xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=ExchangeType.LAYER, thresh=0.5)

        xA_list = self.fpn(xA_list)
        xB_list = self.fpn(xB_list)

        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)
        outA = self.decode_head(outA)
        outB = self.decode_head(outB)
        change_maps = [outA, outB]
        return change_maps

class CSDNet_CBAM(nn.Module):
    def __init__(self, model_name='hrnet_w64'):
        super(CSDNet_CBAM, self).__init__()
        if model_name is None:
            model_name = 'hrnet_w64'
        self.model, self.num_stages, FPN_DICT, self.dim_change = build_backbone(model_name=model_name)
        
        self.fpn = MODELS.build(FPN_DICT)

        self.decode_layersA = nn.Sequential(
            nn.Identity(),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels']),
            ResDecodeBlock(in_channels=FPN_DICT['out_channels'])
        )
        self.sigmoid = nn.Sigmoid()

        self.ss_block = nn.Sequential(
            StyleStrip(in_channels=FPN_DICT['in_channels'][0]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][1]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][2]),
            StyleStrip(in_channels=FPN_DICT['in_channels'][3])
        )

        self.ex_func = FeatureExchanger(training=self.training)

        self.decode_conv = nn.Sequential(
            CBAM(FPN_DICT['out_channels']),
            nn.Conv2d(FPN_DICT['out_channels'], FPN_DICT['out_channels'], kernel_size=3, padding=1),
            nn.BatchNorm2d(FPN_DICT['out_channels']),
            nn.ReLU(inplace=True)
        )
        self.conv_seg = nn.Conv2d(FPN_DICT['out_channels'], 2, kernel_size=1)


    def decode_stage(self, feature_list):
        x1, x2, x3, x4 = feature_list
        x4 = self.decode_layersA[4](x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x3 = x3 + x4
        x3 = self.decode_layersA[3](x3)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x2 = x2 + x3
        x2 = self.decode_layersA[2](x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = x1 + x2
        x1 = self.decode_layersA[1](x1)
        return x1

    def decode_head(self, x):
        x = self.decode_conv(x)
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.conv_seg(x)
        return x

    def forward(self, xA, xB):
        xA_list = self.model(xA)[self.num_stages-4:]
        xB_list = self.model(xB)[self.num_stages-4:]

        if self.dim_change:
            xA_list = [xA_list[i].permute(0, 3, 1, 2) for i in range(len(xA_list))]
            xB_list = [xB_list[i].permute(0, 3, 1, 2) for i in range(len(xB_list))]

        xA_list = [self.ss_block[i](xA_list[i]) for i in range(len(xA_list))]
        xB_list = [self.ss_block[i](xB_list[i]) for i in range(len(xB_list))]

        xA_list, xB_list = self.ex_func.exchange(xA_list, xB_list, mode=ExchangeType.LAYER, thresh=0.5)

        xA_list = self.fpn(xA_list)
        xB_list = self.fpn(xB_list)

        outA = self.decode_stage(xA_list)
        outB = self.decode_stage(xB_list)
        outA = self.decode_head(outA)
        outB = self.decode_head(outB)
        change_maps = [outA, outB]
        return change_maps


if __name__ == '__main__':
    model = SNRNet()
    xA = torch.randn(2, 3, 256, 256)
    xB = torch.randn(2, 3, 256, 256)
    outA, outB = model(xA, xB)
    print(outA.shape, outB.shape)  # Should print torch.Size([2, 2, 256, 256]) for both outputs
    print("Model forward pass successful.")