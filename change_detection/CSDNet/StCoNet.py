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
from timm.models.mambaout import MambaOutStage
from .LaplacianPyramid import LaplacianBackbone

from typing import Tuple, Union
import os


class CSDNet(nn.Module):
    def __init__(self, model_name='hrnet_w64'):
        super(CSDNet, self).__init__()
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
            ContextualContentRefiner(FPN_DICT['out_channels']),
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