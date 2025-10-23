# class ChannelGate_sub(nn.Module):
#     """A mini-network that generates channel-wise gates conditioned on input tensor."""

#     def __init__(self, in_channels, num_gates=None, return_gates=False,
#                  gate_activation='sigmoid', reduction=16, layer_norm=False):
#         super(ChannelGate_sub, self).__init__()
#         if num_gates is None:
#             num_gates = in_channels
#         self.return_gates = return_gates
#         self.global_avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, bias=True, padding=0)
#         self.norm1 = None
#         if layer_norm:
#             self.norm1 = nn.LayerNorm((in_channels//reduction, 1, 1))
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(in_channels//reduction, num_gates, kernel_size=1, bias=True, padding=0)
#         if gate_activation == 'sigmoid':
#             self.gate_activation = nn.Sigmoid()
#         elif gate_activation == 'relu':
#             self.gate_activation = nn.ReLU(inplace=True)
#         elif gate_activation == 'linear':
#             self.gate_activation = None
#         else:
#             raise RuntimeError("Unknown gate activation: {}".format(gate_activation))

#     def forward(self, x):
#         input = x
#         x = self.global_avgpool(x)
#         x = self.fc1(x)
#         if self.norm1 is not None:
#             x = self.norm1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         if self.gate_activation is not None:
#             x = self.gate_activation(x)
#         if self.return_gates:
#             return x
#         return input * x, input * (1 - x), x


# class SNR_Block(nn.Module):
#     def __init__(self, in_channels):
#         super(SNR_Block, self).__init__()
#         self.style_reid_layer1 = ChannelGate_sub(in_channels, num_gates=in_channels, return_gates=False,
#                  gate_activation='sigmoid', reduction=16, layer_norm=False)
#         self.IN1 = nn.InstanceNorm2d(in_channels, affine=True)

#     def forward(self, x):
#         x_IN_1 = self.IN1(x)
#         x_style_1 = x - x_IN_1
#         x_style_1_reid_useful, x_style_1_reid_useless, selective_weight_useful_1 = self.style_reid_layer1(x_style_1)
#         x_1 = x_IN_1 + x_style_1_reid_useful
#         return x_1

import torch
import torch.nn as nn
import torch.nn.functional as F


class GateBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(GateBlock, self).__init__()
        num_gates = in_channels
        mid_channels = in_channels // reduction

        # 平均池化分支
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1_avg = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.norm1_avg = nn.LayerNorm((mid_channels, 1, 1))
        self.relu = nn.ReLU(inplace=True)

        # 最大池化分支
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.fc1_max = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True)
        self.norm1_max = nn.LayerNorm((mid_channels, 1, 1))

        # 两路汇合后的第二层
        self.fc2 = nn.Conv2d(mid_channels, num_gates, kernel_size=1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        input = x

        # avg 分支
        avg = self.global_avgpool(x)
        avg = self.fc1_avg(avg)
        # LayerNorm 需要对 (C,1,1) 的张量指定 normalized_shape
        avg = self.norm1_avg(avg)
        avg = self.relu(avg)

        # max 分支
        mx = self.global_maxpool(x)
        mx = self.fc1_max(mx)
        mx = self.norm1_max(mx)
        mx = self.relu(mx)

        # 两路特征相加，再送入第二层
        x = self.fc2(avg + mx)
        x = self.gate_activation(x)

        return input * x


class StyleStrip(nn.Module):
    def __init__(self, in_channels):
        super(StyleStrip, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.style_strip = GateBlock(in_channels, reduction=16)

    def forward(self, x):
        x_IN_1 = self.instance_norm(x)
        x_style_1 = x - x_IN_1
        x_style_1_useful = self.style_strip(x_style_1)
        x_1 = x_IN_1 + x_style_1_useful
        return x_1


class SPP(nn.Module):
    """空间金字塔池化：多个不同尺度的 avgpool → upsample → concat"""
    def __init__(self, in_ch, out_ch, scales=(1,2,3,6)):
        super().__init__()
        self.Identity = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(scale)
            )
            for scale in scales
        ])
        self.out = nn.Sequential(
            nn.Conv2d(len(scales)*out_ch + out_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        feats = [self.Identity(x)]
        for blk in self.blocks:
            y = blk(x)      # pool
            y = F.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
            feats.append(y)
        return self.out(torch.cat(feats, dim=1))

class ASPP(nn.Module):
    """空洞空间金字塔：不同 dilation rates 的 conv"""
    def __init__(self, in_ch, out_ch, rates=(1,6,12,18)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False)
            for rate in rates
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(out_ch) for _ in rates])
        self.project = nn.Sequential(
            nn.Conv2d(len(rates)*out_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        outs = []
        for conv, bn in zip(self.convs, self.bns):
            y = bn(conv(x))
            outs.append(y)
        return self.project(torch.cat(outs, dim=1))

class DynamicDepthwiseConv2d(nn.Module):
    """
    轻量化动态卷积——只为 depthwise 卷积生成动态核，
    然后再接一个静态的 pointwise 卷积完成跨通道融合。
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # 生成 depthwise kernel 的全连接
        # kernel_dim = in_ch * 1 * ks * ks
        self.kernel_dim = in_ch * kernel_size * kernel_size
        # depthwise_bias_dim = in_ch
        self.bias_dim   = in_ch

        # 用 style_vec 生成 depthwise kernel 和 bias
        self.fc_k = nn.Linear(in_ch, self.kernel_dim)
        self.fc_b = nn.Linear(in_ch, self.bias_dim)

        # static pointwise conv
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

        self.stride, self.padding = stride, padding
        self.in_ch, self.out_ch = in_ch, out_ch
        self.ks = kernel_size

    def forward(self, x, style_vec):
        """
        x:         [N, in_ch, H, W]
        style_vec: [N, in_ch]
        """
        N, C, H, W = x.shape
        # 1. 生成 depthwise 权重 & bias
        #    kernels: [N, C*ks*ks] → [N, C, 1, ks, ks]
        kernels = self.fc_k(style_vec).view(N, C, 1, self.ks, self.ks)
        bias    = self.fc_b(style_vec).view(N, C)

        # 2. 批量并行执行 depthwise conv
        #    weight 需要 [N*C, 1, ks, ks]，groups=N*C
        w = kernels.view(N*C, 1, self.ks, self.ks)
        b = bias.view(N*C)

        #    把输入通道和 batch 合并到一起：x → [1, N*C, H, W]
        x_ = x.view(1, N*C, H, W)
        out = F.conv2d(
            x_, weight=w, bias=b,
            stride=self.stride, padding=self.padding,
            groups=N*C
        )  # → [1, N*C, H, W]

        # 3. 恢复形状 [N, C, H, W]
        out = out.view(N, C, H, W)

        # 4. 接静态 pointwise 完成跨通道融合
        out = self.pointwise(out)
        return out

class StyleContextModule(nn.Module):
    def __init__(self, in_ch=256, num_heads=8, reduction=16):
        super().__init__()

        self.reduction = reduction
        # 解耦
        self.inst_norm = nn.InstanceNorm2d(in_ch, affine=True)
        # 多头自注意力
        self.ln_mha = nn.LayerNorm(in_ch)
        self.mha    = nn.MultiheadAttention(embed_dim=in_ch, num_heads=num_heads, batch_first=True)
        # SPP 内容多尺度
        self.spp = SPP(in_ch, in_ch//self.reduction)
        # ASPP 风格多尺度
        self.aspp = ASPP(in_ch, in_ch//self.reduction)
        # 动态卷积
        self.style_pool = nn.AdaptiveAvgPool2d(1)
        self.dynamic_conv = DynamicDepthwiseConv2d(in_ch, in_ch//self.reduction, kernel_size=3)
        # 交叉注意力
        self.cross_ln = nn.Linear(in_ch, in_ch//self.reduction)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_ch//self.reduction, num_heads=num_heads, batch_first=True)
        # 融合
        self.fuse_ln  = nn.LayerNorm([4*(in_ch//self.reduction),1,1])
        self.fuse_conv= nn.Conv2d((2*(in_ch//self.reduction)+2*in_ch), in_ch, 1, bias=False)
        # SE 门控
        self.se_reduce = nn.Conv2d(in_ch, in_ch//self.reduction, 1)
        self.se_expand = nn.Conv2d(in_ch//self.reduction, in_ch, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        # 1. 解耦
        x_c = self.inst_norm(x)        # 内容
        x_s = x - x_c                  # 风格

        # 2. 内容自注意力
        # flatten 空间到长度 L=H*W
        xc_flat = x_c.view(N, C, -1).permute(0,2,1)  # [N, L, C]
        xc_flat = self.ln_mha(xc_flat)
        attn_out, _ = self.mha(xc_flat, xc_flat, xc_flat)
        F_c1 = attn_out.permute(0,2,1).view(N, C, H, W)

        # 3. 内容 SPP
        F_c2 = self.spp(x_c)

        # 4. 风格 ASPP
        F_s1 = self.aspp(x_s)

        # 5. 风格动态卷积
        style_vec = self.style_pool(x_s).view(N, C)  # [N, C]
        F_s2 = self.dynamic_conv(x_s, style_vec)    # [N, C//r, H, W]

        # 6. 交叉注意力（Q=内容, K/V=风格）
        # flatten到序列
        qc = F_c2.view(N, -1, H*W).permute(0,2,1)    # [N, L, C//r]
        ks = F_s1.view(N, -1, H*W).permute(0,2,1)
        vs = F_s2.view(N, -1, H*W).permute(0,2,1)
        qc = self.cross_ln(qc)
        F_cs, _ = self.cross_attn(qc, ks, vs)
        F_cs = F_cs.permute(0,2,1).view(N, C//self.reduction, H, W)

        # 7. 多路拼接 & 融合
        F_all = torch.cat([F_c1, F_c2, F_s1, F_cs], dim=1)  # [N, 4*C//r, H, W]
        fused = self.fuse_conv(F_all)

        # 8. SE 门控自适应融合
        se = F.adaptive_avg_pool2d(fused,1)
        se = F.relu(self.se_reduce(se), inplace=True)
        se = self.sigmoid(self.se_expand(se))
        out = x + se * fused

        return out


class StyleContextModuleLite(nn.Module):
    def __init__(self, in_ch=256, reduction=16, num_heads=8):
        super().__init__()
        self.reduction = reduction
        # 1. 解耦
        self.inst_norm = nn.InstanceNorm2d(in_ch, affine=True)
        # 2. 内容上下文
        self.linear = nn.Linear(in_ch, in_ch//reduction)
        self.linear_up = nn.Linear(in_ch//reduction, in_ch)
        self.attention = nn.MultiheadAttention(embed_dim=in_ch//self.reduction, num_heads=num_heads, batch_first=True)
        # 3. 风格门控
        self.gate     = StyleStrip(in_ch)
        # 4. 融合
        self.fuse = nn.Conv2d(2*in_ch, in_ch, 1, bias=False)
        # 可选的轻量 layernorm 或 BN
        self.bn   = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 解耦
        x_c = self.inst_norm(x)     # 内容
        x_s = x - x_c                 # 风格残差
        x_c = self.linear(x_c.view(x.size(0), x.size(1), -1).permute(0, 2, 1))  # [N, C//r, H*W]
        # 内容全局
        F_c, _ = self.attention(x_c, x_c, x_c)  # [N, H*W, C//r]
        F_c = self.linear_up(F_c).permute(0, 2, 1).view(x.size(0), x.size(1), x.size(2), x.size(3))
        # 风格有用部分
        F_s = self.gate(x_s)

        # 拼接融合
        out = self.fuse(torch.cat([F_c, F_s], dim=1))
        return self.relu(self.bn(out)) + x   


if __name__ == "__main__":
    # 测试代码
    x = torch.randn(2, 256, 64, 64)  # 假设输入是 [N, C, H, W]
    model = StyleContextModuleLite(256)
    out = model(x)
    print(out.shape)  # 应该输出 [2, 256, 64, 64] 