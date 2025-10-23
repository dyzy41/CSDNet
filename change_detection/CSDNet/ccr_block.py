import torch
import torch.nn as nn

class ContextualGate(nn.Module):
    """
    一个集成了通道门控和空间门控的模块。
    它接收一个特征图，并生成一个时空注意力图来对其进行精炼。
    """
    def __init__(self, in_channels, reduction=16):
        super(ContextualGate, self).__init__()

        # --- 通道门控分支 (Channel Gate Branch) ---
        # 沿用您的优秀设计，结合全局平均和最大池化
        mid_channels = in_channels // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc_shared = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=False)
        )
        
        # --- 空间门控分支 (Spatial Gate Branch) ---
        # 使用一个卷积层来融合跨通道的平均和最大值信息
        self.spatial_conv = nn.Sequential(
            # 输入是2个通道 (avg_pool, max_pool across channels)
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False), 
            nn.BatchNorm2d(1)
        )

        self.gate_activation = nn.Sigmoid()

    def forward(self, x):
        # --- 通道门控计算 ---
        avg_out = self.fc_shared(self.avg_pool(x))
        max_out = self.fc_shared(self.max_pool(x))
        channel_gate = self.gate_activation(avg_out + max_out) # shape: [B, C, 1, 1]

        # --- 空间门控计算 ---
        # 沿着通道维度进行池化
        avg_across_channels = torch.mean(x, dim=1, keepdim=True)
        max_across_channels, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接并进行卷积
        spatial_input = torch.cat([avg_across_channels, max_across_channels], dim=1)
        spatial_gate = self.gate_activation(self.spatial_conv(spatial_input)) # shape: [B, 1, H, W]

        # --- 应用门控 ---
        # 将通道门控和空间门控顺序应用到输入特征上
        # 广播机制会自动处理 (B,C,1,1) * (B,C,H,W) 和 (B,1,H,W) * (B,C,H,W)
        x_gated = x * channel_gate
        x_gated = x_gated * spatial_gate
        
        return x_gated


class ContextualContentRefiner(nn.Module):
    """
    上下文内容精炼模块 (CCR)
    基于您的思路，融合了时空门控进行特征精炼。
    """
    def __init__(self, in_channels, reduction=16):
        super(ContextualContentRefiner, self).__init__()
        # 1. 分解层：Instance Normalization
        # affine=False, 因为我们想自己学习变换，而不是让IN层学
        self.instance_norm = nn.InstanceNorm2d(in_channels, affine=False) 
        
        # 2. 门控层：使用我们新设计的时空联合门控
        self.contextual_gate = ContextualGate(in_channels, reduction)

    def forward(self, x):
        # 1. 分解特征为“内容”和“风格”
        x_content = self.instance_norm(x)
        x_style = x - x_content
        
        # 2. 对“风格”特征进行时空门控，筛选出有用的部分
        x_style_useful = self.contextual_gate(x_style)
        
        # 3. 重组内容和有用的风格
        # 最终输出是基础内容加上经过精炼的、对任务有益的风格信息
        output = x_content + x_style_useful
        
        return output
    
if __name__ == '__main__':
    # --- 模拟参数 ---
    batch_size = 4
    channels = 256  # 假设是特征图的通道数
    height, width = 64, 64
    
    # 创建一个随机的输入特征图
    feature_map = torch.randn(batch_size, channels, height, width)
    
    # --- 初始化并使用 CCR 模块 ---
    ccr_module = ContextualContentRefiner(in_channels=channels)
    
    # 前向传播
    refined_feature_map = ccr_module(feature_map)
    
    # --- 打印结果 ---
    print(f"Input feature map shape:       {feature_map.shape}")
    print(f"Refined feature map shape:     {refined_feature_map.shape}")
    
    # 检查输出shape是否与输入一致
    assert feature_map.shape == refined_feature_map.shape
    print("\nCCR module works correctly!")

    # 预期输出:
    # Input feature map shape:       torch.Size([4, 256, 64, 64])
    # Refined feature map shape:     torch.Size([4, 256, 64, 64])
    #
    # CCR module works correctly!