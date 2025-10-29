import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import cv2
from change_detection.CSDNet.StCoNet import CSDNet
import torch.nn.functional as F

class FeatureExtractor:
    """用于提取中间层特征的钩子类"""
    def __init__(self):
        self.features = None
    
    def hook(self, module, input, output):
        self.features = output.detach()

def load_model(checkpoint_path, device):
    """加载预训练模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型实例
    model = CSDNet(
        in_channels=3,
        num_classes=2,
        backbone_name='hrnet_w48'
    )
    
    # 加载权重
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # 移除 'model.' 前缀（如果有）
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def load_image_pair(img_a_path, img_b_path, img_size=256):
    """加载图像对并进行预处理"""
    # 读取图像
    img_a = Image.open(img_a_path).convert('RGB')
    img_b = Image.open(img_b_path).convert('RGB')
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    img_a_tensor = transform(img_a).unsqueeze(0)
    img_b_tensor = transform(img_b).unsqueeze(0)
    
    # 保存原始图像用于可视化
    img_a_np = np.array(img_a.resize((img_size, img_size)))
    img_b_np = np.array(img_b.resize((img_size, img_size)))
    
    return img_a_tensor, img_b_tensor, img_a_np, img_b_np

def generate_heatmap(features, original_img, alpha=0.5):
    """
    生成热力图
    Args:
        features: 特征图 [B, C, H, W]
        original_img: 原始图像 [H, W, 3]
        alpha: 热力图透明度
    """
    # 对通道维度取平均，得到 [B, H, W]
    heatmap = torch.mean(features, dim=1).squeeze().cpu().numpy()
    
    # 归一化到 0-255
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # 调整大小以匹配原始图像
    h, w = original_img.shape[:2]
    heatmap = cv2.resize(heatmap, (w, h))
    
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # 叠加到原始图像上
    superimposed = cv2.addWeighted(original_img, 1-alpha, heatmap_colored, alpha, 0)
    
    return heatmap_colored, superimposed

def visualize_and_save(img_a_np, img_b_np, heatmap_a, heatmap_b, 
                       superimposed_a, superimposed_b, save_path, name):
    """可视化并保存结果"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：时相A
    axes[0, 0].imshow(img_a_np)
    axes[0, 0].set_title('Time A - Original', fontsize=14)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(heatmap_a)
    axes[0, 1].set_title('Time A - Heatmap', fontsize=14)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(superimposed_a)
    axes[0, 2].set_title('Time A - Superimposed', fontsize=14)
    axes[0, 2].axis('off')
    
    # 第二行：时相B
    axes[1, 0].imshow(img_b_np)
    axes[1, 0].set_title('Time B - Original', fontsize=14)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(heatmap_b)
    axes[1, 1].set_title('Time B - Heatmap', fontsize=14)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(superimposed_b)
    axes[1, 2].set_title('Time B - Superimposed', fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{name}_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {os.path.join(save_path, f'{name}_heatmap.png')}")

def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from {args.checkpoint_path}")
    model = load_model(args.checkpoint_path, device)
    
    # 注册钩子到 decode_conv 层
    feature_extractor = FeatureExtractor()
    if hasattr(model, 'decode_head') and hasattr(model.decode_head, 'decode_conv'):
        hook_handle = model.decode_head.decode_conv.register_forward_hook(feature_extractor.hook)
        print("Hook registered to decode_head.decode_conv")
    else:
        print("Warning: Could not find decode_head.decode_conv in model")
        return
    
    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载图像对
    print(f"Loading images: {args.img_a} and {args.img_b}")
    img_a_tensor, img_b_tensor, img_a_np, img_b_np = load_image_pair(
        args.img_a, args.img_b, args.img_size
    )
    
    # 拼接输入
    img_ab = torch.cat([img_a_tensor, img_b_tensor], dim=1).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = model(img_ab)
    
    # 获取特征图
    features = feature_extractor.features
    print(f"Extracted features shape: {features.shape}")
    
    # 生成热力图 - 对两个时相分别处理
    # 假设特征图对两个时相是共享的，我们可以为每个时相生成相同的热力图
    # 或者您可以修改模型以分别提取两个时相的特征
    heatmap_a, superimposed_a = generate_heatmap(features, img_a_np, args.alpha)
    heatmap_b, superimposed_b = generate_heatmap(features, img_b_np, args.alpha)
    
    # 可视化并保存
    img_name = os.path.splitext(os.path.basename(args.img_a))[0]
    visualize_and_save(
        img_a_np, img_b_np,
        heatmap_a, heatmap_b,
        superimposed_a, superimposed_b,
        args.save_dir, img_name
    )
    
    # 移除钩子
    hook_handle.remove()
    print("Visualization completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate attention heatmaps for CSDNet')
    
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--img_a', type=str, required=True,
                       help='Path to time A image')
    parser.add_argument('--img_b', type=str, required=True,
                       help='Path to time B image')
    parser.add_argument('--save_dir', type=str, default='heatmap_results',
                       help='Directory to save heatmap visualizations')
    parser.add_argument('--img_size', type=int, default=256,
                       help='Input image size')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency of heatmap overlay (0-1)')
    
    args = parser.parse_args()
    main(args)