import torch
import torch.nn as nn
import torch.nn.functional as F

class LaplacianPyramid(nn.Module):
    def __init__(self, num_levels=3, kernel_size=5, sigma=1.0):
        super(LaplacianPyramid, self).__init__()
        self.num_levels = num_levels
        # 构建高斯核
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.gaussian_kernel = self._get_gaussian_kernel(kernel_size, sigma)
        
    def _get_gaussian_kernel(self, kernel_size, sigma):
        import numpy as np
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / np.sum(kernel)
        kernel = torch.from_numpy(kernel).float()
        return kernel

    def _gaussian_blur(self, x):
        C = x.shape[1]
        kernel = self.gaussian_kernel.to(x.device).unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
        kernel = kernel.repeat(C, 1, 1, 1)  # (C,1,K,K)
        return F.conv2d(x, kernel, padding=self.padding, groups=C)
    
    def forward(self, x):
        pyramid = []
        cur = x
        for level in range(self.num_levels - 1):
            # 高斯模糊
            blurred = self._gaussian_blur(cur)
            # 下采样
            down = F.interpolate(blurred, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True)
            # 上采样到原尺寸
            up = F.interpolate(down, size=cur.shape[2:], mode='bilinear', align_corners=False)
            # 计算拉普拉斯层
            laplacian = cur - up
            pyramid.append(laplacian)
            cur = down
        pyramid.append(cur)  # 最后一层直接加进去
        return pyramid  # list, 每层shape: (N,C,H,W)



class LaplacianBackbone(nn.Module):
    def __init__(self, channel_list, kernel_size=5, sigma=1.0):
        super(LaplacianBackbone, self).__init__()
        self.laplacian_pyramid_64 = LaplacianPyramid(num_levels=4, kernel_size=kernel_size, sigma=sigma)
        self.laplacian_pyramid_32 = LaplacianPyramid(num_levels=3, kernel_size=kernel_size, sigma=sigma)
        self.laplacian_pyramid_16 = LaplacianPyramid(num_levels=2, kernel_size=kernel_size, sigma=sigma)

        self.conv0 = nn.Sequential(
            nn.Conv2d(channel_list[0], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_list[1], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_list[2], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_list[3], 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )


    def forward(self, x_list):
        x0, x1, x2, x3 = x_list

        x0 = self.conv0(x0)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x_lap_list = []

        x0_list = self.laplacian_pyramid_64(x0)
        x0_, x1_, x2_, x3_ = x0_list[0], x0_list[1], x0_list[2], x0_list[3]
        x_lap_list.append(x0_)
        x1 = x1+x1_
        x2 = x2+x2_
        x3 = x3+x3_

        x1_list = self.laplacian_pyramid_32(x1)
        x1_, x2_, x3_ = x1_list[0], x1_list[1], x1_list[2]
        x_lap_list.append(x1_)
        x2 = x2 + x2_
        x3 = x3 + x3_

        x2_list = self.laplacian_pyramid_16(x2)
        x2_, x3_ = x2_list[0], x2_list[1]
        x_lap_list.append(x2_)

        x3 = x3 + x3_
        x_lap_list.append(x3)

        return x_lap_list





# 用法示例
if __name__ == '__main__':
    x = torch.randn(2, 256, 64, 64)  # batch=2, channel=256, H=W=64
    lp = LaplacianPyramid(num_levels=4)
    pyr = lp(x)
    for i, l in enumerate(pyr):
        print(f"Laplacian Level {i}: shape={l.shape}")
