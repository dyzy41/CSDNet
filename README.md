## 项目名称

这是一个非常简单的使用Pytorch-lightning工具搭建变化检测算法的训练测试推理框架的代码。
亲测性能稍弱于mmlab的框架，但是代码结构清晰，易于理解和修改。


## 📂 仓库结构

# 遥感变化检测训练框架

基于PyTorch-Lightning搭建的变化检测算法训练/测试框架，支持多种主流变化检测模型。

## 主要特性

- 支持20+种变化检测模型：
  - MM_ISDANet, MM_MSCANet, MM_RCTNet, MM_BASNet, MM_DARNet
  - MM_ScratchFormer, MM_HATNet, MM_ELGCNet, MM_DMINet, MM_CGNet
  - MM_SiamUNet_conc, MM_SiamUNet_diff, MM_CDNet等
  
- 完整训练流程：
  - 支持滑动窗口推理大尺寸图像
  - 丰富的训练监控指标(IoU, F1, Recall等)
  - 早停机制和模型检查点保存

- 便捷的测试功能：
  - 自动计算各项评估指标
  - 结果可视化保存

- 基于comet.ml的实验管理：
  - 可视化训练过程

## 快速开始

 安装依赖：
```bash
bash install_env.sh


