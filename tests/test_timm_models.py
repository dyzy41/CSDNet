import os
import timm
import torch.nn as nn
import torch

# model = timm.create_model('mambaout_base', pretrained=True, features_only=True)
# x = torch.randn(2, 3, 256, 256)

# out = model(x)

# for i, feature in enumerate(out):
#     print(f"Feature {i}: {feature.shape}")


model = timm.create_model('hrnet_w64', pretrained=True, features_only=True)
# model = timm.create_model('convnext_base.clip_laion2b_augreg_ft_in12k_in1k', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'convnext_base.clip_laion2b_augreg_ft_in12k_in1k/pytorch_model.bin')), features_only=True)
x = torch.randn(2, 3, 256, 256)

out = model(x)

for i, feature in enumerate(out):
    print(f"Feature {i}: {feature.shape}")

