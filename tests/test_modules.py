from timm.models.mambaout import MambaOutStage
import torch



mamba_stage = MambaOutStage(dim=256, dim_out=256, depth=3)

x = torch.randn(2, 64, 64, 256)

y = mamba_stage(x)

print(y.shape)