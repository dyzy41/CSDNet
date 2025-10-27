import torch
from thop import profile, clever_format
from change_detection.CSDNet.StCoNet import CSDNet

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

model = CSDNet('swinv2_small_window8_256').eval().to(device)
x1 = torch.randn(1, 3, 256, 256).to(device)
x2 = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    macs, params = profile(model, inputs=(x1, x2), verbose=False)

true_flops = macs * 2  # 严格 FLOPs 口径
F_macs, P = clever_format([macs, params], "%.3f")
F_flops, _ = clever_format([true_flops, params], "%.3f")

print(f"MACs: {F_macs}, Params: {P}")
print(f"FLOPs(=2*MACs): {F_flops}, Params: {P}")
