import timm
import torch

model = timm.create_model('swinv2_tiny_window8_256', pretrained=False, features_only=True)

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

model = model.to(device)
x = torch.randn(2, 3, 256, 256).to(device)

features = model(x)
for i, f in enumerate(features):
    print(f"Feature {i}: shape={f.shape}")
