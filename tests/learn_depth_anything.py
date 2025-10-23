from change_detection.depth_anything_v2.depth2cd import Depth2CD
import torch
from torchsummary import summary

# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Depth2CD(feature_exchange_type='le').to(device)

print(model)