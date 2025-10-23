from torch import randint
from torchmetrics.segmentation import DiceScore
import torch
# preds = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 prediction
# target = randint(0, 2, (4, 5, 16, 16))  # 4 samples, 5 classes, 16x16 target

# print(preds)
# print(target)

# dice_score = DiceScore(num_classes=5, average="micro")
# print(dice_score(preds, target))

# dice_score = DiceScore(num_classes=5, average="none")
# print(dice_score(preds, target))

from torchvision.io import read_image

# 替换为你自己的图片路径
img_path = "/home/jic2/git_code/remote-sensing-change-detection/EfficientCD_LEVIR_TrainingFiles/test_results/test_10.png"
lab_path = '/media/jic2/HDD/DSJJ/CDdata/LEVIR-CD/test/label/test_10.png'

img_tensor = read_image(img_path)//255
lab_tensor = read_image(lab_path)//255
print(torch.max(img_tensor))
print(torch.max(lab_tensor))
dice_score = DiceScore(num_classes=2, average="micro")
print(dice_score(img_tensor, lab_tensor))

