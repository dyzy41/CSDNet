import numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2


x = np.ones((256, 256, 3), dtype=np.float32)

x = x*128.0


train_transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
        ]
    )

x = train_transform(image=x)['image']
print(x)
