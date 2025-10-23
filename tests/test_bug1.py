
from utils.dataset import CDTXTDataset
import os
from utils.transforms import get_best_model_checkpoint, define_transforms
import cv2
import numpy as np
import tqdm


data_root = '/media/jic2/HDD/DSJJ/CDdata/LEVIR-CD'
batch_size =  1
num_workers = 1
src_size = 1024
crop_size = 256

from torch.utils.data import DataLoader

train_transform, test_transform = define_transforms(if_crop=src_size>crop_size, crop_size=crop_size)


train_dataset = CDTXTDataset(os.path.join(data_root, 'train.txt'), transform=train_transform)
val_dataset = CDTXTDataset(os.path.join(data_root, 'val.txt'), transform=test_transform)
test_dataset = CDTXTDataset(os.path.join(data_root, 'test.txt'), transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

vis_path = 'vis'
if not os.path.exists(vis_path):
    os.makedirs(vis_path)

for i, batch in tqdm.tqdm(enumerate(train_loader)):
    imgA = batch['imgAB'][0, :3, :, :]
    imgB = batch['imgAB'][0, 3:, :, :]

    imgA_vis = imgA.permute(1, 2, 0).numpy()
    imgB_vis = imgB.permute(1, 2, 0).numpy()

    lab = batch['lab'][0]
    lab_vis = lab.numpy()*255
    lab_vis = lab_vis.astype(np.uint8)
    lab_vis = cv2.applyColorMap(lab_vis, cv2.COLORMAP_JET)
    cat_result = np.concatenate((imgA_vis, imgB_vis, lab_vis), axis=1)
    cv2.imwrite(os.path.join(vis_path, f'vis_{i}.png'), cat_result)
