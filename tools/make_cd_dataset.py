import os
import glob
import numpy as np
import rasterio
from PIL import Image
import tqdm


# —— 配置 —— #
T1_DIR      = 'T1'
T2_DIR      = 'T2'
LABEL_DIR   = 'labels_change'
OUTPUT_DIR  = 'tiles'         # 切块输出根目录
PATCH_SIZE  = 256

# 三个子文件夹
A_DIR     = os.path.join(OUTPUT_DIR, 'A')
B_DIR     = os.path.join(OUTPUT_DIR, 'B')
LABEL_OUT = os.path.join(OUTPUT_DIR, 'label')

for d in (A_DIR, B_DIR, LABEL_OUT):
    os.makedirs(d, exist_ok=True)

def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

t1_dict    = {get_basename(p): p for p in glob.glob(os.path.join(T1_DIR, '*.tif'))}
t2_dict    = {get_basename(p): p for p in glob.glob(os.path.join(T2_DIR, '*.tif'))}
label_dict = {get_basename(p): p for p in glob.glob(os.path.join(LABEL_DIR, '*.png'))}
keys       = sorted(set(t1_dict) & set(t2_dict) & set(label_dict))

def pad_reflect(arr, target_h, target_w):
    if arr.ndim == 3:
        _, h, w = arr.shape
    else:
        h, w = arr.shape
    pad_h = target_h - h
    pad_w = target_w - w
    top    = pad_h // 2
    bottom = pad_h - top
    left   = pad_w // 2
    right  = pad_w - left
    if arr.ndim == 3:
        pad_width = ((0,0), (top, bottom), (left, right))
    else:
        pad_width = ((top, bottom), (left, right))
    return np.pad(arr, pad_width, mode='reflect')

id_counter = 0

for key in tqdm.tqdm(keys):
    # 读取 T1
    with rasterio.open(t1_dict[key]) as src1:
        arr1 = src1.read()           # (bands, H, W)
    # 读取 T2
    with rasterio.open(t2_dict[key]) as src2:
        arr2 = src2.read()
    # 读取 label
    lbl = np.array(Image.open(label_dict[key]).convert('L'))  # (H, W)

    _, h, w = arr1.shape
    new_h = int(np.ceil(h / PATCH_SIZE) * PATCH_SIZE)
    new_w = int(np.ceil(w / PATCH_SIZE) * PATCH_SIZE)

    arr1_pad = pad_reflect(arr1, new_h, new_w)
    arr2_pad = pad_reflect(arr2, new_h, new_w)
    lbl_pad  = pad_reflect(lbl,   new_h, new_w)

    for i in range(0, new_h, PATCH_SIZE):
        for j in range(0, new_w, PATCH_SIZE):
            tile1 = arr1_pad[:, i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            tile2 = arr2_pad[:, i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            tileL = lbl_pad[    i:i+PATCH_SIZE, j:j+PATCH_SIZE]

            id_str = f"{id_counter:06d}"

            # —— 保存 A (T1) 切块 —— #
            # 转为 H×W×C，假设 bands<=4 且像素值已在 [0,255]
            img1 = np.transpose(tile1, (1,2,0))
            Image.fromarray(img1.astype(np.uint8)).save(os.path.join(A_DIR, f"{id_str}.png"))

            # —— 保存 B (T2) 切块 —— #
            img2 = np.transpose(tile2, (1,2,0))
            Image.fromarray(img2.astype(np.uint8)).save(os.path.join(B_DIR, f"{id_str}.png"))

            # —— 保存 label 切块 —— #
            Image.fromarray(tileL.astype(np.uint8)).save(os.path.join(LABEL_OUT, f"{id_str}.png"))

            id_counter += 1

print(f"所有切块已保存，共 {id_counter} 组。")
