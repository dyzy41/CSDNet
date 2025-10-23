import os
import cv2
import numpy as np


p = '/media/jic2/HDD/DSJJ/CDdata/GZCD/label'

files = os.listdir(p)

for file in files:
    file_path = os.path.join(p, file)
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    
    # 将标签值为 255 的像素点设置为 1
    img[img > 0] = 255
    # 保存修改后的图像
    cv2.imwrite(file_path, img)