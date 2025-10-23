import os
import cv2
import tqdm

path = '/home/realvm/CDdata/LuojiaCLCD'

tgt_size = 1024

files = os.listdir(os.path.join(path, 'A'))

for file in tqdm.tqdm(files):
    img_a = cv2.imread(os.path.join(path, 'A', file))
    img_b = cv2.imread(os.path.join(path, 'B', file))
    label = cv2.imread(os.path.join(path, 'label', file.replace('.tif', '.png')), cv2.IMREAD_GRAYSCALE)

    img_a = cv2.resize(img_a, (tgt_size, tgt_size))
    img_b = cv2.resize(img_b, (tgt_size, tgt_size)) 
    label = cv2.resize(label, (tgt_size, tgt_size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(path, 'A', file), img_a)
    cv2.imwrite(os.path.join(path, 'B', file), img_b)
    cv2.imwrite(os.path.join(path, 'label', file.replace('.tif', '.png')), label)