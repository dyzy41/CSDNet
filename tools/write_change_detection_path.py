# import os
# import random

# path = '/media/jic2/HDD/DSJJ/CDdata/MacaoCD'

# trainval_names = os.listdir(os.path.join(path, 'A'))

# random.shuffle(trainval_names)

# train_names = trainval_names[:7000]
# val_names = trainval_names[7000:]

# with open(os.path.join(path, 'train.txt'), 'w') as f:
#     for name in train_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name)):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name)
#             ))

# with open(os.path.join(path, 'val.txt'), 'w') as f:
#     for name in val_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name)):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name)
#             ))

# test_names = os.listdir(os.path.join(path, 'A'))

# with open(os.path.join(path, 'test.txt'), 'w') as f:
#     for name in test_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name)):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name)
#             ))

# print(f"Train samples: {len(train_names)}, Val samples: {len(val_names)}, Test samples: {len(test_names)}")


# import os
# import random

# path = '/media/jic2/HDD/DSJJ/CDdata/MacaoCD'

# trainval_names = os.listdir(os.path.join(path, 'A'))

# random.shuffle(trainval_names)

# train_names = trainval_names[:int(len(trainval_names) * 0.6)]
# val_names = trainval_names[int(len(trainval_names) * 0.6):int(len(trainval_names) * 0.8)]
# test_names = trainval_names[int(len(trainval_names) * 0.8):]


# with open(os.path.join(path, 'train.txt'), 'w') as f:
#     for name in train_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name)):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name)
#             ))

# with open(os.path.join(path, 'val.txt'), 'w') as f:
#     for name in val_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name)):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name)
#             ))


# with open(os.path.join(path, 'test.txt'), 'w') as f:
#     for name in test_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name)):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name)
#             ))

# print(f"Train samples: {len(train_names)}, Val samples: {len(val_names)}, Test samples: {len(test_names)}")


# import os
# import random

# path = '/home/realvm/CDdata/LuojiaCLCD'

# trainval_names = os.listdir(os.path.join(path, 'A'))

# random.shuffle(trainval_names)

# train_names = trainval_names[:int(len(trainval_names) * 0.6)]
# val_names = trainval_names[int(len(trainval_names) * 0.6):int(len(trainval_names) * 0.8)]
# test_names = trainval_names[int(len(trainval_names) * 0.8):]


# with open(os.path.join(path, 'train.txt'), 'w') as f:
#     for name in train_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name.replace('.tif', '.png'))):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name.replace('.tif', '.png'))
#             ))

# with open(os.path.join(path, 'val.txt'), 'w') as f:
#     for name in val_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name.replace('.tif', '.png'))):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name.replace('.tif', '.png'))
#             ))


# with open(os.path.join(path, 'test.txt'), 'w') as f:
#     for name in test_names:
#         if os.path.exists(os.path.join(path, 'A', name)) \
#         and os.path.exists(os.path.join(path, 'B', name)) \
#         and os.path.exists(os.path.join(path, 'label', name.replace('.tif', '.png'))):
#             f.writelines("{}  {}  {}\n".format(
#                 os.path.join(path, 'A', name),
#                 os.path.join(path, 'B', name),
#                 os.path.join(path, 'label', name.replace('.tif', '.png'))
#             ))

# print(f"Train samples: {len(train_names)}, Val samples: {len(val_names)}, Test samples: {len(test_names)}")


import os

path = '/home/realvm/CDdata/S2Looking'

split = ['train', 'val', 'test']

for s in split:
    with open(os.path.join(path, f'{s}.txt'), 'w') as f:
        for name in os.listdir(os.path.join(path, s, 'Image1')):
            if os.path.exists(os.path.join(path, s, 'Image1', name)) \
            and os.path.exists(os.path.join(path, s, 'Image2', name)) \
            and os.path.exists(os.path.join(path, s, 'label', name)):
                f.writelines("{}  {}  {}\n".format(
                    os.path.join(path, s, 'Image1', name),
                    os.path.join(path, s, 'Image2', name),
                    os.path.join(path, s, 'label', name)
                ))

print(f"Data preparation for MSRSCD completed. Files created for {split}.")