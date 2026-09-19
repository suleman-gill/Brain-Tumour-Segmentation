import os
from tkinter import image_names

import numpy as np

def load_image(image_path:str,image_name):
    images = []
    for i in image_name:
        if (i.split(".")[-1] == "npy"):
            temp = np.load(image_path+i).astype(np.float32)
            images.append(temp)
    return np.array(images)

def image_loader(image_path:str,image_name,mask_path:str,mask_name,batch_size:int):
    L = len(image_name)
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end,L)
            images = load_image(image_path,image_name[batch_start:limit])
            masks = load_image(mask_path,mask_name[batch_start:limit])
            yield (images,masks)
            batch_start += batch_size
            batch_end += batch_size
'''
img_path = 'numpy_data_1/train_images/'
mask_path = 'numpy_data_1/train_masks/'
img_name = sorted(os.listdir(img_path))
mask_name = sorted(os.listdir(mask_path))
loader = image_loader(img_path, img_name, mask_path, mask_name, batch_size=64)'''
'''for i in range(800):
    image, mask = next(loader)
    print(image.shape, mask.shape)
'''



