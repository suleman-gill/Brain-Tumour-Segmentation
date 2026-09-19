import numpy as np
import glob
import os
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras.utils import to_categorical
import cv2


mm = MinMaxScaler()
ss = StandardScaler()

fla_paths = sorted(glob.glob('data/MICCAI_BraTS2020_TrainingData/BraTS*/*flair.nii'))
print(fla_paths[:2])
t1ce_paths = sorted(glob.glob('data/MICCAI_BraTS2020_TrainingData/BraTS*/*t1ce.nii'))
print(t1ce_paths[:2])
t2_paths = sorted(glob.glob('data/MICCAI_BraTS2020_TrainingData/BraTS*/*t2.nii'))
print(t2_paths[:2])
seg_paths = sorted(glob.glob('data/MICCAI_BraTS2020_TrainingData/BraTS*/*seg.nii'))
print(seg_paths[:2])


np.random.seed(786)
val_images = np.random.randint(0,len(fla_paths),int(len(fla_paths)*0.2))
h = 0
n = 192
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(9,9))
start,end = 34,226
os.makedirs('numpy_data/train_images',exist_ok=True)
os.makedirs('numpy_data/train_masks',exist_ok=True)
os.makedirs('numpy_data/val_images',exist_ok=True)
os.makedirs('numpy_data/val_masks',exist_ok=True)

for i in range(len(fla_paths)):
    fla = nib.load(fla_paths[i]).get_fdata()
    t1ce = nib.load(t1ce_paths[i]).get_fdata()
    t2 = nib.load(t2_paths[i]).get_fdata()
    mask = nib.load(seg_paths[i]).get_fdata()

    fla_min = mm.fit_transform(fla.reshape(-1, fla.shape[-1])).reshape(fla.shape)
    fla_z = ss.fit_transform(fla.reshape(-1, fla.shape[-1])).reshape(fla.shape)

    t1ce_min = mm.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
    t1ce_z = ss.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)

    t2_min = mm.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
    t2_z = ss.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)

    combined_arr_min = np.stack([fla_min, t1ce_min, t2_min], axis=3)
    combined_arr_min = combined_arr_min[start:end, start:end, 13:141, :]

    combined_arr_z = np.stack([fla_z, t1ce_z, t2_z], axis=3)
    combined_arr_z = combined_arr_z[start:end, start:end, 13:141, :]

    fla_cla = cv2.normalize(fla, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    t1ce_cla = cv2.normalize(t1ce, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    t2_cla = cv2.normalize(t2, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    combined_arr_cla = np.stack([fla_cla, t1ce_cla, t2_cla], axis=3)
    combined_arr_cla = combined_arr_cla[start:end, start:end, 13:141, :]

    mask = mask[start:end, start:end,13:141]
    mask[mask == 4] = 3
    mask = mask.astype(np.uint8)
    mask_c = to_categorical(mask,num_classes=4)
    if i not in val_images:
        for l in range(combined_arr_cla.shape[2]):
            for j in range(0, combined_arr_cla[:, :, l, :].shape[0], n):
                for k in range(0, combined_arr_cla[:, :, l, :].shape[1], n):
                    for m in range(combined_arr_cla[:, :, l, :].shape[2]):
                        combined_arr_cla[:, :, l, m] = clahe.apply(combined_arr_cla[:, :, l, m])
                    np.save('numpy_data/train_images/combined_min_' + str(h) + '.npy',combined_arr_min[j:j + n, k:k + n, l, :])
                    np.save('numpy_data/train_images/combined_z_' + str(h) + '.npy',combined_arr_z[j:j + n, k:k + n, l, :])
                    np.save('numpy_data/train_images/combined_cla_' + str(h) + '.npy',combined_arr_cla[j:j + n, k:k + n, l, :])
                    np.save('numpy_data/train_masks/mask_' + str(h) + '.npy', mask_c[j:j + n, k:k + n, l, :], )

                    h += 1
    else:
        for l in range(combined_arr_min.shape[2]):
            for j in range(0, combined_arr_min[:, :, l, :].shape[0], n):
                for k in range(0, combined_arr_min[:, :, l, :].shape[1], n):
                    for m in range(combined_arr_cla[:, :, l, :].shape[2]):
                        combined_arr_cla[:, :, l, m] = clahe.apply(combined_arr_cla[:, :, l, m])
                    np.save('numpy_data/val_images/combined_min_' + str(h) + '.npy',combined_arr_min[j:j + n, k:k + n, l, :])
                    np.save('numpy_data/val_images/combined_z_' + str(h) + '.npy',combined_arr_z[j:j + n, k:k + n, l, :])
                    np.save('numpy_data/val_images/combined_cla_' + str(h) + '.npy',combined_arr_cla[j:j + n, k:k + n, l, :])
                    np.save('numpy_data/val_masks/mask_' + str(h) + '.npy',mask_c[j:j + n, k:k + n, l, :])
                    h += 1
    print(i,end = '\t')