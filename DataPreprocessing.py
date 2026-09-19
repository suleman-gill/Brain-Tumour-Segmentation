import numpy as np
import nibabel as nib
import glob
import os
import argparse
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(description='Preprocess BraTS2020 dataset')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path to BraTS2020 training data directory (e.g., data/MICCAI_BraTS2020_TrainingData)')
parser.add_argument('--output-dir', type=str, default='numpy_data',
                    help='Path to output directory for numpy files (default: numpy_data)')
parser.add_argument('--show-plots', action='store_true',
                    help='Show visualization plots during preprocessing')
args = parser.parse_args()

# Ensure paths end with /
data_dir = args.data_dir.rstrip('/') + '/'
output_dir = args.output_dir.rstrip('/') + '/'

# Create output directories
os.makedirs(f'{output_dir}images', exist_ok=True)
os.makedirs(f'{output_dir}mask', exist_ok=True)
os.makedirs(f'{output_dir}val_images', exist_ok=True)
os.makedirs(f'{output_dir}val_masks', exist_ok=True)

n = list(str(np.random.randint(1,369)))
print(n)
lst = ['0','0','0']
lst[3-len(n):] = n
n = ''.join(lst)
print(n)
fla = nib.load(f'{data_dir}BraTS20_Training_{n}/BraTS20_Training_{n}_flair.nii').get_fdata()
t1ce = nib.load(f'{data_dir}BraTS20_Training_{n}/BraTS20_Training_{n}_t1ce.nii').get_fdata()
t2 = nib.load(f'{data_dir}BraTS20_Training_{n}/BraTS20_Training_{n}_t2.nii').get_fdata()
mask = nib.load(f'{data_dir}BraTS20_Training_{n}/BraTS20_Training_{n}_seg.nii').get_fdata()
print(fla.max(), fla.min())
mm = MinMaxScaler()
fla = mm.fit_transform(fla.reshape(-1,fla.shape[-1])).reshape(fla.shape)
print(fla.shape,'\t',fla.min(),'\t',np.max(fla))
t1ce = mm.fit_transform(t1ce.reshape(-1,t1ce.shape[-1])).reshape(t1ce.shape)
print(t1ce.shape,'\t',t1ce.min(),'\t',t1ce.max())
t2 = mm.fit_transform(t2.reshape(-1,t2.shape[-1])).reshape(t2.shape)
print(t2.shape , '\t',t2.min(),'\t',t2.max())

fig,axes = plt.subplots(2,2)
axes = axes.flatten()
axes[0].imshow(fla[:,:,70])
axes[0].title.set_text('Flair')
axes[1].imshow(t1ce[:,:,70])
axes[1].title.set_text('T1ce')
axes[2].imshow(t2[:,:,70])
axes[2].title.set_text('T2')
axes[3].imshow(mask[:,:,70])
axes[3].title.set_text('Mask')
if args.show_plots:
    plt.show()

combined_arr = np.stack([fla,t1ce,t2],axis = 3)
print(combined_arr.shape)
combined_arr = combined_arr[46:196,50:200,13:141,:]
mask = mask[46:196,50:200,13:141]

fig,axes = plt.subplots(2,2)
axes = axes.flatten()
axes[0].imshow(combined_arr[:,:,70-13,0])
axes[0].title.set_text('Flair')
axes[1].imshow(combined_arr[:,:,70-13,1])
axes[1].title.set_text('T1ce')
axes[2].imshow(combined_arr[:,:,70-13,2])
axes[2].title.set_text('T2')
axes[3].imshow(mask[:,:,70-13])
axes[3].title.set_text('Mask')
if args.show_plots:
    plt.show()

mask = mask.astype(np.int8)
print(np.unique(mask))
mask[mask == 4] = 3
print(np.unique(mask))
mask_ca = to_categorical(mask,num_classes=4)
print(mask.shape)


fla_paths = sorted(glob.glob(f'{data_dir}BraTS*/*flair.nii'))
print(fla_paths[:2])
t1ce_paths = sorted(glob.glob(f'{data_dir}BraTS*/*t1ce.nii'))
print(t1ce_paths[:2])
t2_paths = sorted(glob.glob(f'{data_dir}BraTS*/*t2.nii'))
print(t2_paths[:2])
seg_paths = sorted(glob.glob(f'{data_dir}BraTS*/*seg.nii'))
print(seg_paths[:2])
'''path = fla_paths[1].split('/')[-2].split('_')[-1]
path = 'numpy_data/images/combined_'+path+'.npy'
np.save(path,combined_arr)'''
np.random.seed(786)
val_images = np.random.randint(0,len(fla_paths),int(len(fla_paths)*0.2))
for i in range(len(fla_paths)):
    fla = nib.load(fla_paths[i]).get_fdata()
    t1ce = nib.load(t1ce_paths[i]).get_fdata()
    t2 = nib.load(t2_paths[i]).get_fdata()
    mask = nib.load(seg_paths[i]).get_fdata()
    fla = mm.fit_transform(fla.reshape(-1, fla.shape[-1])).reshape(fla.shape)
    t1ce = mm.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
    t2 = mm.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
    combined_arr = np.stack([fla, t1ce, t2], axis=3)
    combined_arr = combined_arr[46:196,50:200,13:141,:]
    mask = mask[46:196,50:200,13:141]
    mask = mask.astype(np.int8)
    mask[mask == 4] = 3
    mask_ca = to_categorical(mask, num_classes=4)
    path_n = fla_paths[i].split('/')[-2].split('_')[-1]
    if i not in val_images:
        path = f'{output_dir}images/combined_{path_n}.npy'
        path_mask = f'{output_dir}mask/mask_{path_n}.npy'
    else:
        path = f'{output_dir}val_images/combined_{path_n}.npy'
        path_mask = f'{output_dir}val_masks/mask_{path_n}.npy'
    np.save(path,combined_arr)
    np.save(path_mask,mask_ca)
'''
fla_paths = sorted(glob.glob('data/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_*/*_flair.nii'))
t1ce_paths = sorted(glob.glob('data/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_*/*_t1ce.nii'))
t2_paths = sorted(glob.glob('data/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_*/*_t2.nii'))
for i in range(len(fla_paths)):
    fla = nib.load(fla_paths[i]).get_fdata()
    t1ce = nib.load(t1ce_paths[i]).get_fdata()
    t2 = nib.load(t2_paths[i]).get_fdata()
    fla = mm.fit_transform(fla.reshape(-1, fla.shape[-1])).reshape(fla.shape)
    t1ce = mm.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
    t2 = mm.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
    combined_arr = np.stack([fla, t1ce, t2], axis=3)
    combined_arr = combined_arr[34:226, 34:226, 13:141, :]
    path_n = fla_paths[i].split('/')[-2].split('_')[-1]
    path = 'numpy_data/test_images/combined_'+path_n+'.npy'
    np.save(path,combined_arr)
'''
