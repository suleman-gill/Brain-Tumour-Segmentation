import segmentation_models_3D as sm
import tensorflow as tf
from DoubleAttention1024 import attention_unet
from image_loader import image_loader
import keras.saving
import os
import glob
import shutil
t_image_dir = '../numpy_data/train_images/'
t_mask_dir = '../numpy_data/train_masks/'
v_image_dir = '../numpy_data/val_images/'
v_mask_dir = '../numpy_data/val_masks/'
t_images = sorted(glob.glob(t_image_dir + '*cla*.npy'))
t_masks = sorted(os.listdir(t_mask_dir))
v_images = sorted(glob.glob(v_image_dir + '*cla*.npy'))
v_masks = sorted(os.listdir(v_mask_dir))
v_images = [i.split('/')[-1] for i in v_images]
t_images = [i.split('/')[-1] for i in t_images]

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
def dice_bce_loss(y_true, y_pred, alpha=0.5):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return alpha * dice + (1 - alpha) * bce


optim = tf.keras.optimizers.Adam(learning_rate= 1e-4)

metrics = [
    'accuracy',
    sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore()
]

batchs = 5
os.makedirs('Attention_final_model_192_clahe_1024',exist_ok=True)

image_datagen = image_loader(t_image_dir,t_images,t_mask_dir,t_masks,batch_size=batchs)
val_datagen = image_loader(v_image_dir,v_images,v_mask_dir,v_masks,batch_size=batchs)

model = attention_unet((192,192,3),4)
model.compile(optimizer = optim,loss = dice_bce_loss,metrics = metrics)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.CSVLogger('Attention_final_model_192_clahe_1024/training_log1.csv'),
    tf.keras.callbacks.ModelCheckpoint(
        'Attention_final_model_192_clahe_1024/best_model1.h5',save_freq= int(5 * 2448)  # Use `.keras` instead of `.h5`
    )
]
model.fit(
    image_datagen,
    steps_per_epoch=len(t_images) // batchs,
    validation_data=val_datagen,
    epochs=40,  # As per image
    verbose=1,
    validation_steps=len(t_images) // batchs,
    callbacks=callbacks
)
# Save the entire model
keras.saving.save_model(model, 'Attention_final_model_192_clahe_1024/my_model.keras')
model.save('Attention_final_model_192_clahe_1024/my_model.h5')
 # Saves as a directory

