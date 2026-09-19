import pandas as pd
import numpy as np
import tensorflow as tf
import os
import segmentation_models_3D as sm


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_bce_loss(y_true, y_pred, alpha=0.5):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return alpha * dice + (1 - alpha) * bce

dice_loss = sm.losses.DiceLoss(class_weights=[0.25, 0.25, 0.25, 0.25])
focal = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1.0 *focal)


model = tf.keras.models.load_model(
    'Attention_final_model_192_clahe/best_model1.h5',
    custom_objects={
        'IOUScore': sm.metrics.IOUScore(),
        'FScore': sm.metrics.FScore()
    }
    ,compile = False
)


arr = np.expand_dims(np.load('../numpy_data/val_images/combined_cla_454.npy').astype('float64'), axis=0)
pred = model.predict(arr)

pred = pred.reshape(pred.shape[1:])
pred = np.argmax(pred, axis=2)
print(pred.shape)
y_true = np.load('../numpy_data/val_masks/mask_454.npy').astype('float64').argmax(axis=2)

import matplotlib.pyplot as plt
plt.imshow(pred)
plt.show()
plt.imshow(y_true)
plt.show()
