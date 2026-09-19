import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Input, Multiply, Add, Activation
from tensorflow.keras.models import Model

# Attention Gate
def attention_gate(feature_map, gating_signal, filters):
    """
    Implements an attention gate.
    Args:
        feature_map: Feature map from the encoder (skip connection).
        gating_signal: Gating signal from the decoder.
        filters: Number of filters for the intermediate convolutions.
    Returns:
        Attention-weighted feature map.
    """
    # Transform the feature map
    theta_x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(feature_map)  # No downsampling
    theta_x = Activation('relu')(theta_x)

    # Transform the gating signal
    phi_g = Conv2D(filters, (1, 1), padding='same')(gating_signal)
    phi_g = Activation('relu')(phi_g)
    # Upsample the gating signal to match the feature map size
    upsample_g = UpSampling2D(size=(2, 2))(phi_g)

    # Add the transformed feature map and upsampled gating signal
    add = Add()([theta_x, upsample_g])
    add = Activation('relu')(add)

    # Compute attention coefficients
    psi = Conv2D(1, (1, 1), padding='same')(add)
    psi = Activation('sigmoid')(psi)

    # Apply attention to the feature map
    attention = Multiply()([feature_map, psi])

    return attention

# Attention U-Net Model
def attention_unet(input_shape=(192, 192, 3), num_classes=4,num_fil = 64):
    """
    Builds an Attention U-Net model.
    Args:
        input_shape: Shape of the input image (height, width, channels).
        num_classes: Number of output classes.
    Returns:
        A TensorFlow/Keras model.
    """
    inputs = Input(input_shape)

    # Encoder
    c1 = Conv2D(num_fil, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(num_fil, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(num_fil*2, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(num_fil*2, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(num_fil*4, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(num_fil*4, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(num_fil*8, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(num_fil*8, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = Conv2D(num_fil*16, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(num_fil*16, (3, 3), activation='relu', padding='same')(c5)
    # Decoder with Attention Gates
    u6 = UpSampling2D((2, 2))(c5)
    att6 = attention_gate(c4, c5, filters=num_fil*8)
    u6 = Concatenate()([u6, att6])
    c6 = Conv2D(num_fil*8, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(num_fil*8, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    att7 = attention_gate(c3, c6, filters=num_fil*4)
    u7 = Concatenate()([u7, att7])
    c7 = Conv2D(num_fil*4, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(num_fil*4, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    att8 = attention_gate(c2, c7, filters=num_fil*2)
    u8 = Concatenate()([u8, att8])
    c8 = Conv2D(num_fil*2, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(num_fil*2, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    att9 = attention_gate(c1, c8, filters=num_fil)
    u9 = Concatenate()([u9, att9])
    c9 = Conv2D(num_fil, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(num_fil, (3, 3), activation='relu', padding='same')(c9)

    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)

    model = Model(inputs, outputs)
    return model

# Build the model
model = attention_unet(input_shape=(192, 192, 3), num_classes=4)
model.summary()