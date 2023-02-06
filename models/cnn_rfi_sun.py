import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

tf.keras.backend.set_floatx('float32')


def CNN_RFI_SUN_block(input_tensor,
                      n_filters,
                      conv_kernel_size=(3, 3),
                      # conv_stride=(1,1),
                      pool_size=(2, 2),
                      pool_stride=(1, 1),
                      batchnorm=True):
    """
    Charl's implementation according to article:
    12 A Robust RFI Identification For Radio Interferometry based on a Convolutional Neural Network
    Haomin Sun , Hui Deng, Feng Wang, Ying Mei, Tingting Xu, Oleg Smirnov, Linhua Deng, and Shoulin Wei
    2022
    Padding is always 1 on all 4 sides."""

    x = layers.ZeroPadding2D(padding=1)(input_tensor)  # pad 1 zero on all 4 sides

    x = layers.Conv2D(filters=n_filters,
                      kernel_size=conv_kernel_size,
                      # kernel_initializer='he_normal',
                      strides=1,  # conv_stride,
                      padding='valid')(x)

    if batchnorm:
        x = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
        # x = layers.BatchNormalization()(x) # tf defaults

    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=pool_size,
                            strides=pool_stride,
                            padding='valid')(x)
    return x


def CNN_RFI_SUN(dropout=0.0, **kwargs):
    """Tensorflow implementation of
    https://github.com/astronomical-data-processing/CNN_RFI.git."""
    # torch
    # expected input: (C, H, W)
    # actual input: (H, W, C) = (1, 1, 2) with C[0]:amp, C[1]:phase
    # first Conv2D: torch.nn.Conv2d(1, 64, kernel_size=(1, 2), padding=1)
    #
    # tf input: (H, W, C) with data_format ='channels_last'
    #
    # Actual input shape might be (1,1,2)
    input_data = tf.keras.Input((1, 2, 1), name='data')

    c1 = CNN_RFI_SUN_block(input_data, 64, conv_kernel_size=(1, 2), pool_size=(1, 1), pool_stride=1)  # 64 3 3
    c2 = CNN_RFI_SUN_block(c1, 64, conv_kernel_size=(1, 2), pool_size=(1, 2), pool_stride=1)  # 64 5 3
    c3 = CNN_RFI_SUN_block(c2, 128, conv_kernel_size=3, pool_size=(1, 2), pool_stride=1)  # 128 5 3
    c4 = CNN_RFI_SUN_block(c3, 128, conv_kernel_size=3, pool_size=2, pool_stride=1)  # 128 4 1
    # c5 = CNN_RFI_SUN_block(c4, 1024, conv_kernel_size=3, pool_size=3, pool_stride=2)
    # c6 = CNN_RFI_SUN_block(c5, 1024, conv_kernel_size=3, pool_size=3, pool_stride=2)

    f = layers.Flatten()(c4)  # should apparently flatten to 512 nodes
    d1 = layers.Dense(1024, activation='relu')(f)
    do = layers.Dropout(dropout)(d1)
    #outputs = layers.Dense(2, activation='softmax')(do)  # replace with 1 channel and sigmoid?
    outputs = layers.Dense(1, activation='sigmoid')(do)  # replace with 1 channel and sigmoid?

    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model
