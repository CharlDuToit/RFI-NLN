import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

# import sys
# sys.path.insert(1, '../')
from .generic_builder import GenericUnet, GenericBlock

tf.keras.backend.set_floatx('float32')

def conv2D_block(input_tensor, n_filters, kernel_size=3, strides=1, dilation_rate=1, dropout=0.0, batchnorm=True):
    # dilation_rate > 1 requires strides=1
    x = layers.Conv2D(filters=n_filters,
                      kernel_size=kernel_size,
                      kernel_initializer='he_normal',
                      strides=strides,
                      dilation_rate=dilation_rate,
                      padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if 0.0 < dropout < 1.0:
        x = layers.Dropout(dropout)(x)

    return x

def UNET_Mesarcik(args, n_filters=16, dropout=0.05, batchnorm=True):
    # Contracting Path
    input_data = tf.keras.Input(args.input_shape, name='data')
    if args.input_shape[0] == 16:
        _str = 1  # cant downsample 16x16 patches
    else:
        _str = 2
    c1 = conv2D_block(input_data,
                      n_filters * 1,
                      kernel_size=3,
                      batchnorm=batchnorm,
                      strides=(_str, _str))
    # p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(c1)

    c2 = conv2D_block(p1,
                      n_filters * 2,
                      kernel_size=3,
                      strides=(2, 2),
                      batchnorm=batchnorm)
    # p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(c2)

    if args.input_shape[1] > 8:
        c3 = conv2D_block(p2,
                          n_filters * 4,
                          kernel_size=3,
                          strides=(2, 2),
                          batchnorm=batchnorm)
        # p3 = layers.MaxPooling2D((2, 2))(c3)
        p3 = layers.Dropout(dropout)(c3)
    else:
        p3 = p2

    if args.input_shape[1] > 16:
        c4 = conv2D_block(p3,
                          n_filters * 8,
                          kernel_size=3,
                          strides=(2, 2),
                          batchnorm=batchnorm)
        # p4 = layers.MaxPooling2D((2, 2))(c4)
        p4 = layers.Dropout(dropout)(c4)
    else:
        p4 = p3

    # c5 = conv2D_block(p4, n_filters=n_filters * 8, kernel_size=3, strides=(2, 2), batchnorm=batchnorm)
    c5 = conv2D_block(p4, n_filters=n_filters * 16, kernel_size=3, strides=(2, 2), batchnorm=batchnorm)

    # Expansive Path
    if args.input_shape[1] > 16:
        u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        u6 = layers.Dropout(dropout)(u6)
        c6 = conv2D_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    else:
        c6 = c5

    if args.input_shape[1] > 8:
        u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        u7 = layers.Dropout(dropout)(u7)
        c7 = conv2D_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    else:
        c7 = c6

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = conv2D_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(dropout)(u9)
    if args.input_shape[0] != 16:  # cant downsample 16x16 patches
        u9 = layers.UpSampling2D((2, 2))(u9)
    c9 = conv2D_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model

def UNET(args):
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if args.height is None else np.minimum(args.height, max_height)

    level_block = GenericBlock('ncba', args.filters, blocks=args.level_blocks)
    x = GenericUnet(height, level_block=level_block)(input_data)

    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model
