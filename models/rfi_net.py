import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
# from model_config import n_layers, n_filters
from .helper import generic_block, generic_unet, GenericBlock, GenericUnet

tf.keras.backend.set_floatx('float32')


def RFINET_downblock(input_tensor, n_filters, kernel_size=3, batchnorm=True, stride=(1, 1)):
    # first layer
    x0 = layers.Conv2D(filters=n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(input_tensor)

    x0 = layers.BatchNormalization()(x0)
    x0 = layers.Activation('relu')(x0)

    x1 = layers.Conv2D(filters=2 * n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x0)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters=2 * n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)

    skip = layers.Conv2D(filters=2 * n_filters,
                         kernel_size=(1, 1), \
                         kernel_initializer='he_normal',
                         strides=stride,
                         padding='same')(input_tensor)
    skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x2, skip])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def RFINET_upblock(input_tensor, n_filters, kernel_size=3, batchnorm=True, stride=(1, 1)):
    x0 = layers.Conv2D(filters=n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(input_tensor)

    x0 = layers.BatchNormalization()(x0)
    x0 = layers.Activation('relu')(x0)

    x1 = layers.Conv2D(filters=n_filters // 2,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x0)

    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters=n_filters // 2,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)

    skip = layers.Conv2D(filters=n_filters // 2,
                         kernel_size=(1, 1),
                         kernel_initializer='he_normal',
                         strides=stride,
                         padding='same')(input_tensor)
    skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x2, skip])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def RFI_NET_mesarcik(args, n_filters=32, dropout=0.05, batchnorm=True):
    """
    Implemented by Mesarcik
    27 Deep residual detection of Radio Frequency Interference for FAST
    Zhicheng Yang, Ce Yu, Jian Xiao, and Bo Zhang
    2020
    """
    # Contracting Path
    input_data = tf.keras.Input(args.input_shape, name='data')
    c0 = layers.Conv2D(filters=32,
                       kernel_size=(3, 3),
                       kernel_initializer='he_normal',
                       strides=1,
                       padding='same')(input_data)

    c1 = RFINET_downblock(c0, n_filters * 1, kernel_size=3, batchnorm=batchnorm, stride=(1, 1))
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(p1)

    c2 = RFINET_downblock(p1, n_filters * 2, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(p2)

    c3 = RFINET_downblock(p2, n_filters * 4, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(dropout)(p3)

    c4 = RFINET_downblock(p3, n_filters * 8, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(dropout)(p4)

    c5 = RFINET_downblock(p4, n_filters * 16, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    # p5 = layers.MaxPooling2D((2, 2))(c5)
    # p5 = layers.Dropout(dropout)(p5)

    # upsampling
    u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = RFINET_upblock(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = RFINET_upblock(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = RFINET_upblock(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = RFINET_upblock(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model


def RFI_NET_gen_old(args, n_filters=32, height=5, dropout=0.05):
    """
    Charls remplementation of Mesarciks version of RFI_NET
    27 Deep residual detection of Radio Frequency Interference for FAST
    Zhicheng Yang, Ce Yu, Jian Xiao, and Bo Zhang
    2020
    """
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int)  # - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    x = layers.Conv2D(filters=n_filters,
                      kernel_size=3,
                      kernel_initializer='he_normal',
                      strides=1,
                      padding='same')(input_data)

    x = generic_unet(x,
                     height,
                     n_filters,
                     dropout=dropout,
                     down_func=down_func,
                     mid_func=down_func,
                     up_func=up_func)

    x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model

def RFI_NET(args):
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int)  # - 1  # why -1?  minimum tensor size 2x2
    height = max_height if args.height is None else np.minimum(args.height, max_height)

    x = layers.Conv2D(filters=args.filters,
                      kernel_size=3,
                      kernel_initializer='he_normal',
                      strides=1,
                      padding='same')(input_data)
    level_block = RFINETBlock(filters=args.filters)
    x = GenericUnet(height, level_block=level_block)(x)

    x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


class RFINETBlock(GenericBlock):

    def __call__(self, input_tensor, level=0, skip_tensor=None, direction='down'):
        f_mult = 2 if direction == 'down' or direction == 'mid' else 0.5
        filters = self.filters,
        n = GenericBlock('n')(input_tensor, skip_tensor=skip_tensor)
        x = GenericBlock('cba cba cb',
                         [filters, filters*f_mult, filters*f_mult])(n, level=level)
        x = GenericBlock('cb p ba',
                         [filters, filters*f_mult, filters*f_mult])(n, level=level, skip_tensor=x)
        return x


def up_func(input_tensor, n_filters, level_id, skip_tensor, kernel_size=3, strides=1):

    n = generic_block(input_tensor,
                      'tn',
                      n_filters,
                      kernel_size=3,
                      strides=2,
                      skip_tensor=skip_tensor,
                      level_id=level_id)

    x = generic_block(n,
                      'cba cba cb',
                      [n_filters, n_filters // 2, n_filters // 2],
                      kernel_size=kernel_size,
                      strides=strides,
                      level_id=level_id)

    x = generic_block(n,
                      'cb p ba',
                      n_filters // 2,
                      skip_tensor=x,
                      kernel_size=1,
                      strides=strides,
                      level_id=level_id)
    return x


def down_func(input_tensor, n_filters, kernel_size=3, strides=1, level_id=0):

    x = generic_block(input_tensor,
                      'cba cba cb',
                      [n_filters, n_filters * 2, n_filters * 2],
                      kernel_size=kernel_size,
                      strides=strides,
                      level_id=level_id)
    x = generic_block(input_tensor,
                      'cb p ba',
                      n_filters * 2,
                      skip_tensor=x,
                      kernel_size=1,
                      strides=strides,
                      level_id=level_id)
    return x
