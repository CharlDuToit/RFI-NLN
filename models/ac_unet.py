import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

from .helper import generic_block, generic_unet

tf.keras.backend.set_floatx('float32')


def AC_UNET_old(args,
                n_filters=64,
                dilation_rate=7,
                dropout=0.05,
                batchnorm=True,  # not mentioned in article
                height=3):
    """
    Charl's implementation according to the article:
    0Radio frequency interference detection based on the AC-UNet model
    Rui-Qing-Yan
    Accepted 2020 December

    Upsampling as mentioned in article is assumed to be Conv2DTranspose. Look at fig5's first Up-conv, 2x2 and the
    AC,3x3 relu just before. The AC would produce 256 filters, and Up-conv does not change the number of filters.
    So in the concatenated layer, there would be 128+256 filters, but 256 is indicated.
    Either the AC is with 128 filters followed by UpConv, or the AC is 256 filters followed by Conv2DTranspose with 128
    filters

    21 layer 100x100 patch
    14 AC layers - fig 5 shows 10, AC with 1x1 kernel is not possible
    two-layer 2x2 maxpool layer stride 2
    two-layer fusion layers
    1 dropout layer lastly somewhere?
    two-layer upsampling layers - see above paragraph
    """
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int)  # - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    f_mult = 1
    down_blocks = [None] * (height - 1)  # [None] *(3-1) : 32, 16, belly 8
    x = input_data

    for lid in range(0, height - 1, 1):
        x = generic_block(x, 'caca', n_filters * f_mult, strides=1, dilation_rate=dilation_rate)
        down_blocks[lid] = x
        x = generic_block(x, 'm')
        f_mult *= 2

    x = generic_block(x, 'caca', n_filters * f_mult, strides=1)

    for lid in range(lid, -1, -1):  # 0..4, 5 values
        f_mult /= 2
        x = generic_block(x,
                          'tn caca',  # layer_chars,
                          n_filters * f_mult,
                          skip_tensor=down_blocks[lid],
                          strides=[2, 1],  # strides,
                          # dropout=dropout
                          dilation_rate=[1, dilation_rate])

    x = generic_block(x, 'dc', 1, strides=1, kernel_size=1, dropout=dropout)
    x = layers.Activation('sigmoid')(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


def AC_UNET(args,
            n_filters=64,
            dilation_rate=7,
            dropout=0.05,
            batchnorm=True,  # not mentioned in article
            height=3):
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int)  # - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    down_kwargs = dict(layer_chars='cba cba', dilation_rate=dilation_rate, dropout=dropout)
    up_kwargs = dict(layer_chars='tn cbacba', strides=[2, 1], dilation_rate=[1, dilation_rate], dropout=dropout)
    x = generic_unet(input_data, height, n_filters, down_kwargs=down_kwargs, mid_kwargs=down_kwargs, up_kwargs=up_kwargs)

    x = generic_block(x, 'dca', 1, strides=1, kernel_size=1, dropout=dropout, activation='sigmoid')

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model
