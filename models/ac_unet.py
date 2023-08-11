import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
# from model_config import n_layers, n_filters

from .generic_builder import GenericBlock, GenericUnet

tf.keras.backend.set_floatx('float32')

def AC_UNET(input_shape, height, filters, dropout, kernel_regularizer, level_blocks, activation, final_activation, dilation_rate, bn_first, **kwargs):
    """
    Charl's implementation according to the article:
    0 Radio frequency interference detection based on the AC-UNet model
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
    input_data = tf.keras.Input(input_shape, name='data')
    if bn_first:
        input_data = GenericBlock('b')(input_data)
    max_height = np.floor(np.log2(input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    dilation_rate = dilation_rate if dilation_rate > 1 else 2  # pretty pointless to have dr of 1

    level_block = GenericBlock('ncbad', filters, activation=activation, blocks=level_blocks, dilation_rate=dilation_rate,
                               dropout=dropout, kernel_regularizer=kernel_regularizer)
    x = GenericUnet(height, level_block=level_block)(input_data)

    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation=final_activation)(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model

def freeze_AC_UNET(model: tf.keras.Model):
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'trainable'):
            model.layers[i].trainable = False
    model.layers[-2].trainable = True # last conv
    model.layers[-5].trainable = True # last BN
    model.layers[-6].trainable = True  # 2nd last conv
    return model