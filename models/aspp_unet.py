import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

from .generic_builder import GenericBlock, GenericUnet

tf.keras.backend.set_floatx('float32')

def ASPP_UNET(input_shape, height, filters, dropout, kernel_regularizer, level_blocks, activation, final_activation, dilation_rate, bn_first, **kwargs):
    """

    """
    input_data = tf.keras.Input(input_shape, name='data')
    if bn_first:
        input_data = GenericBlock('b')(input_data)
    max_height = np.floor(np.log2(input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    dilation_rate = dilation_rate if dilation_rate > 1 else 2  # pretty pointless to have dr of 1

    level_block = ASPPBlock(filters=filters, activation=activation, blocks=level_blocks, dilation_rate=dilation_rate,
                            dropout=dropout, kernel_regularizer=kernel_regularizer)
    x = GenericUnet(height, level_block=level_block)(input_data)

    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation=final_activation)(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


class ASPPBlock(GenericBlock):

    def __call__(self, input_tensor, level=0, skip_tensor=None, direction='down'):
        n = GenericBlock('n')(input_tensor, skip_tensor=skip_tensor)
        p1 = GenericBlock('cbad',
                          self.filters//2,
                          dropout=self.dropout,
                          activation=self.activation,
                          kernel_regularizer=self.kernel_regularizer)(n, level=level)
        p2 = GenericBlock('cbad',
                          self.filters//2,
                          dilation_rate=self.dilation_rate,
                          dropout=self.dropout,
                          activation=self.activation,
                          kernel_regularizer=self.kernel_regularizer)(n, level=level)
        x = layers.concatenate([p1, p2])
        return x


def freeze_ASPP_UNET(model: tf.keras.Model):
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'trainable'):
            model.layers[i].trainable = False
    model.layers[-2].trainable = True  # last conv
    model.layers[-8].trainable = True # BN
    model.layers[-9].trainable = True  # BN
    model.layers[-10].trainable = True
    model.layers[-11].trainable = True
    return model