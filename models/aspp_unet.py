import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

from .generic_builder import GenericBlock, GenericUnet

tf.keras.backend.set_floatx('float32')

def ASPP_UNET(input_shape, height, filters, dropout, kernel_regularizer, level_blocks, final_activation, dilation_rate, **kwargs):
    """

    """
    input_data = tf.keras.Input(input_shape, name='data')
    max_height = np.floor(np.log2(input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    dilation_rate = dilation_rate if dilation_rate > 1 else 2  # pretty pointless to have dr of 1

    level_block = ASPPBlock(filters=filters, blocks=level_blocks, dilation_rate=dilation_rate,
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
                          kernel_regularizer=self.kernel_regularizer)(n, level=level)
        p2 = GenericBlock('cbad',
                          self.filters//2,
                          dilation_rate=self.dilation_rate,
                          dropout=self.dropout,
                          kernel_regularizer=self.kernel_regularizer)(n, level=level)
        x = layers.concatenate([p1, p2])
        return x
