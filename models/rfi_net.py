import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
# from model_config import n_layers, n_filters
from .generic_builder import GenericBlock, GenericUnet

tf.keras.backend.set_floatx('float32')


def RFI_NET(input_shape, height, filters, dropout, kernel_regularizer, final_activation, **kwargs):
    input_data = tf.keras.Input(input_shape, name='data')
    max_height = np.floor(np.log2(input_shape[0])).astype(int)  # - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    #x = GenericBlock('cba cba', args.filters, kernel_size=3, strides=1)(input_data) # look at table 1
    x = input_data

    level_block = RFINETBlock(filters=filters, dropout=dropout, kernel_regularizer=kernel_regularizer)  # note *2
    x = GenericUnet(height, level_block=level_block)(x)

    # removed batchnorm from table 1, i.e. no longer cba
    x = GenericBlock('ca', 1, activation=final_activation, kernel_size=1, strides=1)(x)  # look at table 1
    #x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x) # table 1 used 3x3 conv with softmax activation
    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


class RFINETBlock(GenericBlock):

    def __call__(self, input_tensor, level=0, skip_tensor=None, direction='down'):
        filters = self.filters
        if direction == 'up':
            n = GenericBlock('n')(input_tensor, skip_tensor=skip_tensor)
        else:
            n = input_tensor
        x = GenericBlock('cba cba cb', filters, kernel_regularizer=self.kernel_regularizer)(n, level=level)
        x = GenericBlock('cb p bad', filters,
                         dropout=self.dropout,
                         kernel_regularizer=self.kernel_regularizer)(n, level=level, skip_tensor=x)
        return x
