import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
#from model_config import n_layers, n_filters

# import sys
# sys.path.insert(1, '../')
from .generic_builder import GenericUnet, GenericBlock

tf.keras.backend.set_floatx('float32')


def UNET(input_shape, height, filters, dropout, kernel_regularizer, level_blocks, final_activation, **kwargs):
    input_data = tf.keras.Input(input_shape, name='data')
    max_height = np.floor(np.log2(input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    level_block = GenericBlock('ncbad', filters,
                               blocks=level_blocks,
                               dropout=dropout,
                               kernel_regularizer=kernel_regularizer)
    x = GenericUnet(height, level_block=level_block)(input_data)

    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation=final_activation)(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model
