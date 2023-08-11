import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
#from model_config import n_layers, n_filters

# import sys
# sys.path.insert(1, '../')
from .generic_builder import GenericUnet, GenericBlock

tf.keras.backend.set_floatx('float32')


def UNET(input_shape, height, filters, dropout, kernel_regularizer, level_blocks, activation, final_activation, bn_first, **kwargs):
    input_data = tf.keras.Input(input_shape, name='data')
    if bn_first:
        input_data = GenericBlock('b')(input_data)
    max_height = np.floor(np.log2(input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    level_block = GenericBlock('ncbad', filters,
                               blocks=level_blocks,
                               activation=activation,
                               dropout=dropout,
                               kernel_regularizer=kernel_regularizer)
    x = GenericUnet(height, level_block=level_block)(input_data)

    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation=final_activation)(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


def freeze_UNET(model: tf.keras.Model):
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'trainable'):
            model.layers[i].trainable = False
    model.layers[-2].trainable = True # last conv
    model.layers[-5].trainable = True # last BN
    model.layers[-6].trainable = True  # 2nd last conv
    return model
