import tensorflow as tf
import numpy as np
from keras import layers
#from model_config import n_layers, n_filters
from .generic_builder import GenericBlock

tf.keras.backend.set_floatx('float32')


def RNET5(input_shape, filters, dropout, kernel_regularizer, activation, final_activation, bn_first, **kwargs):
    """
    Charl's reimplementation of RNET5
    """
    input_data = tf.keras.Input(input_shape, name='data')
    if bn_first:
        input_data = GenericBlock('b')(input_data)
    xp = layers.Conv2D(filters=filters, kernel_size=5, strides=(1, 1), padding='same')(input_data)
    x3 = GenericBlock('ba cba cbad p',
                      filters,
                      kernel_size=5,
                      strides=1,
                      activation=activation,
                      dropout=dropout,
                      kernel_regularizer=kernel_regularizer)(xp, skip_tensor=xp)

    x = GenericBlock('cba',
                     filters,
                     kernel_size=5,
                     strides=1,
                     activation=activation,
                     kernel_regularizer=kernel_regularizer)(x3)
    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation=final_activation)(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


def freeze_RNET5(model: tf.keras.Model):
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'trainable'):
            model.layers[i].trainable = False
    model.layers[-2].trainable = True # last conv
    model.layers[-4].trainable = True # last BN
    model.layers[-5].trainable = True  #
    return model

"""m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
 loss=tf.keras.losses.BinaryCrossentropy())"""

