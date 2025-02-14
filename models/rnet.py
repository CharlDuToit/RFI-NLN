import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters
from .generic_builder import  GenericBlock

tf.keras.backend.set_floatx('float32')

def RNET_mesarcik(args):
    """
    Implemented by Mesarcik
    14 Deep Learning improves identification of Radio Frequency Interference 2020
    Alireza Vafaei Sadr, Bruce A. Bassett3, Nadeem Oozeer, Yabebal Fantaye and Chris Finlay
    2 shortcut connections for 7 layers
    1 shortcut connection for 5 and 6 layers
    0 shortcut connections for 3 layers
    Authors believe R-Net5 and R-Net6 is the best, but Mesarcick implemented R-Net7 in the code below

    used MSE and BCE and find MSE to be better
    """
    input_data = tf.keras.Input(args.input_shape, name='data')

    xp = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(input_data)
    x = layers.BatchNormalization()(xp)
    x = tf.nn.relu(x)
    x1 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = tf.nn.relu(x1)
    x2 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = tf.nn.relu(x2)

    x3 = x2 + xp

    x4 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = tf.nn.relu(x4)

    x6 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x4)
    x6 = layers.BatchNormalization()(x6)
    x6 = tf.nn.relu(x6)

    x7 = x6 + x3

    x8 = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(x7)
    x8 = layers.BatchNormalization()(x8)
    x8 = tf.nn.relu(x8)

    # Mesarcic had kernel_size 5, Charl changed it back to 1
    # Mesarcic had relu as final activation, charl changed to sigmoid.
    # Paper 14 never mentions any activation function excpet RELU,
    x_out = layers.Conv2D(filters=1, kernel_size=1, strides=(1, 1), padding='same', activation='sigmoid')(x8)

    model = tf.keras.Model(inputs=[input_data], outputs=[x_out])
    return model


def RNET(input_shape, filters, dropout, kernel_regularizer, activation, final_activation, bn_first, **kwargs):
    """
    Charl's reimplementation of Mesarcik's RNET
    """
    input_data = tf.keras.Input(input_shape, name='data')
    if bn_first:
        input_data = GenericBlock('b')(input_data)
    xp = layers.Conv2D(filters=filters, kernel_size=5, strides=(1, 1), padding='same')(input_data)
    x3 = GenericBlock('ba cba cbad p',
                      filters,
                      kernel_size=5,
                      activation=activation,
                      strides=1,
                      dropout=dropout,
                      kernel_regularizer=kernel_regularizer)(xp, skip_tensor=xp)
    x7 = GenericBlock('cba cbad p',
                      filters,
                      kernel_size=5,
                      activation=activation,
                      strides=1,
                      kernel_regularizer=kernel_regularizer,
                      dropout=dropout)(x3, skip_tensor=x3)
    # Mesarcic had kernel_size 5, Charl changed it back to 1
    # Mesarcic had relu as final activation, charl changed to sigmoid.
    # Paper 14 never mentions any activation function excpet RELU,

    #"After each convolutional layer (except the last one)
    #we find that inserting batch normalization layer and using
    #RELU activation functions to be optimal"

    x = GenericBlock('cba',
                     filters,
                     kernel_size=5,
                     strides=1,
                     activation=activation,
                     kernel_regularizer=kernel_regularizer)(x7)
    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation=final_activation)(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model

def freeze_RNET(model: tf.keras.Model):
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'trainable'):
            model.layers[i].trainable = False
    model.layers[-2].trainable = True # last conv
    # model.layers[-4].trainable = True # last BN
    model.layers[-5].trainable = True  # last BN
    return model

