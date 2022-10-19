import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters
from .helper import generic_block, GenericBlock

tf.keras.backend.set_floatx('float32')

def RNET_old(args):
    """
    Implemented by Mesarcik
    14 Deep Learning improves identification of Radio Frequency Interference 2020
    Alireza Vafaei Sadr, Bruce A. Bassett3, Nadeem Oozeer, Yabebal Fantaye and Chris Finlay
    2 shortcut connections for 7 layers
    1 shortcut connection for 5 and 6 layers
    0 shortcut connections for 3 layers
    Authors believe R-Net5 and R-Net6 is the best, but Mesarcick implemented R-Net7 in the code below
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

    x_out = layers.Conv2D(filters=1, kernel_size=5, strides=(1, 1), padding='same', activation=tf.nn.relu)(x8)

    model = tf.keras.Model(inputs=[input_data], outputs=[x_out])
    return model


def RNET_gen_old(args):
    """
    Charl's reimplementation of Mesarcik's RNET  -- yields exactly the same model
    """
    input_data = tf.keras.Input(args.input_shape, name='data')
    xp = layers.Conv2D(filters=12, kernel_size=5, strides=(1, 1), padding='same')(input_data)
    x3 = generic_block(xp, 'ba cba cba p', 12, skip_tensor=xp, kernel_size=5, strides=1)
    x7 = generic_block(x3, 'cba cba p', 12, skip_tensor=x3, kernel_size=5, strides=1)
    x = generic_block(x7, 'cba ca', [12, 1], kernel_size=5, strides=1)
    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


def RNET(args):
    """
    Charl's reimplementation of Mesarcik's RNET  -- yields exactly the same model
    """
    input_data = tf.keras.Input(args.input_shape, name='data')
    xp = layers.Conv2D(filters=args.filters, kernel_size=5, strides=(1, 1), padding='same')(input_data)
    x3 = GenericBlock('ba cba cba p', args.filters, kernel_size=5, strides=1)(xp, skip_tensor=xp)
    x7 = GenericBlock('cba cba p', args.filters, kernel_size=5, strides=1)(x3, skip_tensor=x3)
    x = GenericBlock('cba ca', [args.filters, 1], kernel_size=5, strides=1)(x7)
    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model

