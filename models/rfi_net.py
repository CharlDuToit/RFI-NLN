import tensorflow as tf
import numpy as np
# from tensorflow.keras import layers
# from model_config import n_layers, n_filters
from .generic_builder import GenericBlock, GenericUnet

tf.keras.backend.set_floatx('float32')


def RFI_NET(input_shape, height, filters, dropout, kernel_regularizer, activation, final_activation, bn_first, **kwargs):
    input_data = tf.keras.Input(input_shape, name='data')
    if bn_first:
        input_data = GenericBlock('b')(input_data)
    max_height = np.floor(np.log2(input_shape[0])).astype(int)  # - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    #x = GenericBlock('cba cba', args.filters, kernel_size=3, strides=1)(input_data) # look at table 1
    x = input_data

    level_block = RFINETBlock(filters=filters, activation=activation, dropout=dropout,
                              kernel_regularizer=kernel_regularizer)  # note *2
    x = GenericUnet(height, level_block=level_block)(x)

    # removed batchnorm from table 1, i.e. no longer cba
    x = GenericBlock('ca', 1, activation=final_activation, kernel_size=1, strides=1)(x)  # look at table 1
    #x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x) # table 1 used 3x3 conv with softmax activation
    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


class RFINETBlock(GenericBlock):

    def __call__(self, input_tensor, level=0, skip_tensor=None, direction='down'):
        if direction == 'up':
            n = GenericBlock('n')(input_tensor, skip_tensor=skip_tensor)
        else:
            n = input_tensor
        x = GenericBlock('cba cba cb',
                         self.filters,
                         activation=self.activation,
                         kernel_regularizer=self.kernel_regularizer)(n, level=level)
        x = GenericBlock('cb p bad',
                         self.filters,
                         kernel_size=1,
                         activation=self.activation,
                         dropout=self.dropout,
                         kernel_regularizer=self.kernel_regularizer)(n, level=level, skip_tensor=x)
        return x

def freeze_RFI_NET(model: tf.keras.Model):
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'trainable'):
            model.layers[i].trainable = False
    model.layers[-2].trainable = True # last conv
    model.layers[-5].trainable = True # last BN
    model.layers[-7].trainable = True  # BN
    model.layers[-8].trainable = True  # BN
    model.layers[-9].trainable = True  # 2nd last conv
    model.layers[-10].trainable = True  # 3rd last conv
    model.layers[-12].trainable = True  # 4th last BN
    model.layers[-13].trainable = True  # 4th last conv
    model.layers[-15].trainable = True  # 5th last BN
    model.layers[-16].trainable = True  # 5th last conv
    return model
