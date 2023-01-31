import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from .generic_builder import GenericUnet, GenericBlock

tf.keras.backend.set_floatx('float32')


def DSC_MONO_RESUNET(args):
    """
    Charl's implementation of article below, but with one UNET
    46 DSC based Dual-Resunet for radio frequency interference identification
    Yan-Jun Zhang, Yan-Zuo Li, Jun Cheng,Yi-Hua Yan
    2021 September
    """
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if args.height is None else np.minimum(args.height, max_height)

    res_block = ResidualBlock(filters=args.filters, dropout=args.dropout, kernel_regularizer=args.kernel_regularizer)
    dsc_unet_0 = GenericUnet(height, level_block=res_block)
    x = dsc_unet_0(input_data)

    x = GenericBlock('ca', filters=1, kernel_size=1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


class ResidualBlock(GenericBlock):

    def __call__(self, input_tensor, level=0, skip_tensor=None, direction='not applicable'):
        skip0 = GenericBlock('nsb',
                             self.filters,
                             kernel_regularizer=self.kernel_regularizer,
                             dropout=self.dropout)(input_tensor, level=level, skip_tensor=skip_tensor)
        skip1 = GenericBlock('sbpd',
                             self.filters,
                             kernel_regularizer=self.kernel_regularizer,
                             dropout=self.dropout)(skip0, level=level, skip_tensor=skip0)
        x = GenericBlock('sbpd',
                         self.filters,
                         kernel_regularizer=self.kernel_regularizer,
                         dropout=self.dropout)(skip1, level=level, skip_tensor=skip1)
        return x

