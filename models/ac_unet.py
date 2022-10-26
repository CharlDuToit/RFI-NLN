import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

from .generic_builder import GenericBlock, GenericUnet

tf.keras.backend.set_floatx('float32')

def AC_UNET(args):
    """
    Charl's implementation according to the article:
    0 Radio frequency interference detection based on the AC-UNet model
    Rui-Qing-Yan
    Accepted 2020 December

    Upsampling as mentioned in article is assumed to be Conv2DTranspose. Look at fig5's first Up-conv, 2x2 and the
    AC,3x3 relu just before. The AC would produce 256 filters, and Up-conv does not change the number of filters.
    So in the concatenated layer, there would be 128+256 filters, but 256 is indicated.
    Either the AC is with 128 filters followed by UpConv, or the AC is 256 filters followed by Conv2DTranspose with 128
    filters

    21 layer 100x100 patch
    14 AC layers - fig 5 shows 10, AC with 1x1 kernel is not possible
    two-layer 2x2 maxpool layer stride 2
    two-layer fusion layers
    1 dropout layer lastly somewhere?
    two-layer upsampling layers - see above paragraph
    """
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if args.height is None else np.minimum(args.height, max_height)

    dilation_rate = args.dilation_rate if args.dilation_rate > 1 else 2  # pretty pointless to have dr of 1

    level_block = GenericBlock('ncba', args.filters, blocks=args.level_blocks, dilation_rate=dilation_rate)
    x = GenericUnet(height, level_block=level_block)(input_data)

    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model
