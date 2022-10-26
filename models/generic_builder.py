# import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.math import scalar_mul


def resolve_generic_params(layer_chars, filters, kernel_size, strides, dilation_rate, dropout,
                           activation):
    if isinstance(layer_chars, str):
        layer_chars_list = layer_chars.split()

    if isinstance(layer_chars, list):
        layer_chars_list = layer_chars

    layer_chars_str = ''
    for i in range(len(layer_chars_list)):
        layer_chars_list[i] = layer_chars_list[i].replace(' ', '')
        layer_chars_str += layer_chars_list[i]

    if layer_chars_str.count('p') + layer_chars_str.count('n') > 1:
        raise ValueError('One merge layer (p or n) at most may be passed')

    #just dont merge
    #if (layer_chars_str.count('p') + layer_chars_str.count('n') == 1) and skip_tensor is None:
        #raise ValueError('Merge layer requested, but no skip_tensor provided')

    valid_chars = [char for char in layer_chars_str if char in 'cutdampnbs']
    if len(valid_chars) != len(layer_chars_str):
        raise ValueError('Invalid layer char. Valid chars are `cutdampnbs`')

    if isinstance(kernel_size, list):
        if len(kernel_size) == 1:
            kernel_size = kernel_size * len(layer_chars_list)
        if len(kernel_size) != len(layer_chars_list):
            raise ValueError('Number of blocks not equal to number of kernel_sizes')
    else:
        kernel_size = [kernel_size] * len(layer_chars_list)

    if isinstance(strides, list):
        if len(strides) == 1:
            strides = strides * len(layer_chars_list)
        if len(strides) != len(layer_chars_list):
            raise ValueError('Number of blocks not equal to number of strides')
    else:
        strides = [strides] * len(layer_chars_list)

    if isinstance(dilation_rate, list):
        if len(dilation_rate) == 1:
            dilation_rate = dilation_rate * len(layer_chars_list)
        if len(dilation_rate) != len(layer_chars_list):
            raise ValueError('Number of blocks not equal to number of dilation_rates')
    else:
        dilation_rate = [dilation_rate] * len(layer_chars_list)

    if isinstance(dropout, list):
        if len(dropout) == 1:
            dropout = dropout * len(layer_chars_list)
        if len(dropout) != len(layer_chars_list):
            raise ValueError('Number of blocks not equal to number of dropouts')
    else:
        dropout = [dropout] * len(layer_chars_list)

    if isinstance(filters, list):
        if len(filters) == 1:
            filters = filters * len(layer_chars_list)
        if len(filters) != len(layer_chars_list):
            raise ValueError('Number of blocks not equal to number of items in n_filters list')
    else:
        filters = [filters] * len(layer_chars_list)

    if isinstance(activation, list):
        if len(activation) == 1:
            activation = activation * len(layer_chars_list)
        if len(activation) != len(layer_chars_list):
            raise ValueError('Number of blocks not equal to number of activations')
    else:
        activation = [activation] * len(layer_chars_list)

    for s, dr in zip(strides, dilation_rate):
        if s > 1 and dr > 1:
            raise ValueError('Dilation rate > 1 requires strides = 1')

    return layer_chars_list, filters, kernel_size, strides, dilation_rate, dropout, activation


class GenericBlock:
    """
    layer_chars is a string seperated by spaces, or a list of string. Every string item represents a block.
    Each character represents a different keras.layer.
    skip_tensor is required if p or n is provided. Only one p or n may exist.
    n_filters, kernel_size, strides, dilation_rate, dropout can be scalar/Tuple values or a list of scalar/Tuples.
    The length of these lists must be equal to the number of blocks.
    Maxpool and Upsampling layers are hard coded to x2 sampling

    c Conv2D \n
    u Upsampling2D \n
    t Conv2DTranspose\n
    d Dropout\n
    a Activation\n
    m Maxpool2D\n
    p Plus\n
    n Co(n)cat\n
    b Batchnormalization\n

    """

    def __init__(self,
                 layer_chars='nca',
                 filters=64,
                 kernel_size=3,
                 strides=1,
                 dilation_rate=1,
                 dropout=0.0,
                 activation='relu',
                 blocks=1,
                 ):
        self.layer_chars = layer_chars
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.activation = activation
        self.blocks = blocks

    def __call__(self, input_tensor, level=0, skip_tensor=None, direction='down'):
        #direction = down mid up
        # direction not used for generic block
        (layer_chars_list,
         n_filters_list,
         kernel_size_list,
         strides_list,
         dilation_rate_list,
         dropout_list,
         activation_list) = resolve_generic_params(self.layer_chars,
                                                   self.filters,
                                                   self.kernel_size,
                                                   self.strides,
                                                   self.dilation_rate,
                                                   self.dropout,
                                                   self.activation)
        x = input_tensor
        for b in range(self.blocks):
            for chars, f, k, s, dr, d, a in zip(layer_chars_list,
                                                n_filters_list,
                                                kernel_size_list,
                                                strides_list,
                                                dilation_rate_list,
                                                dropout_list,
                                                activation_list):
                for char in chars:
                    if char == 'c':
                        x = layers.Conv2D(filters=f * 2 ** level,
                                          kernel_size=k,
                                          kernel_initializer='he_normal',
                                          strides=s,
                                          dilation_rate=dr,
                                          padding='same')(x)
                    if char == 'u':
                        x = layers.UpSampling2D(size=(2, 2))(x)  # perhaps use s instead of 2?
                    if char == 't':
                        x = layers.Conv2DTranspose(filters=f * 2 ** level,
                                                   kernel_size=k,
                                                   strides=s,
                                                   dilation_rate=dr,
                                                   kernel_initializer='glorot_uniform',
                                                   padding='same')(x)
                    if char == 'd':
                        if 0.0 < d < 1.0:
                            x = layers.Dropout(d)(x)
                    if char == 'a':
                        x = layers.Activation(a)(x)
                    if char == 'm':
                        x = layers.MaxPooling2D((2, 2), strides=2)(x)  # perhaps use s instead of 2?
                    # only first block can have merge with input tensor, otherwise ignore letters m p
                    if skip_tensor is not None and b == 0:
                        if char == 'p':
                            x = layers.Add()([x, skip_tensor])
                        if char == 'n':
                            x = layers.concatenate([x, skip_tensor])
                    if char == 'b':
                        x = layers.BatchNormalization()(x)
                    if char == 's':
                        x = layers.SeparableConv2D(filters=f * 2 ** level,
                                                   kernel_size=k,
                                                   depth_multiplier=1,
                                                   kernel_initializer='he_normal',
                                                   strides=s,
                                                   dilation_rate=dr,
                                                   padding='same')(x)

        return x


class GenericUnet:

    def __init__(self,
                 height=3,  # at least 2
                 prev_unet_down_tensors=None,
                 concat_multiplier=1.0,
                 level_block=GenericBlock(filters=64),
                 ):
        self.height = height
        self.prev_unet_down_tensors = prev_unet_down_tensors
        self.level_block = level_block
        self.concat_multiplier = float(concat_multiplier)
        self.down_tensors = [None] * (self.height - 1)
        self.output_tensor = None

        if prev_unet_down_tensors is not None and len(prev_unet_down_tensors) != height-1:
            raise ValueError('Invalid prev_unet_down_tensors length')

    def __call__(self, input_tensor):
        x = input_tensor
        for level in range(0, self.height - 1, 1):
            x = self.level_block(x, level=level, direction='down')
            if self.prev_unet_down_tensors is not None:
                x = layers.Add()([x, self.prev_unet_down_tensors[level]])
            self.down_tensors[level] = x * self.concat_multiplier
            #self.down_tensors[level] = scalar_mul(self.concat_multiplier, x)
            x = layers.MaxPooling2D((2, 2), strides=2)(x)

        x = self.level_block(x, level=level + 1, direction='mid')

        for level in range(level, -1, -1):
            x = layers.Conv2DTranspose(self.level_block.filters * 2 ** level, kernel_size=3, strides=2, padding='same')(x)
            x = self.level_block(x, level=level, skip_tensor=self.down_tensors[level], direction='up')

        self.output_tensor = x
        return x

