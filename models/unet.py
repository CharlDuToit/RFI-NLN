import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

# import sys
# sys.path.insert(1, '../')
from .helper import conv2D_block, generic_block, generic_unet, GenericUnet, GenericBlock

tf.keras.backend.set_floatx('float32')

def UNET_Mesarcik(args, n_filters=16, dropout=0.05, batchnorm=True):
    # Contracting Path
    input_data = tf.keras.Input(args.input_shape, name='data')
    if args.input_shape[0] == 16:
        _str = 1  # cant downsample 16x16 patches
    else:
        _str = 2
    c1 = conv2D_block(input_data,
                      n_filters * 1,
                      kernel_size=3,
                      batchnorm=batchnorm,
                      strides=(_str, _str))
    # p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(c1)

    c2 = conv2D_block(p1,
                      n_filters * 2,
                      kernel_size=3,
                      strides=(2, 2),
                      batchnorm=batchnorm)
    # p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(c2)

    if args.input_shape[1] > 8:
        c3 = conv2D_block(p2,
                          n_filters * 4,
                          kernel_size=3,
                          strides=(2, 2),
                          batchnorm=batchnorm)
        # p3 = layers.MaxPooling2D((2, 2))(c3)
        p3 = layers.Dropout(dropout)(c3)
    else:
        p3 = p2

    if args.input_shape[1] > 16:
        c4 = conv2D_block(p3,
                          n_filters * 8,
                          kernel_size=3,
                          strides=(2, 2),
                          batchnorm=batchnorm)
        # p4 = layers.MaxPooling2D((2, 2))(c4)
        p4 = layers.Dropout(dropout)(c4)
    else:
        p4 = p3

    # c5 = conv2D_block(p4, n_filters=n_filters * 8, kernel_size=3, strides=(2, 2), batchnorm=batchnorm)
    c5 = conv2D_block(p4, n_filters=n_filters * 16, kernel_size=3, strides=(2, 2), batchnorm=batchnorm)

    # Expansive Path
    if args.input_shape[1] > 16:
        u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        u6 = layers.Dropout(dropout)(u6)
        c6 = conv2D_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    else:
        c6 = c5

    if args.input_shape[1] > 8:
        u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        u7 = layers.Dropout(dropout)(u7)
        c7 = conv2D_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    else:
        c7 = c6

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = conv2D_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(dropout)(u9)
    if args.input_shape[0] != 16:  # cant downsample 16x16 patches
        u9 = layers.UpSampling2D((2, 2))(u9)
    c9 = conv2D_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model


def UNET2(args,
          n_filters=16,
          dropout=0.05,
          dilation_rate=1,
          height=None,
          dilations_per_level=0,
          upsample2D_last=False,
          batchnorm=True,
          maxpool=False):
    """Comments are with height = 5, input_shape = (32,32,1).
    Total number of layers is the depth.
    Total number of levels is the height. Levels consist of many layers"""

    input_data = tf.keras.Input(args.input_shape, name='data')
    current_size = args.input_shape[0]

    max_height = np.floor(np.log2(current_size)).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    up_int = int(not upsample2D_last)  # False: 1. True: 0
    block_strides = 1 if maxpool else 2
    filter_multiplier = 1
    down_blocks = [None] * (height - 1 + up_int)
    # upsample2D_last   True   False
    # len(down_blocks)  4      5

    next_tensor = input_data
    if not upsample2D_last:
        # d.shape = ( 32, 32, n_filters)
        d = conv2D_block(next_tensor,
                         n_filters,
                         kernel_size=3,
                         batchnorm=batchnorm,
                         dilation_rate=dilation_rate,
                         strides=1)
        if 0.0 < dropout < 1.0:
            p = layers.Dropout(dropout)(d)
        else:
            p = d
        down_blocks[0] = d
        next_tensor = p
        filter_multiplier *= 2
        current_size //= 2

    # contracting path
    # upsample2D_last True: h in [0...4]
    #                 False: h in [1...5]
    # the last iteration does not write to down_blocks
    for h in range(up_int, height + up_int, 1):
        n_dil_layers = dilations_per_level if (current_size > 3 and current_size >= 2 * (dilation_rate - 1) + 2) else 0
        for cl in range(n_dil_layers, -1, -1):
            d = conv2D_block(next_tensor,
                             n_filters * filter_multiplier,
                             kernel_size=3,
                             batchnorm=batchnorm,
                             dilation_rate=dilation_rate if cl != 0 else 1,
                             dropout=dropout if cl != 0 else 0.0,
                             strides=block_strides if cl == 0 else 1)
            next_tensor = d
        if maxpool:
            d = layers.MaxPooling2D(2, strides=2, padding='valid')(d)
        if 0.0 < dropout < 1.0 and h != height - 1 + up_int:
            p = layers.Dropout(dropout)(d)
        else:
            p = d
        if h != height - 1 + up_int:
            down_blocks[h] = d
            filter_multiplier *= 2
        next_tensor = p
        current_size //= 2

    # Expanding path
    # upsample2D_last True: h in [4...1]
    #                 False: h in [5...1]
    # last iteration does not call conv2D_block
    for h in range(height - 1 + up_int, 0, -1):
        filter_multiplier //= 2
        u = layers.Conv2DTranspose(n_filters * filter_multiplier, 3, strides=2, padding='same')(next_tensor)
        u = layers.concatenate([u, down_blocks[h - 1]])  # note: h-1
        if 0.0 < dropout < 1.0:
            u = layers.Dropout(dropout)(u)
        if h != 1:
            u = conv2D_block(u, n_filters * filter_multiplier, kernel_size=3, strides=1, batchnorm=batchnorm)
        next_tensor = u

    if upsample2D_last:
        next_tensor = layers.UpSampling2D((2, 2))(next_tensor)
        next_tensor = conv2D_block(next_tensor, n_filters, kernel_size=3, strides=1, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(next_tensor)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model


def UNET3(args,
          n_filters=16,
          dropout=0.05,
          dilation_rate=1,
          height=None,
          layers_per_level=0,
          #maxpool=False
          batchnorm=True,
          ):
    """Comments are with height = 5, input_shape = (32,32,1).
    Total number of layers is the depth.
    Total number of levels is the height. Levels consist of many layers"""

    input_data = tf.keras.Input(args.input_shape, name='data')
    current_size = args.input_shape[0]

    max_height = np.floor(np.log2(current_size)).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    filter_multiplier = 1
    down_blocks = []
    nxt = input_data

    # contracting path
    for h in range(0, height + 1, 1):  # 0..4, 5 values
        if h == 0 and layers_per_level == 0:
            _n_layers = 1
        else:
            _n_layers = layers_per_level
        for cl in range(_n_layers, -1 + int(h == height), -1):  # layers_per_level = 3, i in [3...0] so 4 values
            nxt = conv2D_block(nxt,
                               n_filters * filter_multiplier,  # *m,
                               kernel_size=3,
                               batchnorm=batchnorm,
                               dilation_rate=dilation_rate if cl != 0 else 1,
                               #dropout=dropout if (cl != 0 and (cl != 1 and layers_per_level>0)) else 0.0,
                               strides=2 if cl == 0 else 1)
            d = layers.Dropout(dropout)(nxt) if 0.0 < dropout < 1.0 else nxt
            if (cl == 1 or (cl == 0 and layers_per_level == 0)) and h + 1 < height + np.minimum(layers_per_level, 1):
                down_blocks.append(nxt)
                filter_multiplier *= 2
                current_size //= 2
            nxt = d

    # Expanding path
    _n_layers = np.maximum(1, layers_per_level)  # at least 1
    for h in range(height - 1, -1, -1):
        filter_multiplier //= 2
        nxt = layers.Conv2DTranspose(n_filters * filter_multiplier, 3, strides=2, padding='same')(nxt)
        nxt = layers.concatenate([nxt, down_blocks[h]])
        if 0.0 < dropout < 1.0:
            nxt = layers.Dropout(dropout)(nxt)
        for cl in range(_n_layers, 0, -1):  # at least 1 iteration
            nxt = conv2D_block(nxt,
                               n_filters * filter_multiplier,
                               dilation_rate=dilation_rate,
                               kernel_size=3,
                               strides=1,
                               batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(nxt)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model


def UNET_gen(args,
          n_filters=16,
          dropout=0.05,
          #dilation_rate=1,
          height=None,
          #layers_per_level=0,
          #kernel_size=3,
          #strides=2,
          #batchnorm=True,
          #first_down_block_level = 0
          ):
    """
    Charl's reimplementation of Mesarcik's UNET -- yields exactly the same model
    Comments are with height = 5, input_shape = (32,32,1).
    Total number of layers is the depth.
    Total number of levels is the height. Levels consist of many layers"""

    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int) #- 1  # why -1?  minimum tensor size 2x2
    height = max_height if height is None else np.minimum(height, max_height)

    f_mult = 1
    down_blocks = [None]*(height-1)
    x = input_data #  #[None] *(5-1) : 16, 8, 4, 2, belly 1

    for lid in range(0, height-1, 1):
        x = generic_block(x, 'cba', n_filters * f_mult, strides=2)
        down_blocks[lid] = x
        x = generic_block(x, 'd', dropout=dropout)
        f_mult *= 2

    x = generic_block(x, 'cba', n_filters * f_mult, strides=2)

    for lid in range(lid, -1, -1):
        f_mult /= 2
        layer_chars = 'tnd cba' if lid != 0 else 'tnd'
        strides = [2, 1] if lid != 0 else 2
        x = generic_block(x,
                          layer_chars,
                          n_filters * f_mult,
                          skip_tensor=down_blocks[lid],
                          strides=strides,
                          dropout=dropout)

    x = generic_block(x, 'ucba c', [n_filters * f_mult, 1], strides=1, kernel_size=[3, 1])
    x = layers.Activation('sigmoid')(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


def UNET_old(args):
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if args.height is None else np.minimum(args.height, max_height)

    down_kwargs = dict(layer_chars='cba cba', dropout=args.dropout)
    up_kwargs = dict(layer_chars='tn cbacba', strides=[2, 1], dropout=args.dropout)
    x = generic_unet(input_data, height, args.filters, down_kwargs=down_kwargs, mid_kwargs=down_kwargs,
                     up_kwargs=up_kwargs, dropout=args.dropout)

    x = generic_block(x, 'ca', 1, strides=1, kernel_size=1, dropout=args.dropout, activation='sigmoid')

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model

def UNET(args):
    input_data = tf.keras.Input(args.input_shape, name='data')
    max_height = np.floor(np.log2(args.input_shape[0])).astype(int) - 1  # why -1?  minimum tensor size 2x2
    height = max_height if args.height is None else np.minimum(args.height, max_height)

    level_block = GenericBlock('ncba', args.filters, blocks=args.level_blocks)
    x = GenericUnet(height, level_block=level_block)(input_data)

    x = GenericBlock('ca', 1, kernel_size=1, strides=1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model
