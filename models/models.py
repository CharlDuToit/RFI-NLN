import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from model_config import n_layers, n_filters

tf.keras.backend.set_floatx('float32')


class Encoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=args.input_shape)
        self.conv, self.pool, self.batchnorm, self.dropout = [], [], [], []
        self.latent_dim = args.latent_dim

        for n in range(n_layers):
            self.conv.append(layers.Conv2D(filters=(n_layers - n) * n_filters,
                                           kernel_size=(3, 3),
                                           strides=(2, 2),
                                           padding='same',
                                           activation='relu'))
            # self.pool.append(layers.MaxPooling2D(pool_size=(2,2),padding='same'))

            self.batchnorm.append(layers.BatchNormalization())
            self.dropout.append(layers.Dropout(0.05))

        # output shape = 2,2
        self.flatten = layers.Flatten()
        self.dense_ae = layers.Dense(self.latent_dim, activation=None)

        self.dense_vae = layers.Dense(n_filters, activation='relu')
        self.mean = layers.Dense(self.latent_dim)
        self.logvar = layers.Dense(self.latent_dim)

    def call(self, x, vae=False):
        x = self.input_layer(x)

        for layer in range(n_layers):
            x = self.conv[layer](x)
            # if layer !=n_layers-1:
            #    x = self.pool[layer](x)
            x = self.batchnorm[layer](x)
            x = self.dropout[layer](x)
        x = self.flatten(x)

        if vae:
            x = self.dense_vae(x)
            mean = self.mean(x)
            logvar = self.logvar(x)
            return [mean, logvar]
        else:
            x = self.dense_ae(x)
            return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.latent_dim = args.latent_dim
        self.input_layer = layers.InputLayer(input_shape=[self.latent_dim, ])
        self.dense = layers.Dense(args.input_shape[0] // 2 ** (n_layers - 1) *
                                  args.input_shape[1] // 2 ** (n_layers - 1) *
                                  n_filters, activation='relu')
        self.reshape = layers.Reshape((args.input_shape[0] // 2 ** (n_layers - 1),
                                       args.input_shape[1] // 2 ** (n_layers - 1),
                                       n_filters))

        self.conv, self.pool, self.batchnorm, self.dropout = [], [], [], []
        for n in range(n_layers - 1):
            self.conv.append(layers.Conv2DTranspose(filters=(n + 1) * n_filters,
                                                    kernel_size=(3, 3),
                                                    strides=(2, 2),
                                                    padding='same',
                                                    activation='relu'))

            self.pool.append(layers.UpSampling2D(size=(2, 2)))
            self.batchnorm.append(layers.BatchNormalization())
            self.dropout.append(layers.Dropout(0.05))

        self.conv_output = layers.Conv2DTranspose(filters=args.input_shape[-1],
                                                  kernel_size=(3, 3),
                                                  padding='same',
                                                  activation='sigmoid')

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense(x)
        x = self.reshape(x)

        for layer in range(n_layers - 1):
            x = self.conv[layer](x)
            # x = self.pool[layer](x)
            x = self.batchnorm[layer](x)
            x = self.dropout[layer](x)

        x = self.conv_output(x)
        return x


class Autoencoder(tf.keras.Model):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def call(self, x):
        z = self.encoder(x, vae=False)
        x_hat = self.decoder(z)
        return x_hat


class Discriminator_x(tf.keras.Model):
    def __init__(self, args):
        super(Discriminator_x, self).__init__()
        self.network = Encoder(args)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        z = self.network(x)
        classifier = self.flatten(z)  # Is this required? Encoder already flattens? Maybe required for vae
        classifier = self.dense(classifier)
        return z, classifier


def Conv2D_block(input_tensor, n_filters, kernel_size=3, batchnorm=True, stride=(1, 1)):
    # first layer
    x = layers.Conv2D(filters=n_filters,
                      kernel_size=(kernel_size, kernel_size), \
                      kernel_initializer='he_normal',
                      strides=stride,
                      padding='same')(input_tensor)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def UNET(args, n_filters=16, dropout=0.05, batchnorm=True):
    # Contracting Path
    input_data = tf.keras.Input(args.input_shape, name='data')
    if args.input_shape[0] == 16:
        _str = 1  # cant downsample 16x16 patches
    else:
        _str = 2
    c1 = Conv2D_block(input_data,
                      n_filters * 1,
                      kernel_size=3,
                      batchnorm=batchnorm,
                      stride=(_str, _str))
    # p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(c1)

    c2 = Conv2D_block(p1,
                      n_filters * 2,
                      kernel_size=3,
                      stride=(2, 2),
                      batchnorm=batchnorm)
    # p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(c2)

    if args.input_shape[1] > 8:
        c3 = Conv2D_block(p2,
                          n_filters * 4,
                          kernel_size=3,
                          stride=(2, 2),
                          batchnorm=batchnorm)
        # p3 = layers.MaxPooling2D((2, 2))(c3)
        p3 = layers.Dropout(dropout)(c3)
    else:
        p3 = p2

    if args.input_shape[1] > 16:
        c4 = Conv2D_block(p3,
                          n_filters * 8,
                          kernel_size=3,
                          stride=(2, 2),
                          batchnorm=batchnorm)
        # p4 = layers.MaxPooling2D((2, 2))(c4)
        p4 = layers.Dropout(dropout)(c4)
    else:
        p4 = p3

    c5 = Conv2D_block(p4, n_filters=n_filters * 16, kernel_size=3, stride=(2, 2), batchnorm=batchnorm)

    # Expansive Path
    if args.input_shape[1] > 16:
        u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        u6 = layers.Dropout(dropout)(u6)
        c6 = Conv2D_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    else:
        c6 = c5

    if args.input_shape[1] > 8:
        u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        u7 = layers.Dropout(dropout)(u7)
        c7 = Conv2D_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    else:
        c7 = c6

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(dropout)(u8)
    c8 = Conv2D_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(dropout)(u9)
    if args.input_shape[0] != 16:  # cant downsample 16x16 patches
        u9 = layers.UpSampling2D((2, 2))(u9)
    c9 = Conv2D_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model


def RNET(args):
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


def RFINET_downblock(input_tensor, n_filters, kernel_size=3, batchnorm=True, stride=(1, 1)):
    # first layer
    x0 = layers.Conv2D(filters=n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(input_tensor)

    x0 = layers.BatchNormalization()(x0)
    x0 = layers.Activation('relu')(x0)

    x1 = layers.Conv2D(filters=2 * n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x0)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters=2 * n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)

    skip = layers.Conv2D(filters=2 * n_filters,
                         kernel_size=(1, 1), \
                         kernel_initializer='he_normal',
                         strides=stride,
                         padding='same')(input_tensor)
    skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x2, skip])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def RFINET_upblock(input_tensor, n_filters, kernel_size=3, batchnorm=True, stride=(1, 1)):
    x0 = layers.Conv2D(filters=n_filters,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(input_tensor)

    x0 = layers.BatchNormalization()(x0)
    x0 = layers.Activation('relu')(x0)

    x1 = layers.Conv2D(filters=n_filters // 2,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x0)

    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = layers.Conv2D(filters=n_filters // 2,
                       kernel_size=(kernel_size, kernel_size), \
                       kernel_initializer='he_normal',
                       strides=stride,
                       padding='same')(x1)
    x2 = layers.BatchNormalization()(x2)

    skip = layers.Conv2D(filters=n_filters // 2,
                         kernel_size=(1, 1), \
                         kernel_initializer='he_normal',
                         strides=stride,
                         padding='same')(input_tensor)
    skip = layers.BatchNormalization()(skip)

    x = layers.Add()([x2, skip])
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x


def RFI_NET(args, n_filters=32, dropout=0.05, batchnorm=True):
    # Contracting Path
    input_data = tf.keras.Input(args.input_shape, name='data')
    c0 = layers.Conv2D(filters=32,
                       kernel_size=(3, 3), \
                       kernel_initializer='he_normal',
                       strides=1,
                       padding='same')(input_data)

    c1 = RFINET_downblock(c0, n_filters * 1, kernel_size=3, batchnorm=batchnorm, stride=(1, 1))
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(dropout)(p1)

    c2 = RFINET_downblock(p1, n_filters * 2, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(dropout)(p2)

    c3 = RFINET_downblock(p2, n_filters * 4, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(dropout)(p3)

    c4 = RFINET_downblock(p3, n_filters * 8, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(dropout)(p4)

    c5 = RFINET_downblock(p4, n_filters * 16, kernel_size=3, stride=(1, 1), batchnorm=batchnorm)
    p5 = layers.MaxPooling2D((2, 2))(c5)
    p5 = layers.Dropout(dropout)(p5)

    # upsampling 
    u6 = layers.Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = RFINET_upblock(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = layers.Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = RFINET_upblock(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = layers.Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = RFINET_upblock(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = layers.Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = RFINET_upblock(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model


def CNN_RFI_SUN_block(input_tensor,
                      n_filters,
                      conv_kernel_size=(3, 3),
                      # conv_stride=(1,1),
                      pool_size=(2, 2),
                      pool_stride=(1, 1),
                      batchnorm=True):
    """Padding is always 1 on all 4 sides."""

    x = layers.ZeroPadding2D(padding=1)(input_tensor)  # pad 1 zero on all 4 sides

    x = layers.Conv2D(filters=n_filters,
                      kernel_size=conv_kernel_size,
                      # kernel_initializer='he_normal',
                      strides=1,  # conv_stride,
                      padding='valid')(x)

    if batchnorm:
        x = layers.BatchNormalization(momentum=0.1, epsilon=1e-05)(x)
        # x = layers.BatchNormalization()(x) # tf defaults

    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=pool_size,
                            strides=pool_stride,
                            padding='valid')(x)
    return x


def CNN_RFI_SUN(args, dropout=0.0):
    """Tensorflow implementation of
    https://github.com/astronomical-data-processing/CNN_RFI.git."""
    # torch
    # expected input: (C, H, W)
    # actual input: (H, W, C) = (1, 1, 2) with C[0]:amp, C[1]:phase
    # first Conv2D: torch.nn.Conv2d(1, 64, kernel_size=(1, 2), padding=1)
    #
    # tf input: (H, W, C) with data_format ='channels_last'
    #
    input_data = tf.keras.Input((1, 2, 1), name='data')

    c1 = CNN_RFI_SUN_block(input_data, 64, conv_kernel_size=(1, 2), pool_size=(1, 1), pool_stride=1)  # 64 3 3
    c2 = CNN_RFI_SUN_block(c1, 64, conv_kernel_size=(1, 2), pool_size=(1, 2), pool_stride=1)  # 64 5 3
    c3 = CNN_RFI_SUN_block(c2, 128, conv_kernel_size=3, pool_size=(1, 2), pool_stride=1)  # 128 5 3
    c4 = CNN_RFI_SUN_block(c3, 128, conv_kernel_size=3, pool_size=2, pool_stride=1)  # 128 4 1
    # c5 = CNN_RFI_SUN_block(c4, 1024, conv_kernel_size=3, pool_size=3, pool_stride=2)
    # c6 = CNN_RFI_SUN_block(c5, 1024, conv_kernel_size=3, pool_size=3, pool_stride=2)

    f = layers.Flatten()(c4)  # should apparently flatten to 512 nodes
    d1 = layers.Dense(1024, activation='relu')(f)
    do = layers.Dropout(dropout)(d1)
    outputs = layers.Dense(2, activation='softmax')(do)  # replace with 1 channel and sigmoid?

    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model

def AC_UNET_downpath(input_tensor, n_blocks, n_filters, kernel_size=3, stride=1, dilation_rate=3, droput=0.0, batchnorm=True):

    for i in range(n_blocks):
        if i ==0:
            x = input_tensor
        x = AC_UNET_conv2D_block(input_tensor,
                                 n_filters=n_filters,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 dilation_rate=dilation_rate,
                                 droput=droput,
                                 batchnorm=batchnorm)(x)

    x = layers.MaxPooling2D(pool_size=2,
                            strides=2,
                            padding='valid')(x)

    return x


def AC_UNET_uppath(input_tensor, skip_tensor, n_blocks, n_filters, kernel_size=3, stride=1, dilation_rate=3, droput=0.0,
                      batchnorm=True):
    """skip_tensor concatenated with processed input_tensor"""
    for i in range(n_blocks):
        if i == 0:
            x = input_tensor
        x = AC_UNET_conv2D_block(input_tensor,
                                 n_filters=n_filters,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 dilation_rate=dilation_rate,
                                 droput=droput,
                                 batchnorm=batchnorm)(x)

    x = layers.Conv2DTranspose(n_filters, 3, strides=2, padding='same')(x)

    x = layers.concatenate([x, skip_tensor])

    return x

def AC_UNET_conv2D_multiblocks(input_tensor, n_blocks, n_filters, kernel_size=3, stride=1, dilation_rate=3, dropout=0.0,
                               batchnorm=True):
    x = input_tensor
    for i in range(n_blocks):
        x = AC_UNET_conv2D_block(x,
                                 n_filters=n_filters,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 dilation_rate=dilation_rate,
                                 dropout=dropout,
                                 batchnorm=batchnorm)

    return x
def AC_UNET_conv2D_block(input_tensor, n_filters, kernel_size=3, stride=1, dilation_rate=3, dropout=0.0, batchnorm=True):

    x = layers.Conv2D(filters=n_filters,
                      kernel_size=kernel_size,
                      kernel_initializer='he_normal',
                      strides=stride,
                      dilation_rate=dilation_rate,
                      padding='same')(input_tensor)

    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    if dropout > 0.0 and dropout < 1.0:
        x = layers.Dropout(dropout)(x)

    return x


def AC_UNET(args, n_filters=64, dilation_rate=7, dropout=0.05, batchnorm=True):
    """
    Radio frequency interference detection based on the AC-UNet model
    Rui-Qing-Yan
    Accepted 2020 December

    21 layer 100x100 patch
    14 AC layers
    twoo-layer 2x2 maxpool layer stride 2
    two-layer fusion layers
    1 dropout layer
    two-layer upsampling layers
    """
    input_data = tf.keras.Input(args.input_shape, name='data')

    d1 = AC_UNET_conv2D_multiblocks(input_data, 2, n_filters, dilation_rate=dilation_rate, batchnorm=batchnorm, dropout=dropout)
    m1 = layers.MaxPooling2D(pool_size=2,strides=2,padding='valid')(d1)

    d2 = AC_UNET_conv2D_multiblocks(m1, 2, n_filters * 2, dilation_rate=dilation_rate, batchnorm=batchnorm, dropout=dropout)
    m2 = layers.MaxPooling2D(pool_size=2,strides=2,padding='valid')(d2)

    # x is the center layers in the belly of the U
    x = AC_UNET_conv2D_multiblocks(m2, 2, n_filters * 4, dilation_rate=dilation_rate, batchnorm=batchnorm, dropout=dropout)
    u2 = layers.Conv2DTranspose(n_filters*2, 3, strides=2, padding='same')(x)
    x = layers.concatenate([u2, d2])

    x = AC_UNET_conv2D_multiblocks(x, 2, n_filters * 2, dilation_rate=dilation_rate, batchnorm=batchnorm, dropout=dropout)
    u1 = layers.Conv2DTranspose(n_filters, 3, strides=2, padding='same')(x)
    x = layers.concatenate([u1, d1])

    x = AC_UNET_conv2D_multiblocks(x, 2, n_filters, dilation_rate=dilation_rate, batchnorm=batchnorm, dropout=dropout)

    outputs = layers.Conv2D(1, 1, strides=1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model

def dummy_model():

    input_data = tf.keras.Input((100, 100, 1), name='data')
    x = layers.Conv2D(filters=50,
                      kernel_size=3,
                      kernel_initializer='he_normal',
                      strides=1,
                      dilation_rate=1,
                      padding='same')(input_data)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    #input_data = tf.compat.v1.placeholder('float32', shape=(1, 100, 100, 1))

    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model