import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
#from model_config import n_layers, n_filters

tf.keras.backend.set_floatx('float32')


class Encoder(tf.keras.layers.Layer):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=args.input_shape)
        self.conv, self.pool, self.batchnorm, self.dropout = [], [], [], []
        self.latent_dim = args.latent_dim
        self.height = args.height

        for n in range(args.height):
            self.conv.append(layers.Conv2D(filters=(args.height - n) * args.filters,
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
        #self.shape = (None, self.latent_dim)

        self.dense_vae = layers.Dense(args.filters, activation='relu')
        self.mean = layers.Dense(self.latent_dim)
        self.logvar = layers.Dense(self.latent_dim)

    #def __call__(self, x, vae=False):
    def call(self, x, vae=False):
        x = self.input_layer(x)

        for layer in range(self.height):
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
        self.height = args.height
        self.input_layer = layers.InputLayer(input_shape=[self.latent_dim, ])
        self.dense = layers.Dense(args.input_shape[0] // 2 ** (self.height - 1) *
                                  args.input_shape[1] // 2 ** (self.height - 1) *
                                  args.filters, activation='relu')
        self.reshape = layers.Reshape((args.input_shape[0] // 2 ** (self.height - 1),
                                       args.input_shape[1] // 2 ** (self.height - 1),
                                       args.filters))

        self.conv, self.pool, self.batchnorm, self.dropout = [], [], [], []
        for n in range(self.height - 1):
            self.conv.append(layers.Conv2DTranspose(filters=(n + 1) * args.filters,
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
    #def __call__(self, x):
    def call(self, x):
        x = self.input_layer(x)
        x = self.dense(x)
        x = self.reshape(x)

        for layer in range(self.height - 1):
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
    #def __call__(self, x):
    def call(self, x):
        z = self.encoder(x, vae=False)
        x_hat = self.decoder(z)
        return x_hat


class Discriminator(tf.keras.Model):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.network = Encoder(args)
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation='sigmoid')

    #def __call__(self, x):
    def call(self, x):
        z = self.network(x)
        classifier = self.flatten(z)  # Is this required? Encoder already flattens? Maybe required for vae
        classifier = self.dense(classifier)
        return z, classifier
