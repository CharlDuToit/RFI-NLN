import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from model_config import n_layers,n_filters
tf.keras.backend.set_floatx('float32')

class Encoder(tf.keras.layers.Layer):
    def __init__(self,args):
        super(Encoder, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=args.input_shape)
        self.conv, self.pool, self.batchnorm = [],[],[]
        self.latent_dim  = args.latent_dim

        for n in range(n_layers):
            self.conv.append(layers.Conv2D(filters = n_filters, 
                                       kernel_size = (2,2), 
                                       #strides = (2,2),
                                       padding = 'same',
                                       activation='relu'))
            self.pool.append(layers.MaxPooling2D(pool_size=(2,2),padding='same'))

            self.batchnorm.append(layers.BatchNormalization())

        #output shape = 2,2
        self.flatten = layers.Flatten()
        self.dense_ae = layers.Dense(self.latent_dim, activation=None)

        self.dense_vae = layers.Dense(n_filters, activation='relu')
        self.mean = layers.Dense(self.latent_dim)
        self.logvar = layers.Dense(self.latent_dim)

    def call(self, x,vae=False):
        x = self.input_layer(x)

        for layer in range(n_layers):
            x = self.conv[layer](x)
            if layer !=n_layers-1:
                x = self.pool[layer](x)
            x = self.batchnorm[layer](x)
        x = self.flatten(x)

        if vae: 
            x = self.dense_vae(x)
            mean = self.mean(x)
            logvar = self.logvar(x)
            return [mean,logvar] 
        else: 
            x = self.dense_ae(x)
            return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self,args):
        super(Decoder, self).__init__()
        self.latent_dim = args.latent_dim
        self.input_layer = layers.InputLayer(input_shape=[self.latent_dim,])
        self.dense= layers.Dense(args.input_shape[0]//2**(n_layers-1) *
                                 args.input_shape[1]//2**(n_layers-1) *
                                 n_filters,activation='relu')
        self.reshape = layers.Reshape((args.input_shape[0]//2**(n_layers-1),
                                       args.input_shape[1]//2**(n_layers-1),
                                       n_filters))

        self.conv, self.pool, self.batchnorm = [],[],[]
        for _ in range(n_layers-1):

            self.conv.append(layers.Conv2DTranspose(filters = n_filters, 
                                               kernel_size = (2,2), 
                                               #strides = (2,2),
                                               padding = 'same',
                                               activation='relu'))

            self.pool.append(layers.UpSampling2D(size=(2,2)))
            self.batchnorm.append(layers.BatchNormalization())

        self.conv_output = layers.Conv2DTranspose(filters = args.input_shape[-1], 
                                           kernel_size = (2,2), 
                                           padding = 'same',
                                           activation='sigmoid')

    def call(self, x):
        x = self.input_layer(x)
        x = self.dense(x)
        x = self.reshape(x)

        for layer in range(n_layers -1):
            x = self.conv[layer](x)
            x = self.pool[layer](x)
            x = self.batchnorm[layer](x)
        
        x = self.conv_output(x)
        return  x

class Autoencoder(tf.keras.Model):
    def __init__(self,args):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def call(self,x):
        z = self.encoder(x,vae=False)
        x_hat = self.decoder(z)
        return x_hat 
class MultiEncoder(tf.keras.Model):
    def __init__(self,args):
        super(MultiEncoder, self).__init__()
        self.input_layer = layers.InputLayer(input_shape=args.input_shape)
        self.latent_dim  = args.latent_dim
        self.nneighbours = args.neighbors[0]
        self.input_convs  = []
        for n in range(self.nneighbours+1):
            self.input_convs.append([layers.Conv2D(n_filters, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'), 
                                     layers.Conv2D(n_filters, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same')])

        self.concat = []

        self.conv = []
        self.conv.append(layers.Conv2D(n_filters, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*2, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*2, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*4, (4, 4), strides=2, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters*2, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(n_filters, (3, 3), strides=1, activation=layers.LeakyReLU(alpha=0.2), padding='same'))
        self.conv.append(layers.Conv2D(self.latent_dim, (8, 8), strides=1, activation='linear', padding='valid'))

        self.reshape = layers.Reshape((self.latent_dim,))

    def call(self, x, nln):
        """
            x (np.array): train batch
            nln (tuple/list): a list of inputs to MultiEncoder in the following order [ NLN_0,...,NLN_n]
        """
        outputs = []
        x = self.input_convs[0][0](x)
        outputs.append(self.input_convs[0][1](x))

        for n, inp in enumerate(nln):
            x = self.input_convs[n+1][0](inp)
            outputs.append(self.input_convs[n+1][1](x))
        x = layers.concatenate(outputs)

        for layer in self.conv:
            x = layer(x)
        x = self.reshape(x)

        return x
class Discriminator_x(tf.keras.Model):
    def __init__(self,args):
        super(Discriminator_x, self).__init__()
        self.network = Encoder(args)
        self.flatten = layers.Flatten()
        self.dense  = layers.Dense(1,activation='sigmoid')

    def call(self,x):
        z = self.network(x)
        classifier = self.flatten(z) 
        classifier = self.dense(classifier) 
        return z,classifier 

class Discriminator_z(tf.keras.Model):
    def __init__(self,args):
        super(Discriminator_z, self).__init__()
        self.latent_dim = args.latent_dim
        self.model = tf.keras.Sequential()
        self.model.add(layers.InputLayer(input_shape=[self.latent_dim,]))

        self.model.add(layers.Dense(64))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Dense(128))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Dense(256))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(0.3))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))
    def call(self,x):
        return self.model(x)

class VAE(tf.keras.Model):
    def __init__(self,args):
        super(VAE, self).__init__()
        self.encoder = encoder(args)
        self.decoder = decoder(args)
        self.latent_dim = args.latent_dim

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(self.latent_dim))
        return self.decode(eps)#, apply_sigmoid=True)

    def reparameterize(self, mean, logvar):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        eps = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * logvar) * eps


    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self,x):
        mean, logvar = self.encoder(x,vae=True)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat 

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def UNET( n_filters = 16, dropout = 0.05, batchnorm = True):
    # Contracting Path
    input_data = Input((128, 64,1),name='data') 
    c1 = conv2d_block(input_data, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_data], outputs=[outputs])
    return model

