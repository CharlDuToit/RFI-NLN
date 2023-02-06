import tensorflow as tf


def DKNN(input_shape, patch_x, patch_y, **kwargs):
    # Why 256? Becasue it is nearest 224?
    # What if patch_x > 256 ? Will Upsampling still work?
    if patch_x > 256 or patch_y > 256:
        raise ValueError('Patches may not be larger than 256 for DKNN')
    s_x = 256 // patch_x
    s_y = 256 // patch_y

    inputs = tf.keras.layers.Input(shape=input_shape)
    rgb = tf.keras.layers.Concatenate(axis=-1)([inputs, inputs, inputs])
    resize = tf.keras.layers.UpSampling2D(size=(s_x, s_y))(rgb)
    # why crop it?
    crop = tf.keras.layers.Cropping2D(16)(resize)

    # Why do we use the weights of imagenet? does this not produce 1000 classes?
    resnet = tf.keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3),
                                                   pooling='max')(crop)

    model = tf.keras.Model(inputs=inputs, outputs=resnet)
    return model
