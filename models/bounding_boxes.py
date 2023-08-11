import tensorflow as tf
import numpy as np
from keras import layers
# from model_config import n_layers, n_filters
# from .generic_builder import GenericBlock

tf.keras.backend.set_floatx('float32')


def BoundingBox(input_shape, filters, dropout, kernel_regularizer, activation, final_activation, num_anchors, **kwargs):
    """
    Charl's of a basic boundary box predictor
    """
    input_data = tf.keras.Input(input_shape, name='data')  # 64, 64, 1
    x = layers.Conv2D(filters=filters, kernel_size=5, strides=(1, 1), activation='relu', padding='valid')(input_data)  # 60, 60
    x = layers.Conv2D(filters=filters, kernel_size=5, strides=(1, 1), activation='relu', padding='valid')(x)  # 56, 56
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)  # 28, 28, 1
    x = layers.Conv2D(filters=filters*2, kernel_size=5, strides=(1, 1), activation='relu', padding='valid')(x)  # 24, 24
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)  # 12, 12, 1
    x = layers.Conv2D(filters=filters*4, kernel_size=5, strides=(1, 1), activation='relu', padding='valid')(x)  # 8, 8

    # Define the anchor box generation layer
    x = layers.Conv2D(filters=num_anchors * 4, kernel_size=(1, 1), activation='linear', padding='same')(x)

    # Define classification layer
    x_class = layers.Conv2D(filters=num_anchors, kernel_size=(1, 1), activation='sigmoid', padding='same')(x)

    # Define regression layer
    x_box = layers.Conv2D(filters=num_anchors * 4, kernel_size=(1, 1), activation='linear', padding='same')(x)

    # Define the final output layer
    outputs = tf.keras.layers.Concatenate(axis=-1)([x_class, x_box])

    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model


def BoundingBox_v3(input_shape, filters, num_anchors, **kwargs):
    """
    Charl's of a basic boundary box predictor
    """
    input_data = tf.keras.Input(input_shape, name='data')  # 64, 64, 1
    x = layers.Conv2D(filters=filters, kernel_size=5, strides=(1, 1), padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(filters=filters, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)  # 32, 32
    x = layers.Conv2D(filters=filters*2, kernel_size=1, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(filters=filters*4, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)  # 16, 16
    x = layers.Conv2D(filters=filters*4, kernel_size=1, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(filters=filters*8, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)  # 8, 8
    x = layers.Conv2D(filters=filters*4, kernel_size=1, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)
    x = layers.Conv2D(filters=filters*8, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.1)(x)

    # 6 filters, 1 for class, 5 for boundary box
    x = layers.Conv2D(filters=5, kernel_size=1, strides=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(negative_slope=0.0)(x)

    model = tf.keras.Model(inputs=[input_data], outputs=[x])
    return model


def main():
    model = BoundingBox_v3(input_shape=(64, 64, 1), num_anchors=1, filters=16)
    x = np.random.random((3, 64, 64, 1))
    x_hat = model(x)
    print(x_hat.shape)


if __name__ == '__main__':
    main()