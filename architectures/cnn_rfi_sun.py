import tensorflow as tf
import numpy as np
from sklearn import neighbors
from matplotlib import pyplot as plt
import time
from models import CNN_RFI_SUN

from utils.plotting import (generate_and_save_images,
                            generate_and_save_training)

from utils.training import print_epoch, save_checkpoint
from model_config import *
from .helper import end_routine
from inference import infer

optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(model, x, y):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        x_hat = model(x, training=True)
        loss = bce(x_hat, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train(cnn_rfi_sun, train_dataset, train_images, train_masks, test_images, test_labels, test_masks, args, verbose=True,
          save=True):
    cnn_rfi_sun_loss = []
    # might have to remove reshape
    train_mask_dataset = tf.data.Dataset.from_tensor_slices(train_masks.astype('float32').reshape((-1,1,1,2)).transpose(0,1,3,2)).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)
    train_data_dataset = tf.data.Dataset.from_tensor_slices(train_images.reshape((-1,1,1,2)).transpose(0,1,3,2)).shuffle(BUFFER_SIZE, seed=42).batch(BATCH_SIZE)
    for epoch in range(args.epochs):
        start = time.time()

        for image_batch, mask_batch in zip(train_data_dataset, train_mask_dataset):
            auto_loss = train_step(cnn_rfi_sun, image_batch, mask_batch)

        # training step is per pixel, image generation not possible
        #generate_and_save_images(cnn_rfi_sun,
        #                         epoch + 1,
        #                         image_batch[:25, ...],
        #                         'CNN_RFI_SUN',
        #                         args)
        save_checkpoint(cnn_rfi_sun, epoch, args, 'CNN_RFI_SUN', 'cnn_rfi_sun')

        cnn_rfi_sun_loss.append(auto_loss)  # auto_loss for the last batch

        print_epoch('CNN_RFI_SUN', epoch, time.time() - start, {'CNN_RFI_SUN Loss': auto_loss.numpy()}, None)

    generate_and_save_training([cnn_rfi_sun_loss],
                               ['cnn_rfi_sun loss'],
                               'CNN_RFI_SUN', args)
    #generate_and_save_images(cnn_rfi_sun, epoch, image_batch[:25, ...], 'CNN_RFI_SUN', args)

    return cnn_rfi_sun


def main(train_dataset, train_images, train_labels, train_masks, test_images, test_labels, test_masks, test_masks_orig,
         args):
    cnn_rfi_sun = CNN_RFI_SUN(args, dropout=0.0)

    #input_data = tf.keras.Input((1, 2, 1), name='data')
    #if train_images.shape[-1] != 2:
    #    raise Exception(f'Expected 2 channels in last dimension, got {train_images.shape[-1]}')
    #a.shape = (2,3,4,2)
    #a1 = a.reshape((-1, 1, 1, 2)).transpose(0, 1, 3, 2)
    #a1_back = a1.transpose(0,1,3,2).reshape((-1,3,4,2))

    cnn_rfi_sun = train(cnn_rfi_sun, train_dataset, train_images,train_masks, test_images, test_labels, test_masks, args)
    end_routine(train_images, test_images, test_labels, test_masks, test_masks_orig, [cnn_rfi_sun], 'CNN_RFI_SUN', args)


if __name__ == '__main__':
    main()
