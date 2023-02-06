from .losses import get_loss_func, save_loss
from .train_steps import train_step
from .printer import print_epoch
from .checkpointer import save_checkpoint
from utils.common import model_dir
from utils.plotting import save_data_masks_inferred, save_epochs_curve

import numpy as np
from tensorflow import optimizers
import time


def fit_fcn_train_val(model,
                      train_data_batches,
                      train_masks_batches,
                      val_data_batches,
                      val_masks_batches,
                      epochs,
                      lr,
                      loss,
                      early_stop,
                      images_per_epoch,
                      epoch_image_interval,
                      **kwargs):
    loss_func = get_loss_func(loss)
    optimizer = optimizers.Adam(lr=lr)

    train_losses = []
    val_losses = []
    _dir = model_dir(**kwargs)
    train_start = time.time()
    for epoch in range(epochs):
        start = time.time()

        # Train model
        for data_batch, mask_batch in zip(train_data_batches, train_masks_batches):
            train_step(model, data_batch, mask_batch, loss_func, optimizer)

        # Calculate train loss
        train_loss, save_data, save_masks, save_inferred = calc_loss(model,
                                                                     train_data_batches,
                                                                     train_masks_batches,
                                                                     loss_func,
                                                                     True,
                                                                     images_per_epoch)
        train_losses.append(train_loss)
        save_loss(train_loss, data='train', **kwargs)

        # Calculate validation loss
        val_loss = calc_loss(model, val_data_batches, val_masks_batches, loss_func, False)
        val_losses.append(val_loss)
        save_loss(val_loss, data='val', **kwargs)

        # Print losses for epoch
        print_epoch(epoch=epoch, time=time.time() - start, metrics=[train_loss, val_loss],
                    metric_labels=['train loss', 'val loss'], **kwargs)

        # Save inferred images
        if epoch_image_interval > 0 and images_per_epoch > 0 and epoch % epoch_image_interval == 0:
            save_data_masks_inferred(_dir,
                                     save_data,
                                     save_masks,
                                     save_inferred,
                                     epoch)

        # Save checkpoint if lowest loss thus far
        save_checkpoint(model, losses=val_losses, **kwargs)

        # Early stopping
        if np.argmin(val_losses) + early_stop < len(val_losses):  # No improvement for 20 epochs
            print(f'No improvement for {early_stop} epochs, stopping training')
            break

    # END for epoch in range(self.epochs):
    train_time = time.time() - train_start
    print(f'Total training time: {train_time // 60} min')

    # Values to return
    last_epoch = epoch
    epoch_time = train_time / (epoch + 1)
    val_loss = np.min(val_losses)
    train_loss = train_losses[np.argmin(val_losses)]

    # save_epochs_curve
    save_epochs_curve(_dir, [train_losses, val_losses], ['train loss', 'val loss'])

    return train_loss, val_loss, epoch_time, last_epoch


def fit_fcn_train(model,
                  train_data_batches,
                  train_masks_batches,
                  epochs,
                  lr,
                  loss,
                  images_per_epoch,
                  epoch_image_interval,
                  **kwargs):
    loss_func = get_loss_func(loss)
    optimizer = optimizers.Adam(lr=lr)

    train_losses = []
    _dir = model_dir(**kwargs)
    train_start = time.time()
    for epoch in range(epochs):
        start = time.time()

        # Train model
        for data_batch, mask_batch in zip(train_data_batches, train_masks_batches):
            train_step(model, data_batch, mask_batch, loss_func, optimizer)

        # Calculate train loss
        train_loss, save_data, save_masks, save_inferred = calc_loss(model,
                                                                     train_data_batches,
                                                                     train_masks_batches,
                                                                     loss_func,
                                                                     True,
                                                                     images_per_epoch)
        train_losses.append(train_loss)
        save_loss(train_loss, data='train', **kwargs)

        # Print losses for epoch
        print_epoch(epoch=epoch, time=time.time() - start, metrics=[train_loss], metrics_labels=['train loss'], **kwargs)

        # Save inferred images
        if epoch_image_interval > 0 and images_per_epoch > 0 and epoch % epoch_image_interval == 0:
            save_data_masks_inferred(_dir,
                                     save_data,
                                     save_masks,
                                     save_inferred,
                                     epoch)

        # Save checkpoint if lowest loss thus far
        save_checkpoint(model, losses=train_losses, **kwargs)

    # END for epoch in range(self.epochs):
    train_time = time.time() - train_start

    print(f'Total training time: {train_time // 60} min')

    # Values to return
    epoch_time = train_time / (epoch + 1)
    train_loss = np.min(train_losses)

    # save_epochs_curve
    save_epochs_curve(_dir, [train_losses], ['train loss', ])

    return train_loss, epoch_time, epoch


def calc_loss(model, data_batches, masks_batches, loss_func, ret_first_batch=False, n_images=10):
    tot_loss = 0.0
    ret_data_batch, ret_masks_batch, ret_inferred_batch = None, None, None
    n_obs = 0
    for data_batch, mask_batch in zip(data_batches, masks_batches):
        x_hat = model(data_batch, training=False)
        loss = loss_func(x_hat, mask_batch)
        tot_loss += loss * len(data_batch)
        n_obs += len(data_batch)
        if ret_data_batch is None:  # get the first batch
            ret_data_batch = data_batch.numpy()[:n_images]
            ret_masks_batch = mask_batch.numpy()[:n_images]
            ret_inferred_batch = x_hat.numpy()[:n_images]
    tot_loss = (tot_loss / n_obs).numpy()  # divide by number of observations
    if not ret_first_batch:
        return tot_loss
    else:
        return tot_loss, ret_data_batch, ret_masks_batch, ret_inferred_batch
