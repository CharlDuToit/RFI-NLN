from .losses import get_loss_func, save_loss
# from .train_steps import train_step
from .printer import print_epoch
from .checkpointer import save_checkpoint
from .train_steps import get_train_step_func
from utils.common import model_dir, training_metrics_file, checkpoint_file
from utils.plotting import save_data_masks_inferred, save_epochs_curve
from utils import batch

import numpy as np
from tensorflow import optimizers
import time
import random

import tensorflow as tf


def fit_fcn_train_val(model,
                      train_data,
                      train_masks,
                      val_data,
                      val_masks,
                      lr,
                      epochs,
                      loss,
                      early_stop,
                      images_per_epoch,
                      epoch_image_interval,
                      model_type=None,
                      lr_lin_decay=1.0,
                      batch_size=64,
                      **kwargs):
    loss_func = get_loss_func(loss, **kwargs)
    optimizer = optimizers.Adam(lr=lr)

    train_losses = []
    val_losses = []
    _dir = model_dir(**kwargs)
    train_start = time.time()

    train_step_func = get_train_step_func()

    # Reset graph
    # train_step.experimental_restart() # error
    # tf.keras.backend.clear_session()  # does nothing
    # tf.compat.v1.reset_default_graph()  # does nothing

    # intervals to split training data into to save memory
    n_total_train = len(train_data)
    n_train_splits = 5
    interval_length = int((n_total_train // (n_train_splits * batch_size)) * batch_size)
    interval_length = np.maximum(interval_length, batch_size)
    interval_length = np.minimum(interval_length, n_total_train)
    n_train_splits = int(np.ceil(n_total_train / interval_length))
    intervals = []
    for i in range(n_train_splits):
        if i == n_train_splits - 1:
            intervals.append((interval_length * i, n_total_train))
        else:
            intervals.append((interval_length * i, interval_length * (i + 1)))
    #print(n_total_train)
    #print(intervals)
    # validation batches
    (val_data_batches,
     val_masks_batches) = batch(batch_size,
                                val_data,
                                val_masks)

    # loop epochs
    for epoch in range(epochs):
        start = time.time()
        total_train_loss = 0

        for inter in intervals:
            n_int = inter[1] - inter[0]
            (train_data_batches,
             train_masks_batches) = batch(batch_size,
                                          train_data[inter[0]:inter[1], ...],
                                          train_masks[inter[0]:inter[1], ...])
            # Train model for this interval
            for data_batch, mask_batch in zip(train_data_batches, train_masks_batches):
                train_step_func(model, data_batch, mask_batch, loss_func, optimizer, init_lr=lr, epochs=epochs,
                                epoch=epoch,
                                lindecay=lr_lin_decay)

            # Calculate train loss
            train_loss = calc_loss(model,
                                   train_data_batches,
                                   train_masks_batches,
                                   loss_func,
                                   save_images=(inter[0] == 0),  # only first interval
                                   images_per_epoch=images_per_epoch,
                                   epoch_image_interval=epoch_image_interval,
                                   epoch=epoch,
                                   model_type=model_type,
                                   save_dir=_dir)
            total_train_loss += (train_loss * n_int)

        total_train_loss /= n_total_train
        train_losses.append(total_train_loss)
        save_loss(total_train_loss, data_subset='train', model_type=model_type, **kwargs)

        # Calculate validation loss
        val_loss = calc_loss(model, val_data_batches, val_masks_batches, loss_func, save_images=False)
        val_losses.append(val_loss)
        save_loss(val_loss, data_subset='val', model_type=model_type, **kwargs)

        # Print losses for epoch
        print_epoch(epoch=epoch, time=time.time() - start, metrics=[total_train_loss, val_loss],
                    metric_labels=['train loss', 'val loss'], **kwargs)

        # Save checkpoint if lowest loss thus far
        save_checkpoint(model, losses=val_losses, model_type=model_type, **kwargs)

        # Early stopping
        if np.argmin(val_losses) + early_stop < len(val_losses) and early_stop > 1:  # No improvement for 20 epochs
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
    train_loss_over_val_loss = train_loss / val_loss

    # save_epochs_curve
    metrics_file = training_metrics_file(model_type=model_type, **kwargs)
    save_epochs_curve(None, [train_losses, val_losses], ['train loss', 'val loss'], file_name=metrics_file)

    return train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch


def calc_loss(model, data_batches, masks_batches, loss_func, save_images=False, images_per_epoch=10,
              epoch_image_interval=0,
              epoch=0,
              model_type=None,
              save_dir='./blahblah'
              ):
    tot_loss = 0.0
    # ret_data_batch, ret_masks_batch, ret_inferred_batch = None, None, None
    n_obs = 0
    if len(data_batches) > 1:
        b_ind = np.random.randint(0, len(data_batches) - 1)
    else:
        b_ind = 0
    i = 0
    for data_batch, mask_batch in zip(data_batches, masks_batches):
        x_hat = model(data_batch, training=False)
        loss = loss_func(x_hat, mask_batch)
        tot_loss += loss * len(data_batch)
        n_obs += len(data_batch)
        if b_ind == i and save_images and (
                epoch_image_interval > 0 and images_per_epoch > 0 and epoch % epoch_image_interval == 0):
            images_per_epoch = np.minimum(images_per_epoch, len(data_batch))
            im_inds = random.sample(range(len(data_batch)), images_per_epoch)
            ret_data_batch = data_batch.numpy()[im_inds]
            ret_masks_batch = mask_batch.numpy()[im_inds]
            ret_inferred_batch = x_hat.numpy()[im_inds]
            save_data_masks_inferred(save_dir,
                                     ret_data_batch,
                                     ret_masks_batch,
                                     ret_inferred_batch,
                                     epoch,
                                     model_type=model_type)
        i += 1

    tot_loss = (tot_loss / n_obs).numpy()  # divide by number of observations
    return tot_loss


def fit_fcn_train_val_tf(model,
                         train_dataset,
                         val_dataset,
                         lr,
                         epochs,
                         loss,
                         early_stop,
                         images_per_epoch,
                         epoch_image_interval,
                         model_type=None,
                         **kwargs):
    # compile
    loss_func = get_loss_func(loss, **kwargs)
    optimizer = optimizers.Adam(lr=lr)
    model.compile(
        optimizer=optimizer,
        loss=loss_func,
        # metrics=[tf.keras.metrics.MeanSquaredError()],
        # metrics=[tf.keras.metrics.AUC(curve='PR'), tf.keras.metrics.BinaryIoU(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )

    # callbacks
    chckpnt_file = checkpoint_file(model_type=model_type, **kwargs)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=chckpnt_file,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=early_stop,
        restore_best_weights=True,
        mode='min')

    # Fit
    train_start = time.time()
    history = model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[model_checkpoint_callback, early_stop_callback],
    )

    # Fit results
    train_time = time.time() - train_start
    print(f'Total training time: {train_time // 60} min')

    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    last_epoch = len(train_losses) - 1
    epoch_time = train_time / (last_epoch + 1)

    val_loss = np.min(val_losses)
    train_loss = train_losses[np.argmin(val_losses)]
    train_loss_over_val_loss = train_loss / val_loss

    # save_epochs_curve
    metrics_file = training_metrics_file(model_type=model_type, **kwargs)
    save_epochs_curve(None, [train_losses, val_losses], ['train loss', 'val loss'], file_name=metrics_file)

    return train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch

    # END for epoch in range(self.epochs):

    # Values to return
    last_epoch = epoch
    epoch_time = train_time / (epoch + 1)
    val_loss = np.min(val_losses)
    train_loss = train_losses[np.argmin(val_losses)]
    train_loss_over_val_loss = train_loss / val_loss

    # save_epochs_curve
    metrics_file = training_metrics_file(model_type=model_type, **kwargs)
    save_epochs_curve(None, [train_losses, val_losses], ['train loss', 'val loss'], file_name=metrics_file)

    return train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch

    _dir = model_dir(**kwargs)
    train_start = time.time()

    train_step_func = get_train_step_func()

    # Reset graph
    # train_step.experimental_restart() # error
    # tf.keras.backend.clear_session()  # does nothing
    # tf.compat.v1.reset_default_graph()  # does nothing
    for epoch in range(epochs):
        start = time.time()

        # Train model
        for data_batch, mask_batch in zip(train_data_batches, train_masks_batches):
            train_step_func(model, data_batch, mask_batch, loss_func, optimizer)

        # Calculate train loss
        train_loss, save_data, save_masks, save_inferred = calc_loss(model,
                                                                     train_data_batches,
                                                                     train_masks_batches,
                                                                     loss_func,
                                                                     True,
                                                                     images_per_epoch)
        train_losses.append(train_loss)
        save_loss(train_loss, data_subset='train', model_type=model_type, **kwargs)

        # Calculate validation loss
        val_loss = calc_loss(model, val_data_batches, val_masks_batches, loss_func, False)
        val_losses.append(val_loss)
        save_loss(val_loss, data_subset='val', model_type=model_type, **kwargs)

        # Print losses for epoch
        print_epoch(epoch=epoch, time=time.time() - start, metrics=[train_loss, val_loss],
                    metric_labels=['train loss', 'val loss'], **kwargs)

        # Save inferred images
        if epoch_image_interval > 0 and images_per_epoch > 0 and epoch % epoch_image_interval == 0:
            save_data_masks_inferred(_dir,
                                     save_data,
                                     save_masks,
                                     save_inferred,
                                     epoch,
                                     model_type=model_type)

        # Save checkpoint if lowest loss thus far
        save_checkpoint(model, losses=val_losses, model_type=model_type, **kwargs)

        # Early stopping
        if np.argmin(val_losses) + early_stop < len(val_losses) and early_stop > 1:  # No improvement for 20 epochs
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
    train_loss_over_val_loss = train_loss / val_loss

    # save_epochs_curve
    metrics_file = training_metrics_file(model_type=model_type, **kwargs)
    save_epochs_curve(None, [train_losses, val_losses], ['train loss', 'val loss'], file_name=metrics_file)

    return train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch
