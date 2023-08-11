from utils.data import batch, rfi_ratio_indexes
from .fit_fcn import fit_fcn_train_val, fit_fcn_train_val_tf
from .fit_bb import fit_bb_train_val
from utils import images_to_cells, create_anchor_boxes, normalize_cells
from utils import images_masks_to_cells
import numpy as np

import tensorflow as tf


def train(model,
          train_data,
          train_masks,
          # train_rfi_ratio,
          val_data,
          val_masks,
          # val_rfi_ratio,
          batch_size,
          rfi_set,
          **kwargs
          ):
    """Batches the  data and trains based on model_class"""
    # tf.clear_graph() # error
    # tf.reset_default_graph() # error
    # tf.keras.backend.clear_session() # does nothing
    # tf.compat.v1.reset_default_graph() # does nothing

    # train_rfi_ratio, val_rfi_ratio, test_rfi_ratio = scaled_rfi_ratio_args(train_masks, val_masks, test_masks)
    model_class = kwargs['model_class']

    if rfi_set == 'separate':
        return train_separate(model,
                              train_data,
                              train_masks,
                              val_data,
                              val_masks,
                              batch_size,
                              rfi_set,
                              **kwargs
                              )

    # Batch it up
    # (train_data_batches,
    #  train_masks_batches,
    #  val_data_batches,
    #  val_masks_batches) = batch(batch_size,
    #                             train_data,
    #                             train_masks,
    #                             val_data,
    #                             val_masks)

    # optimizer = optimizers.Adam(lr=lr)

    results_dict = {}
    if model_class not in ('AE-SSIM', 'DKNN', 'AE', 'DAE', 'BB'):

        # ============================= old way ================================

        train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch = fit_fcn_train_val(model,
                                                                                                   train_data,
                                                                                                   train_masks,
                                                                                                   val_data,
                                                                                                   val_masks,
                                                                                                   model_type=None,
                                                                                                   batch_size=batch_size,
                                                                                                   **kwargs)
        results_dict = dict(train_loss=train_loss, val_loss=val_loss, train_loss_over_val_loss=train_loss_over_val_loss,
                            epoch_time=epoch_time, last_epoch=last_epoch)

        # ============================= end old way ================================
        # else:
        #     train_loss, epoch_time, last_epoch = fit_fcn_train(model,
        #                                                        train_data_batches, train_masks_batches,
        #                                                        **kwargs)
        #     results_dict = dict(train_loss=train_loss, val_loss=None, epoch_time=epoch_time, last_epoch=last_epoch)
    elif model_class == 'BB':
        return train_bb(model,
                        train_data,
                        train_masks,
                        val_data,
                        val_masks,
                        batch_size,
                        **kwargs
                        )
    elif model_class == 'AE-SSIM':
        # kwargs['loss'] = 'ae-ssim' # Handle this at main_args.py
        pass
    elif model_class == 'DAE':
        pass
    elif model_class == 'AE':
        pass
    elif model_class == 'DKNN':
        pass

    return results_dict


def train_separate(model,
                   train_data,
                   train_masks,
                   val_data,
                   val_masks,
                   batch_size,
                   rfi_set,
                   rfi_split_ratio,
                   **kwargs
                   ):
    """Batches the  data and trains based on model_class"""
    # train_rfi_ratio, val_rfi_ratio, test_rfi_ratio = scaled_rfi_ratio_args(train_masks, val_masks, test_masks)
    model_low = model[0]
    model_high = model[1]

    train_lo_ind, train_hi_ind = rfi_ratio_indexes(train_masks, rfi_split_ratio)
    val_lo_ind, val_hi_ind = rfi_ratio_indexes(val_masks, rfi_split_ratio)

    num_val_low = np.sum(val_lo_ind) / len(val_data)
    num_train_low = np.sum(train_lo_ind) / len(train_data)
    num_low_dict = dict(num_val_low=num_val_low, num_train_low=num_train_low)

    # Batch it up
    (train_data_low_batches,
     train_masks_low_batches,
     train_data_high_batches,
     train_masks_high_batches,
     val_data_low_batches,
     val_masks_low_batches,
     val_data_high_batches,
     val_masks_high_batches,
     ) = batch(batch_size,
               train_data[train_lo_ind],
               train_masks[train_lo_ind],
               train_data[train_hi_ind],
               train_masks[train_hi_ind],
               val_data[val_lo_ind],
               val_masks[val_lo_ind],
               val_data[val_hi_ind],
               val_masks[val_hi_ind],
               )

    # optimizer = optimizers.Adam(lr=lr)

    results_dict = {}
    model_class = kwargs['model_class']
    if model_class not in ('AE-SSIM', 'DKNN', 'AE', 'DAE'):
        # train low rfi
        train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch = fit_fcn_train_val(model_low,
                                                                                                   train_data_low_batches,
                                                                                                   train_masks_low_batches,
                                                                                                   val_data_low_batches,
                                                                                                   val_masks_low_batches,
                                                                                                   model_type='low',
                                                                                                   **kwargs)
        results_dict_low = dict(train_loss_low=train_loss, val_loss_low=val_loss,
                                train_loss_over_val_loss_low=train_loss_over_val_loss,
                                epoch_time_low=epoch_time,
                                last_epoch_low=last_epoch)

        # train high rfi
        train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch = fit_fcn_train_val(model_high,
                                                                                                   train_data_high_batches,
                                                                                                   train_masks_high_batches,
                                                                                                   val_data_high_batches,
                                                                                                   val_masks_high_batches,
                                                                                                   model_type='high',
                                                                                                   **kwargs)
        results_dict_high = dict(train_loss_high=train_loss, val_loss_high=val_loss,
                                 train_loss_over_val_loss_high=train_loss_over_val_loss,
                                 epoch_time_high=epoch_time,
                                 last_epoch_high=last_epoch)
        return {**num_low_dict, **results_dict_low, **results_dict_high}

    elif model_class == 'AE-SSIM':
        # kwargs['loss'] = 'ae-ssim' # Handle this at main_args.py
        pass
    elif model_class == 'DAE':
        pass
    elif model_class == 'AE':
        pass
    elif model_class == 'DKNN':
        pass

    return results_dict


def train_bb(model,
             train_data,
             train_masks,
             val_data,
             val_masks,
             batch_size,
             # num_anchors,
             **kwargs
             ):
    """Batches the  data and trains based on model_class"""
    # train_rfi_ratio, val_rfi_ratio, test_rfi_ratio = scaled_rfi_ratio_args(train_masks, val_masks, test_masks)
    anchor_boxes = create_anchor_boxes((8,))
    num_anchors = anchor_boxes.shape[0]
    train_cells = images_to_cells(train_masks, anchor_boxes, 8, 8)
    val_cells = images_to_cells(val_masks, anchor_boxes, 8, 8)
    # train_cells = images_masks_to_cells(train_masks, 8, 8)
    # val_cells = images_masks_to_cells(val_masks, 8, 8)

    # Batch it up
    (train_data_batches,
     train_cells_batches,
     val_data_batches,
     val_cells_batches,
     ) = batch(batch_size,
               train_data,
               train_cells,
               val_data,
               val_cells
               )

    # optimizer = optimizers.Adam(lr=lr)

    train_loss, val_loss, epoch_time, last_epoch = fit_bb_train_val(model,
                                                                    train_data_batches,
                                                                    train_cells_batches,
                                                                    val_data_batches,
                                                                    val_cells_batches,
                                                                    # num_anchors=num_anchors,
                                                                    **kwargs)
    results_dict = dict(num_anchors=num_anchors, train_loss=train_loss, val_loss=val_loss,
                        epoch_time=epoch_time, last_epoch=last_epoch)

    return results_dict


def train_combined_tf(model,
                      train_data,
                      train_masks,
                      # train_rfi_ratio,
                      val_data,
                      val_masks,
                      # val_rfi_ratio,
                      batch_size,
                      rfi_set,
                      **kwargs
                      ):
    """Batches the  data and trains based on model_class"""

    model_class = kwargs['model_class']

    # Batch it up
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_masks)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_masks)).batch(batch_size)

    results_dict = {}
    if model_class not in ('AE-SSIM', 'DKNN', 'AE', 'DAE', 'BB'):

        train_loss, val_loss, train_loss_over_val_loss, epoch_time, last_epoch = fit_fcn_train_val_tf(model,
                                                                                                      train_dataset,
                                                                                                      val_dataset,
                                                                                                      model_type=None,
                                                                                                      **kwargs)
        results_dict = dict(train_loss=train_loss, val_loss=val_loss, train_loss_over_val_loss=train_loss_over_val_loss,
                            epoch_time=epoch_time, last_epoch=last_epoch)

    elif model_class == 'BB':
        return train_bb(model,
                        train_data,
                        train_masks,
                        val_data,
                        val_masks,
                        batch_size,
                        **kwargs
                        )
    elif model_class == 'AE-SSIM':
        # kwargs['loss'] = 'ae-ssim' # Handle this at main_args.py
        pass
    elif model_class == 'DAE':
        pass
    elif model_class == 'AE':
        pass
    elif model_class == 'DKNN':
        pass

    return results_dict
