from utils.data import batch
from .fit_fcn import fit_fcn_train, fit_fcn_train_val


def train(model,
          train_data,
          train_masks,
          val_data,
          val_masks,
          batch_size,
          **kwargs
          ):
    """Batches the  data and trains based on model_class"""

    # Batch it up
    (train_data_batches,
     train_masks_batches,
     val_data_batches,
     val_masks_batches) = batch(batch_size,
                                train_data,
                                train_masks,
                                val_data,
                                val_masks)

    results_dict = {}
    model_class = kwargs['model_class']
    if model_class not in ('AE-SSIM', 'DKNN', 'AE', 'DAE'):
        if val_data_batches is not None:
            train_loss, val_loss, epoch_time, last_epoch = fit_fcn_train_val(model,
                                                                             train_data_batches, train_masks_batches,
                                                                             val_data_batches, val_masks_batches,
                                                                             **kwargs)
            results_dict = dict(train_loss=train_loss, val_loss=val_loss, epoch_time=epoch_time, last_epoch=last_epoch)
        else:
            train_loss, epoch_time, last_epoch = fit_fcn_train(model,
                                                               train_data_batches, train_masks_batches,
                                                               **kwargs)
            results_dict = dict(train_loss=train_loss, val_loss=None, epoch_time=epoch_time, last_epoch=last_epoch)

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
