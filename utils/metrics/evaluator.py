from .inference import infer
from utils import reconstruct
from utils import save_data_masks_inferred
from .segmentation_metrics import auroc, auprc, f1, precision, accuracy, recall
from utils import model_dir
import time


def infer_and_get_metrics(model, data, masks, patches_per_image, data_subset='train', save_images=False, batch_size=64,
                          images_per_epoch=10, **kwargs):

    if data is None or masks is None:
        return {}

    start = time.time()
    masks_inferred = infer(model, data, batch_size=batch_size)
    infer_time = time.time() - start
    time_patch = infer_time / len(data)  # per patch
    time_image = time_patch * patches_per_image

    if save_images:
        n_p = patches_per_image * images_per_epoch
        reconstruct_and_save_images(data[:n_p], masks[:n_p], masks_inferred[:n_p], **kwargs)

    _auroc = auroc(masks, masks_inferred)
    _auprc = auprc(masks, masks_inferred)
    _f1 = f1(masks, masks_inferred)
    _accuracy = accuracy(masks, masks_inferred)
    _recall = recall(masks, masks_inferred)
    _precision = precision(masks, masks_inferred)

    return {
        f'{data_subset}_auroc': _auroc,
        f'{data_subset}_auprc': _auprc,
        f'{data_subset}_f1': _f1,
        f'{data_subset}_accuracy': _accuracy,
        f'{data_subset}_recall': _recall,
        f'{data_subset}_precision': _precision,
        'time_image': time_image,
        'time_patch': time_patch
    }


def reconstruct_and_save_images(data, masks, inferred_masks, raw_input_shape, patch_x, patch_y, **kwargs):
    data_recon = reconstruct(data, raw_input_shape, patch_x, patch_y, None, None)
    masks_recon = reconstruct(masks, raw_input_shape, patch_x, patch_y, None, None)
    inferred_masks_recon = reconstruct(inferred_masks, raw_input_shape, patch_x, patch_y, None, None)
    _dir = model_dir(**kwargs)
    save_data_masks_inferred(_dir, data_recon, masks_recon, inferred_masks_recon)
    save_data_masks_inferred(_dir, data_recon, masks_recon, inferred_masks_recon, thresh=0.5)


def evaluate(model,
             train_data,
             train_masks,
             val_data,
             val_masks,
             test_data,
             test_masks,
             **kwargs
             ):
    train_metrics = infer_and_get_metrics(model, train_data, train_masks, data_subset='train', **kwargs)
    val_metrics = infer_and_get_metrics(model, val_data, val_masks, data_subset='val', **kwargs)
    test_metrics = infer_and_get_metrics(model, test_data, test_masks, data_subset='test',
                                         save_images=True and not kwargs['shuffle_patches'],
                                         **kwargs)

    return {
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }
