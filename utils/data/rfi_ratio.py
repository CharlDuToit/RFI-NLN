from .scaler import scale
import numpy as np


def ratios_and_labels(train_masks, val_masks, test_masks):
    labels = []
    ratios = []
    if train_masks is not None:
        labels.append('train')
        ratios.append(np.array([np.mean(m) for m in train_masks]))
    if val_masks is not None:
        labels.append('val')
        ratios.append(np.array([np.mean(m) for m in val_masks]))
    if test_masks is not None:
        labels.append('test')
        ratios.append(np.array([np.mean(m) for m in test_masks]))
    return ratios, labels


def rfi_ratio_split(data, masks, rfi_split_ratio, **kwargs):
    rfi_ratio = np.array([np.mean(m) for m in masks])

    lo_ind = rfi_ratio <= rfi_split_ratio
    hi_ind = rfi_ratio > rfi_split_ratio

    return data[lo_ind], masks[lo_ind], data[hi_ind], masks[hi_ind]


def rfi_ratio_indexes(masks, rfi_split_ratio, **kwargs):
    rfi_ratio = np.array([np.mean(m) for m in masks])

    lo_ind = rfi_ratio < rfi_split_ratio
    hi_ind = rfi_ratio >= rfi_split_ratio

    return lo_ind, hi_ind


def scaled_rfi_ratio(masks):
    rfi_ratios = np.array([np.mean(m) for m in masks])
    return scale(rfi_ratios, scale_per_image=False)


def scaled_rfi_ratio_args(*args):
    ret_args = []
    for a in args:
        if a is None:
            ret_args.append(None)
        else:
            ret_args.append(scaled_rfi_ratio(a))
    if len(ret_args) > 1:
        return tuple(ret_args)
    else:
        return ret_args[0]
