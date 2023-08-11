import copy
import numpy as np
import time

from .patches import get_multichannel_patches, num_patches_per_image
from .scaler import scale
from .clipper import clip
from .splitter import split
from .batcher import batch
from .channels import first_channels
from .shuffler import shuffle
from .replace_zeros import replace_zeros_with_min
from utils.flagging import flag_data


# args.clip_per = choice (none, dataset, patch, image)

def preprocess(data,
               masks,
               data_name,
               rfi_threshold,
               flag_test_data,
               limit,
               clipper,
               std_min,
               std_max,
               perc_min,
               perc_max,
               clip_per_image,
               rescale,
               scale_per_image,
               log,
               patches,
               patch_x,
               patch_y,
               patch_stride_x,
               patch_stride_y,
               data_subset='train',  # train, test, hyp
               **kwargs):
    """ Assume we don't have to retain raw data and masks, i.e. we keep the reference.
    Passed raw_data will thus change and cant be preprocessed again"""
    # ret_data = copy.deepcopy(data)
    # ret_masks = copy.deepcopy(masks)

    # limit
    print(limit)
    print(perc_min)
    if limit is not None and limit != 'None' and 'test' not in data_subset:
        limit = int(limit)
        data = data[:limit]
        if masks is not None:
            masks = masks[:limit]

    # Flag
    if rfi_threshold is not None and rfi_threshold != 'None':
        if (flag_test_data and 'test' in data_subset) or 'test' not in data_subset:
            masks = flag_data(data, data_name, rfi_threshold)

    # Replace zeros in each image with min nonzero value in image
    data[..., 0] = replace_zeros_with_min(data[..., 0])  # (N_im, width, height, channels)

    # Clip
    data[..., 0:1] = clip(data[..., 0:1], masks, std_min, std_max, perc_min, perc_max, clip_per_image, data_name, clipper)

    # log
    if log:
        data[..., 0] = np.log(data[..., 0])

    # Scale
    if rescale:
        data[..., 0] = scale(data[..., 0], scale_per_image)
        if data.shape[-1] == 2:  # phase
            data[..., 1] = scale(data[..., 1], scale_per_image=False)

    # Patches
    if patches:
        data = get_multichannel_patches(data, patch_x, patch_y, patch_stride_x, patch_stride_y)
        masks = get_multichannel_patches(masks, patch_x, patch_y, patch_stride_x, patch_stride_y)

    return data, masks


def preprocess_all(train_data,
                   train_masks,
                   test_data,
                   test_masks,
                   task,
                   input_channels,
                   val_split,
                   shuffle_seed,
                   shuffle_patches,
                   use_hyp_data,
                   data_name,
                   rfi_threshold,
                   flag_test_data,
                   limit,
                   clipper,
                   std_min,
                   std_max,
                   perc_min,
                   perc_max,
                   clip_per_image,
                   rescale,
                   scale_per_image,
                   log,
                   patches,
                   patch_x,
                   patch_y,
                   patch_stride_x,
                   patch_stride_y,
                   train_with_test,
                   **kwargs):
    # ret_data = copy.deepcopy(data)
    # ret_masks = copy.deepcopy(masks)

    start = time.time()

    # Get raw_input shape and extract the first channels
    raw_input_shape = None
    if train_data is not None:
        raw_input_shape = train_data.shape[1:]
        train_data = first_channels(input_channels, train_data)
    if test_data is not None:
        raw_input_shape = test_data.shape[1:]
        test_data = first_channels(input_channels, test_data)
    if train_data is None and test_data is None:
        raise ValueError('train_data and test_data is both None')

    # Train on hyperparameter data ?
    if task in ('train', 'transfer_train') and not train_with_test:
        shuffle(42, train_data, train_masks)
        if use_hyp_data:
            train_data, train_masks = split(0.2, train_data, train_masks)[2:4]
        else:
            train_data, train_masks = split(0.2, train_data, train_masks)[0:2]

    # Train with test set data?
    # print(train_with_test)
    if train_with_test:
        # print('aaaaaaaaaaaa')
        limit = int(limit)
        shuffle_seed = shuffle(shuffle_seed, test_data, test_masks)
        train_data, train_masks = test_data[:limit], test_masks[:limit]
        test_data, test_masks = test_data[limit:], test_masks[limit:]
    else:
        # affects train-val split if patches arent shuffled
        shuffle_seed = shuffle(shuffle_seed, train_data, train_masks)

    # Preprocess training data
    input_shape = None
    if train_data is not None:
        train_data, train_masks = preprocess(train_data,
                                             train_masks,
                                             data_name=data_name,
                                             rfi_threshold=rfi_threshold,
                                             flag_test_data=flag_test_data,
                                             limit=limit,
                                             clipper=clipper,
                                             std_min=std_min,
                                             std_max=std_max,
                                             perc_min=perc_min,
                                             perc_max=perc_max,
                                             clip_per_image=clip_per_image,
                                             rescale=rescale,
                                             scale_per_image=scale_per_image,
                                             log=log,
                                             patches=patches,
                                             patch_x=patch_x,
                                             patch_y=patch_y,
                                             patch_stride_x=patch_stride_x,
                                             patch_stride_y=patch_stride_y,
                                             data_subset='train',
                                             **kwargs)
        input_shape = train_data.shape[1:]

    # Preprocess test data
    if test_data is not None:
        test_data, test_masks = preprocess(test_data,
                                           test_masks,
                                           data_name=data_name,
                                           rfi_threshold=rfi_threshold,
                                           flag_test_data=flag_test_data,
                                           limit=None,
                                           clipper=clipper,
                                           std_min=std_min,
                                           std_max=std_max,
                                           perc_min=perc_min,
                                           perc_max=perc_max,
                                           clip_per_image=clip_per_image,
                                           rescale=rescale,
                                           scale_per_image=scale_per_image,
                                           log=log,
                                           patches=patches,
                                           patch_x=patch_x,
                                           patch_y=patch_y,
                                           patch_stride_x=patch_stride_x,
                                           patch_stride_y=patch_stride_y,
                                           data_subset='test',
                                           **kwargs)
        input_shape = test_data.shape[1:]

    # Shuffle the patches
    if shuffle_patches and patches:
        shuffle(shuffle_seed, train_data, train_masks)
        shuffle(shuffle_seed, test_data, test_masks)

    # Split to train and validation sets
    val_data, val_masks, num_val, num_train, num_test = None, None, 0, 0, 0
    #if task in ('train', 'transfer_train'):
    train_data, train_masks, val_data, val_masks = split(val_split, train_data, train_masks)

    # Record number of training and validation images
    if train_data is not None:
        num_train = train_data.shape[0]
    if val_data is not None:
        num_val = val_data.shape[0]
    if test_data is not None:
        num_test = test_data.shape[0]

    patches_per_image = num_patches_per_image(raw_input_shape, patch_x, patch_y)

    results_dict = dict(
        raw_input_shape=raw_input_shape,
        input_shape=input_shape,
        num_train=num_train,
        num_val=num_val,
        num_test=num_test,
        shuffle_seed=shuffle_seed,
        patches_per_image=patches_per_image,
    )

    print('Preproccess time: {:.2f} sec'.format(time.time() - start))

    return train_data, train_masks, val_data, val_masks, test_data, test_masks, results_dict
