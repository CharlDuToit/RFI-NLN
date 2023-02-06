import numpy as np
# import tensorflow as tf
# from data import *
from utils import main_args
from utils import to_dict, dict_to_str
from utils import load_raw_data
from utils import preprocess, preprocess_all
from utils import split
from utils import batch
from utils import shuffle
from utils import train, load_checkpoint
from utils import save_csv
from utils import save_summary, params_and_flops_as_dict
from utils import solution_file

from models import load_model
from utils import num_patches_per_image, evaluate
# from architectures import *
# from utils.hardcoded_args import *
# from utils.data import DataCollection
from data_collection import load_data_collection
from architectures import load_architecture

import time


# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def main(kwargs: dict):
    """
        Reads data and cmd arguments and trains models
    """

    start = time.time()
    results = {}
    print('===============================================')

    args_str = dict_to_str(kwargs, ', ', 'model_class', 'data_name', 'task', 'use_hyp_data', 'loss', 'lr', 'dropout',
                           'kernel_regularizer', 'input_channels', 'patch_x', 'batch_size')
    print(f'Running main with args:\n    {args_str}')

    # ----------------------------------- Load and pre-process data -----------------------------------
    # Raw data
    train_data, train_masks, test_data, test_masks = load_raw_data(**kwargs)

    # Preprocess all
    train_data, train_masks, val_data, val_masks, test_data, test_masks, proc_dict = preprocess_all(train_data,
                                                                                                    train_masks,
                                                                                                    test_data,
                                                                                                    test_masks,
                                                                                                    **kwargs)
    # Save to kwargs
    kwargs = {**kwargs, **proc_dict}
    kwargs['images_per_epoch'] = np.min([kwargs['images_per_epoch'], kwargs['batch_size'], kwargs['num_train']])

    # ----------------------------------- Load model  -----------------------------------

    args_str = dict_to_str(kwargs, ', ', 'model_class', 'model_name', 'parent_model_name', 'filters', 'height')
    print(f'Loading/creating model:\n    {args_str}')

    model = load_model(**kwargs)

    task = kwargs['task']
    if task in ('eval', 'infer', 'transfer_train'):
        load_checkpoint(model, **kwargs)

    save_summary(model, **kwargs)
    model_dict = params_and_flops_as_dict(model, **kwargs)
    results = {**results, **model_dict}

    # ----------------------------------- Perform task -----------------------------------

    if task in ('train', 'transfer_train'):
        train_dict = train(model, train_data, train_masks, val_data, val_masks, **kwargs)
        results = {**results, **train_dict}

    if task in ('train', 'transfer_train', 'eval'):
        metrics_dict = evaluate(model, train_data, train_masks, val_data, val_masks, test_data, test_masks, **kwargs)
        results = {**results, **metrics_dict}

    # elif kwargs['task'] == 'infer':
    #    arch = load_architecture(args.args, checkpoint='self')
    #    arch.infer_and_save_train_data(data_collection)

    # ----------------------------------- Save results -----------------------------------

    results = {**kwargs, **results}
    save_csv(results_dict=results, **kwargs)
    file = solution_file(**kwargs)
    with open(file, 'w') as fp:
        for k in results.keys():
            fp.write(f'{k}: {results[k]}\n')

    print('Total time: {:.2f} min'.format((time.time() - start) / 60))
    print('===============================================')


def my_args(args_):
    from coolname import generate_slug as new_name

    args_.model_class = 'UNET'
    args_.task = 'train'
    args_.model_name = new_name()
    args_.parent_model_name = None
    args_.limit = 40  # full size images
    args_.anomaly_class = 'rfi'
    args_.anomaly_type = 'MISO'
    args_.percentage_anomaly = 0.0
    args_.epochs = 10
    args_.latent_dim = 8
    args_.alphas = [1.0]
    args_.neighbours = [2, 8]
    args_.radius = 2
    args_.algorithm = 2
    args_.data_name = 'LOFAR'
    args_.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    args_.seed = 'testseed'
    args_.debug = 0
    args_.log = True
    args_.rotate = False
    args_.crop = False
    args_.crop_x = 512
    args_.crop_y = 512
    size = 64
    args_.patches = True
    args_.patch_x = size
    args_.patch_y = size
    args_.patch_stride_x = size
    args_.patch_stride_y = size
    args_.flag_test_data = False
    args_.rfi = None
    args_.rfi_threshold = None
    args_.lofar_subset = 'all'
    args_.scale_per_image = True
    args_.clip_per_image = True
    args_.clipper = 'known'  # 'None', 'std', 'dyn_std', 'known'
    args_.std_max = 4
    args_.std_min = -1
    args_.filters = 4
    args_.height = 2
    args_.level_blocks = 1
    args_.model_config = 'rubbish'
    args_.dropout = 0.05
    args_.batch_size = 64
    args_.buffer_size = 1024
    args_.optimal_alpha = True
    args_.optimal_neighbours = True
    args_.use_hyp_data = True
    args_.lr = 1e-4
    args_.loss = 'bce'
    args_.kernel_regularizer = 'l2'
    args_.input_channels = 1
    args_.dilation_rate = 1
    args_.epoch_image_interval = 1
    args_.images_per_epoch = 10
    args_.early_stop = 20
    args_.shuffle_seed = 1 # None
    args_.val_split = 0.2
    args_.final_activation = 'sigmoid'
    args_.output_path = './outputs'
    args_.save_dataset = False
    args_.shuffle_patches = False
    return args_


if __name__ == '__main__':
    main_args.args = my_args(main_args.args)
    main(to_dict(main_args.args))
