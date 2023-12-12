import numpy as np
import tensorflow as tf
# from data import *
from utils import main_args
from utils import to_dict, dict_to_str
from utils import load_raw_data
from utils import preprocess_all, scaled_rfi_ratio_args
# from utils import split
# from utils import batch
# from utils import shuffle
from utils import train, load_checkpoint, train_combined_tf
from utils import save_csv
from utils import save_summary, params_and_flops_as_dict
from utils import solution_file, get_loss_metrics, checkpoint_file
# from utils import plot_model_to_file_kwargs

from models import load_model, freeze_top_layers
from utils import evaluate, save_percentile, model_dir, ratios_and_labels, evaluate_curves, infer_and_get_f1_and_save
# from architectures import *
# from utils.hardcoded_args import *
# from utils.data import DataCollection
# from data_collection import load_data_collection
# from architectures import load_architecture

import pandas as pd

import time, os


# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def main(kwargs: dict):
    """
        Reads data and cmd arguments and trains models
    """
    if kwargs['task'] == 'eval_test':
        file = f'{kwargs["output_path"]}/results_{kwargs["data_name"]}_{kwargs["seed"]}.csv'
        if os.path.exists(file):
            df_eval_sofar = pd.read_csv(file)
            df = df_eval_sofar.query(f'model_name=="{kwargs["model_name"]}"')
            if len(df) > 0:
                print(f'{kwargs["model_name"]} as already been evaluated, exiting...')
                return

    kwargs = main_args.validate_main_kwargs_dict(kwargs)

    if kwargs['task'] in ('eval', 'eval_test'):
        f = checkpoint_file(**kwargs) + '.index'
        if not os.path.exists(f):
            print(f'Checkpoint for {kwargs["model_name"]} does not exist, exitting...')
            return
    start = time.time()
    results = {}
    print('===============================================')

    args_str = dict_to_str(kwargs, ', ', 'model_class', 'data_name', 'task', 'use_hyp_data',
                           'input_channels', 'patch_x', 'batch_size',
                           'clipper', 'scale_per_image', 'clip_per_image')
    print(f'Running main with args:\n    {args_str}')

    # ----------------------------------- Load and pre-process data -----------------------------------
    # Raw data
    train_data, train_masks, test_data, test_masks = load_raw_data(**kwargs)

    # Preprocess all
    #print(kwargs['train_with_test'])
    #print(kwargs['train_with_test'] == True)
    #print(kwargs['train_with_test'] == False)
    if np.isnan(kwargs['train_with_test']): kwargs['train_with_test'] = False


    train_data, train_masks, val_data, val_masks, test_data, test_masks, proc_dict = preprocess_all(train_data,
                                                                                                    train_masks,
                                                                                                    test_data,
                                                                                                    test_masks,
                                                                                                    **kwargs)
    # train_rfi_ratio, val_rfi_ratio, test_rfi_ratio = scaled_rfi_ratio_args(train_masks, val_masks, test_masks)

    # Save RFI ratio percentiles
    _dir = model_dir(**kwargs)
    # ratios, labels = ratios_and_labels(train_masks, val_masks, test_masks)
    # save_percentile(ratios, logy=True, labels=labels, xlabel='rfi ratio percentiles', ylabel='rfi ratio', dir_path=_dir, file_name='rfi_percentiles')

    # Save to kwargs
    kwargs = {**kwargs, **proc_dict}
    kwargs['images_per_epoch'] = np.min([kwargs['images_per_epoch'], kwargs['batch_size'], kwargs['num_train']])

    # ----------------------------------- Load model  -----------------------------------

    args_str = dict_to_str(kwargs, ', ', 'model_class', 'rfi_set', 'model_name', 'parent_model_name', 'filters', 'height', 'loss',
                           'activation', 'final_activation', 'dropout', 'kernel_regularizer', 'lr')
    print(f'Creating model:\n    {args_str}')

    kwargs['num_anchors'] = 1
    model = load_model(**kwargs)

    task = kwargs['task']
    if task in ('eval', 'infer', 'eval_test'):
        load_checkpoint(model, load_parent=False, **kwargs)

    if task == 'transfer_train':
        load_checkpoint(model, load_parent=True, **kwargs)
        if kwargs['freeze_top_layers']:
            model = freeze_top_layers(model, **kwargs)

    # # stupid code, delete later
    # if kwargs['train_with_test']:
    #     kwargs['model_'], kwargs['test_data'], kwargs['test_masks'] = model, test_data, test_masks

    save_summary(model, **kwargs)
    # plot_model_to_file_kwargs(model, **kwargs) # pip install pydot
    #model_dict = params_and_flops_as_dict(model, **kwargs)
    #results = {**results, **model_dict}

    # ----------------------------------- Perform task -----------------------------------
    # print('num val = ', kwargs['num_val'])

    kwargs['pos_weight'] = 1.0
    print('shuffle_seed = ', kwargs['shuffle_seed'])
    if task in ('train', 'transfer_train'):
        #train_dict = train_combined_tf(model, train_data, train_masks, val_data, val_masks, **kwargs)
        train_dict = train(model, train_data, train_masks, val_data, val_masks, **kwargs)
        results = {**results, **train_dict}
        load_checkpoint(model, load_parent=False, **kwargs)

    if task in ('train', 'transfer_train', 'eval'):
        if 'train_loss' not in results.keys():
            loss_metrics = get_loss_metrics(**kwargs)
            results = {**results, **loss_metrics}
        metrics_dict = evaluate(model, train_data, train_masks, val_data, val_masks, test_data, test_masks, **kwargs)
        results = {**results, **metrics_dict}

    if task in ('eval_test',):
        # metrics_dict = evaluate_test(model, test_data[0:64, ...], test_masks[0:64, ...], **kwargs)
        metrics_dict = evaluate_curves(model, test_data, test_masks, val_data, val_masks, train_data, train_masks, **kwargs)
        # metrics_dict = infer_and_get_f1_and_save(model, test_data, test_masks, **kwargs)
        results = {**results, **metrics_dict}

    # elif kwargs['task'] == 'infer':
    #     arch = load_architecture(args.args, checkpoint='self')
    #     arch.infer_and_save_train_data(data_collection)

    # ----------------------------------- Save results -----------------------------------

    results = {**kwargs, **results}
    save_csv(results_dict=results, **kwargs)
    file = solution_file(**kwargs)
    with open(file, 'w') as fp:
        for k in results.keys():
            if '_vals' not in k:
                fp.write(f'{k}: {results[k]}\n')

    print('Total time: {:.2f} min'.format((time.time() - start) / 60))
    print('===============================================')


def my_args(args_):
    from coolname import generate_slug as new_name
    print('RUNNING MY ARGS, LINE IN if __main__ NOT COMMENTED OUT')

    args_.fold = 0
    args_.model_class = 'RNET5'
    args_.cmd_args = ['data_path', 'data_name', 'output_path', 'model_name', 'n_splits', 'seed', 'limit']
    args_.seed = 'bestmodel_f1' # 'eval_curves_adaptive'
    args_.n_splits = 1  # full size images
    args_.limit = 14  # full size images
    args_.task = 'train'
    args_.train_with_test = False
    args_.freeze_top_layers = True
    args_.calc_train_val_auc = True
    args_.rfi_set = 'combined'
    args_.rfi_split_ratio = 0.01
    # RNET5 MSE: pompous-feathered-degu-of-attack, dice: debonair-holistic-gaur-of-reading, bce: victorious-abstract-petrel-of-saturation, logcoshdice: economic-soft-mantis-of-fury
    # args_.model_name = 'proud-malamute-of-abstract-intensity' # Best L2 model
    args_.model_name = 'test' # 'burrowing-tricky-swan-of-wealth' # Best H3 model
    args_.parent_model_name = None # 'crazy-mini-catfish-from-hyperborea'#'determined-happy-skunk-of-opportunity'
    args_.anomaly_class = 'rfi'
    args_.anomaly_type = 'MISO'
    args_.percentage_anomaly = 0.0
    # HERA: 14: 800, 28: 400, 56: 200, 112 (50% of hyp set): 100
    args_.epochs = 2
    args_.latent_dim = 8
    args_.alphas = [1.0]
    args_.neighbours = [2, 8]
    args_.radius = 2
    args_.algorithm = 2
    #args_.data_name = 'LOFAR'
    args_.data_name = 'HERA_CHARL'
    #args_.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    args_.data_path = '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/'
    args_.debug = 0
    args_.log = True
    args_.rotate = False
    args_.crop = False
    args_.crop_x = 512
    args_.crop_y = 512
    size = 64
    args_.patches = True
    args_.rescale = True
    args_.bn_first = False
    args_.patch_x = size
    args_.patch_y = size
    args_.patch_stride_x = size
    args_.patch_stride_y = size
    args_.flag_test_data = False
    args_.rfi = None
    args_.rfi_threshold = None
    args_.lofar_subset = 'all'
    args_.scale_per_image = False
    args_.clip_per_image = False
    args_.clipper = 'perc'  # 'None', 'std', 'dyn_std', 'known'
    args_.std_max = 4
    args_.std_min = -1
    args_.perc_max = 99.8
    # args_.perc_min = 0.2 # HERA
    args_.perc_min = 0.2 # LOFAR
    args_.filters = 16
    args_.height = 4
    args_.level_blocks = 1
    args_.model_config = 'rubbish'
    args_.dropout = 0.1
    args_.batch_size = 64
    args_.buffer_size = 1024
    args_.optimal_alpha = True
    args_.optimal_neighbours = True
    args_.use_hyp_data = True
    args_.lr = 1e-4
    args_.lr_lin_decay = 1.0
    args_.loss = 'dice'
    args_.kernel_regularizer = 'l2'
    args_.input_channels = 1
    args_.dilation_rate = 1
    args_.epoch_image_interval = 50
    args_.images_per_epoch = 10
    args_.early_stop = 1000
    args_.shuffle_seed = 42  # 3476595572 2342063437
    args_.val_split = 0.2
    args_.final_activation = 'sigmoid'
    args_.activation = 'relu'
    args_.output_path = '/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/junk' # './outputs'
    args_.save_dataset = False
    args_.shuffle_patches = True
    return args_


if __name__ == '__main__':
    # main-args, pre_processor, main, entire models folder checkpointer

    # main_args.args = my_args(main_args.args)
    main(to_dict(main_args.args))
    # tf.keras.backend.clear_session()
    # main(to_dict(main_args.args))
    # print(main_args.args)
