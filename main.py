# import numpy as np
# import tensorflow as tf
from data import *
from utils import args
# from architectures import *
#from utils.hardcoded_args import *
# from utils.data import DataCollection
from data_collection import get_data_collection_from_args
from architectures import get_architecture_from_args

import time


# import os
# os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

def main():
    """
        Reads data and cmd arguments and trains models
    """
    # print(args.args)
    # return

    start = time.time()
    print('===============================================')


    print('Running with args:\n   model: {}, data: {}, task: {},  model_config: {}, use_hyp_data: {}, loss: {}, lr: {},'
          ' dropout: {}, kernel_regularizer: {}, input_channels: {}'.format(
          args.args.model, args.args.data, args.args.task, args.args.model_config, args.args.use_hyp_data, args.args.loss, args.args.lr,
          args.args.dropout, args.args.kernel_regularizer, args.args.input_channels))


    print("__________________________________ \nFetching and preprocessing data: {}".format(args.args.data))
    data_collection = get_data_collection_from_args(args.args)
    data_collection.load_raw_data()
    data_collection.preprocess()
    # Final shapes are determined after loading data_collection
    args.args.raw_input_shape = data_collection.raw_input_shape
    args.args.input_shape = data_collection.input_shape
    print(args.args.input_shape)
    print('Data time : {:.2f} sec'.format(time.time() - start))


    print("__________________________________ \nModel: {}, Name: {}, Parent name: {}".format(
        args.args.model, args.args.model_name, args.args.parent_model_name))

    if args.args.task == 'train':
        arch = get_architecture_from_args(args.args, checkpoint='None')
        arch.train(data_collection)
        arch.evaluate_and_save(data_collection)
    elif args.args.task == 'eval':
        arch = get_architecture_from_args(args.args, checkpoint='self')
        arch.evaluate_and_save(data_collection)
    elif args.args.task == 'infer':
        arch = get_architecture_from_args(args.args, checkpoint='self')
        arch.infer_and_save_train_data(data_collection)
    elif args.args.task == 'tranfer_train':
        arch = get_architecture_from_args(args.args, checkpoint='parent')
        arch.train(data_collection)
        arch.evaluate_and_save(data_collection)


    print('Total time : {:.2f} min'.format((time.time() - start) / 60))
    print('===============================================')


def tiny_args(args_):
    args_.optimal_alpha = True
    args_.optimal_neighbours = True
    args_.model = 'UNET'
    args_.task = 'train'
    #args_.model_name = ''
    args_.parent_model_name = None
    args_.anomaly_class = 'rfi'
    args_.anomaly_type = 'MISO'
    args_.filters = 4
    args_.height = 2
    args_.level_blocks = 1
    size = 64
    args_.patches = True
    args_.patch_x = size
    args_.patch_y = size
    args_.patch_stride_x = size
    args_.patch_stride_y = size
    args_.epochs = 2
    #args_.limit = 112  # full size images
    args_.limit = 10  # full size images
    # args_.limit = None  # full size images
    args_.batch_size = 64
    args_.rfi_threshold = None
    args_.rfi = None
    args_.lr = 1e-4
    args_.seed = 'test'
    args_.loss = 'bce'
    args_.kernel_regularizer = 'l2'
    args_.dropout = 0.05
    # args_.use_hyp_data = True
    args_.use_hyp_data = True
    args_.dilation_rate = 1
    #args_.data_path = '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/'
    args_.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    args_.data = 'HERA'
    args_.input_channels = 1
    args_.final_activation = 'sigmoid'
    args_.early_stop = 20
    args_.val_split = 0.2
    args_.split_seed = 42
    args_.epoch_image_interval = 1
    args_.images_per_epoch = 1
    args_.save_dataset = False
    return args_

def infer_args(args_):
    # -anomaly_class
    # rfi \
    # - anomaly_type
    # MISO \
    # - use_hyp_data
    # False \
    # - output_path. / outputs / infer_test \
    args_.anomaly_class = 'rfi'
    args_.anomaly_type = 'MISO'
    args_.output_path = './outputs/infer_test'
    args_.model_config = 'tiny'
    args_.optimal_alpha = True
    args_.optimal_neighbours = True
    args_.model = 'RNET'
    #args_.model = 'DSC_DUAL_RESUNET'
    #args_.model = 'UNET'
    args_.model_name = 'illustrious-poodle-of-stimulating-joviality' # RNET
    #args_.model_name = 'eccentric-cute-grasshopper-of-refinement' # DSC_DUAL_RESUNET
    #args_.model_name = 'hospitable-intrepid-coucal-of-novelty' # UNET
    #args_ = resolve_model_config_args(args_)
    args_.filters = 16 # RNET DSC_DUAL
    #args_.filters = 64 #UNET
    args_.height = 4
    args.level_blocks = 1
    size = 64
    args.patches = True
    args.patch_x = size
    args.patch_y = size
    args.patch_stride_x = size
    args.patch_stride_y = size
    args_.epochs = 2
    args_.limit = None  # full size images
    # args_.limit = None  # full size images
    args_.rfi_threshold = None
    args_.seed = 'test'
    args_.loss = 'bce'
    #args_.kernel_regularizer = 'l2'
    args_.kernel_regularizer = None
    args_.dropout = 0.05
    # args_.use_hyp_data = True
    args_.use_hyp_data = False
    args_.data_path = '/home/ee487519/PycharmProjects/correlator/'
    # args_.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    args_.data = 'ant_fft_000_094_t4096_f4096'
    args_.input_channels = 1
    args_.final_activation = 'sigmoid'
    args_.early_stop = 20
    args_.val_split = 0.2
    args_.split_seed = 42
    args_.epoch_image_interval = 10
    args_.images_per_epoch = 10
    args_.parent_model_name = None
    args_.task = 'infer'
    args_.save_dataset = True
    return args_

if __name__ == '__main__':
    args.args = tiny_args(args.args)
    # print(args.args)
    #args.args = infer_args(args.args)
    main()
