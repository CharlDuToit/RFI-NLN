# import numpy as np
# import tensorflow as tf
from data import *
from utils import args
# from architectures import *
from utils.hardcoded_args import *
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
    print('Running with args:\n   model: {}, data: {}, model_config: {}, use_hyp_data: {}, loss: {}, lr: {},'
          ' dropout: {}, kernel_regularizer: {}, input_channels: {}'.format(
          args.args.model, args.args.data, args.args.model_config, args.args.use_hyp_data, args.args.loss, args.args.lr,
          args.args.dropout, args.args.kernel_regularizer, args.args.input_channels))
    print("__________________________________ \nFetching and preprocessing data: {}".format(args.args.data))
    data_collection = get_data_collection_from_args(args.args)
    data_collection.load_raw_data()
    data_collection.preprocess()
    # Final shapes are determined after loading data_collection
    args.args.raw_input_shape = data_collection.raw_input_shape
    args.args.input_shape = data_collection.input_shape
    print('Data time : {:.2f} sec'.format(time.time() - start))
    print("__________________________________ \nModel: {}, Name: {}".format(args.args.model, args.args.model_name))
    arch = get_architecture_from_args(args.args)
    arch.train(data_collection)
    print("__________________________________ \nEvaluating data")
    eval_start = time.time()
    arch.evaluate_and_save(data_collection)
    print('Data Evaluation time : {:.2f} sec'.format(time.time() - eval_start))
    print('Total time : {:.2f} min'.format((time.time() - start) / 60))
    print('===============================================')


def tiny_args(args_):
    args_ = set_hera_args(args_)
    args_.model_config = 'tiny'
    args_.optimal_alpha = True
    args_.optimal_neighbours = True
    args_.model = 'UNET'
    args_ = resolve_model_config_args(args_)
    args_.epochs = 2
    args_.limit = 112  # full size images
    # args_.limit = None  # full size images
    args_.rfi_threshold = None
    args_.seed = 'test'
    args_.loss = 'bce'
    args_.kernel_regularizer = 'l2'
    args_.dropout = 0.05
    # args_.use_hyp_data = True
    args_.use_hyp_data = True
    args_.data_path = '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/'
    # args_.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    args_.data = 'HERA_PHASE'
    args_.input_channels = 1
    return args_


if __name__ == '__main__':
    args.args = tiny_args(args.args)
    # print(args.args)
    main()
