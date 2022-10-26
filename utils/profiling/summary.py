import os
import numpy as np

from .flops import get_flops


def save_summary(model, args):
    dir_path = 'outputs/{}/{}/{}'.format(args.model,
                                         args.anomaly_class,
                                         args.model_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(dir_path + '/model_summary', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f'GFLOPS: {get_flops(model) / 1e9}')


def save_summary_to_folder(model, folder, args):
    if folder[-1] == '/':
        folder = folder[0:-1]
    summ_path = f'{folder}/{args.model}_{args.model_config}_height_{args.height}_filters_{args.filters}_blocks_{args.level_blocks}'

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(summ_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f'GFLOPS: {get_flops(model)/1e9}')


def num_trainable_params(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


def num_non_trainable_params(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_variables])
