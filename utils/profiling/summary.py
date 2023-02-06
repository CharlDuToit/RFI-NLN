import os
import numpy as np

from .flops import get_flops
from utils.common import summary_file


def save_summary(model, **kwargs):
    file = summary_file(**kwargs)
    with open(file, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f'GFLOPS: {get_flops(model) / 1e9}')


def save_summary_to_folder(model, folder, args):
    if folder[-1] == '/':
        folder = folder[0:-1]

    if args.patches:
        summ_path = f'{folder}/{args.model}_{args.model_config}_height_{args.height}_filters_{args.filters}_blocks_{args.level_blocks}_patch_{args.patch_x}'
    else:
        summ_path = f'{folder}/{args.model}_{args.model_config}_height_{args.height}_filters_{args.filters}_blocks_{args.level_blocks}'

    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(summ_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        #f.flush()
        #f.close()

    flops = get_flops(model)
    flops_image = flops * (args.raw_input_shape[0] // args.patch_x) * (args.raw_input_shape[1] // args.patch_y)
    #with open(summ_path + '_flops', 'a+') as f:
    with open(summ_path, 'a+') as f:
        f.write(f'GFLOPS patch: {flops / 1e9}\n')
        f.write(f'GFLOPS image: {flops_image / 1e9}')
    #with open(summ_path + '_flops', 'w') as f:
    #    f.write(f'GFLOPS patch: {flops / 1e9}\n')
    #    f.write(f'GFLOPS image: {flops_image / 1e9}')


def params_and_flops_as_dict(model, patches_per_image, **kwargs):
    n_train_p = num_trainable_params(model)
    n_nontrain_p = num_non_trainable_params(model)
    flops_patch = get_flops(model)
    flops_image = flops_patch * patches_per_image
    return {
        'trainable_params': n_train_p,
        'nontrainable_params': n_nontrain_p,
        'flops_image': flops_image,
        'flops_patch': flops_patch
    }

def num_trainable_params(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


def num_non_trainable_params(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_variables])
