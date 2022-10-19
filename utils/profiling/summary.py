import os
import numpy as np


def save_summary(model, args):
    dir_path = 'outputs/{}/{}/{}'.format(args.model,
                                         args.anomaly_class,
                                         args.model_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(dir_path + '/model_summary', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))


def num_trainable_params(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])


def num_non_trainable_params(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.non_trainable_variables])
