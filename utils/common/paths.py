
import os

# EPOCHS_DIR = 'epochs'
# CHECKPOINTS_DIR = 'checkpoints'
# LOSSES_DIR = 'losses'
# INFERRED_DIR = 'inferred'

# ------------------------------------ Directories -----------------------------

def model_dir(output_path, model_class, anomaly_class, model_name, **kwargs):
    _dir = os.path.join(output_path, model_class, anomaly_class, model_name)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


def epochs_dir(output_path, model_class, anomaly_class, model_name, **kwargs):
    _dir =  os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), 'epochs')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


def checkpoints_dir(output_path, model_class, anomaly_class, model_name, **kwargs):
    _dir =  os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), 'training_checkpoints')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


def losses_dir(output_path, model_class, anomaly_class, model_name, **kwargs):
    _dir = os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), 'losses')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


def inferred_dir(output_path, model_class, anomaly_class, model_name, **kwargs):
    _dir =  os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), 'inferred')
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


def inferred_data_dir(output_path, model_class, anomaly_class, model_name, data_name, **kwargs):
    _dir =  os.path.join(inferred_dir(output_path, model_class, anomaly_class, model_name), data_name)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir


# ------------------------------------ Files -----------------------------


def loss_file(output_path, model_class, anomaly_class, model_name, model_type=None, data_subset='val', **kwargs):
    f = ''
    if model_type is not None:
        f = model_type
    if data_subset is not None:
        if f:
            f += '_' + data_subset
        else:
            f = data_subset
    if f:
        f += '_epoch_losses.txt'
    else:
        f = 'epoch_losses.txt'
    return os.path.join(losses_dir(output_path, model_class, anomaly_class, model_name), f)


def solution_file(output_path, model_class, anomaly_class, model_name, task=None, **kwargs):
    if task is None:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), 'solution.txt')
    else:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), f'{task}_solution.txt')


def summary_file(output_path, model_class, anomaly_class, model_name, model_type=None, **kwargs):
    if model_type is None:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), f'{model_class}_summary.txt')
    else:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name),
                            f'{model_class}_{model_type}_summary.txt')


def model_plot_file(output_path, model_class, anomaly_class, model_name, model_type=None, **kwargs):
    if model_type is None:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), f'{model_class}_plot.png')
    else:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name),
                            f'{model_class}_{model_type}_plot.png')


def checkpoint_file(output_path, model_class, anomaly_class, model_name, model_type=None, **kwargs):
    # if rfi_set == 'separate':
    #     low_file = checkpoint_file(output_path, model_class, anomaly_class, model_name, model_type='low')
    #     high_file = checkpoint_file(output_path, model_class, anomaly_class, model_name, model_type='high')
    #     return low_file, high_file
    if model_type is None:
        return os.path.join(checkpoints_dir(output_path, model_class, anomaly_class, model_name), f'checkpoint_{model_class}')
    else:
        return os.path.join(checkpoints_dir(output_path, model_class, anomaly_class, model_name), f'checkpoint_{model_class}_{model_type}')


def training_metrics_file(output_path, model_class, anomaly_class, model_name, model_type=None, data_subset=None, **kwargs):
    f = ''
    f += model_class
    if model_type is not None:
        f += '_' + model_type
    if data_subset is not None:
        f += '_' + data_subset

    return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), f'{f}_training_metrics.png')


def rfi_file(output_path, model_class, anomaly_class, model_name, data_subset='val', model_type=None, **kwargs):
    if model_type is None:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), f'{data_subset}_rfi.csv')
    else:
        return os.path.join(model_dir(output_path, model_class, anomaly_class, model_name), f'{model_type}_{data_subset}_rfi.csv')

