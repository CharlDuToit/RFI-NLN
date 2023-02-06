
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


def loss_file(output_path, model_class, anomaly_class, model_name, model_type=None, data='val', **kwargs):
    f = ''
    if model_type is not None:
        f = model_type
    if data is not None:
        if f:
            f += '_' + data
        else:
            f = data
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
                            f'{model_type}_summary.txt')


def checkpoint_file(output_path, model_class, anomaly_class, model_name, model_type=None, **kwargs):
    if model_type is None:
        return os.path.join(checkpoints_dir(output_path, model_class, anomaly_class, model_name), f'checkpoint_{model_class}')
    else:
        return os.path.join(checkpoints_dir(output_path, model_class, anomaly_class, model_name), f'checkpoint_{model_type}')
