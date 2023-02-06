import numpy as np
from scipy.io import loadmat
import os


def get_astron_data(data_path):
    # file_names will change, then we can loop through
    # Alternatively just merge something and save to disc prior to calling this function
    data_name = ['ant_fft_000_094_t4032_f4096',
                 'ant_fft_000_094_t4096_f4096',
                 'ant_fft_000_094_t8128_f8192',
                 'ant_fft_000_094_t12160_f16384']
    file = data_name[1]
    file = os.path.join(data_path, f'{file}.mat')
    raw_train_data = np.expand_dims(loadmat(file)['sbdata'], (0, -1)) / 1e3
    raw_train_masks = None
    raw_test_data = None
    raw_test_masks = None

    return raw_train_data, raw_train_masks, raw_test_data, raw_test_masks
