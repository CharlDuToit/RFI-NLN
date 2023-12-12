from .lofar import get_lofar_data
from .astron import get_astron_data
from .hera import get_hera_data
from .hera_charl import get_hera_charl_data
from .hera_charl_aof import get_hera_charl_aof_data

import time


def load_raw_data(data_path, data_name, rfi=None, lofar_subset='full', **kwargs):
    start = time.time()

    if data_name == 'ASTRON':
        train_data, train_masks, test_data, test_masks = get_astron_data(data_path)
    elif data_name == 'LOFAR':
        train_data, train_masks, test_data, test_masks = get_lofar_data(data_path, lofar_subset)
    elif data_name == 'HERA':
        train_data, train_masks, test_data, test_masks = get_hera_data(data_path, rfi)
    elif data_name == 'HERA_CHARL':
        train_data, train_masks, test_data, test_masks = get_hera_charl_data(data_path)
    elif data_name == 'HERA_CHARL_AOF':
        train_data, train_masks, test_data, test_masks = get_hera_charl_aof_data(data_path)
    else:
        raise ValueError(f'data_name {data_name} not supported')

    print(f'Data load time: {time.time() - start} sec')
    print(f'Test Data Shape: {test_masks.shape}')
    return train_data, train_masks, test_data, test_masks

