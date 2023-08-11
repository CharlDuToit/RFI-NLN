import numpy as np
#from sklearn.model_selection import train_test_split


def get_hera_charl_data(data_path):
    (train_data, train_masks,
    test_data, test_masks) = np.load(f'{data_path}/HERA_28-03-2023_all.pkl', allow_pickle=True)

    train_data[train_data==np.inf] = np.finfo(train_data.dtype).max
    test_data[test_data==np.inf] = np.finfo(test_data.dtype).max

    return train_data.astype('float32'), train_masks, test_data.astype('float32'), test_masks
