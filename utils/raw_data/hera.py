import numpy as np
#from sklearn.model_selection import train_test_split


def get_hera_data(data_path, rfi=None):
    rfi_models = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']

    if rfi is not None:
        (_,_,test_data,
         test_masks) = np.load(f'{data_path}/HERA_04-03-2022_{rfi}.pkl', allow_pickle=True)
        rfi_models.remove(rfi)

        (train_data,
         train_masks,_,_) = np.load(f'{data_path}/HERA_04-03-2022_{"-".join(rfi_models)}.pkl', allow_pickle=True)

    else:
        (train_data, train_masks,
          test_data, test_masks) = np.load(f'{data_path}/HERA_04-03-2022_all.pkl', allow_pickle=True)

    train_data[train_data==np.inf] = np.finfo(train_data.dtype).max
    test_data[test_data==np.inf] = np.finfo(test_data.dtype).max

    return train_data.astype('float32'), train_masks, test_data.astype('float32'), test_masks
