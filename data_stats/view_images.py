import numpy as np
import os
import pandas as pd
from utils.plotting import save_scatter
from utils.data import get_lofar_data, get_hera_data
from utils.data import get_patches, reconstruct
from utils.plotting import save_data_inferred
import matplotlib.pyplot as plt
import copy
import pickle


def view(data, masks):
    pass


lofar_broken_indexes = [986,
                        1086,
                        1264,
                        1473,
                        1598,
                        1750,
                        2050,
                        2094,
                        2430,
                        2518,
                        2566,
                        3135,
                        3639,
                        3718,
                        3872,
                        4026,
                        4104,
                        4336,
                        4417,
                        4594,
                        4663,
                        4838,
                        4841,
                        4894,
                        4926,
                        5476,
                        5882,
                        5995,
                        6246,
                        6827,
                        7074,
                        7113,
                        7219,
                        7381,
                        7462,
                        ]

if __name__ == '__main__':
    class Namespace:
        def __init__(self):
            self.data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
            self.lofar_subset = 'full'
            self.rfi = None


    (raw_train_data, raw_train_masks, raw_test_data, raw_test_masks) = get_lofar_data(Namespace())
    # save_data_inferred(dir_path='./broken_images',
    #                    data=np.log(raw_train_data[lofar_broken_indexes]),
    #                    masks_inferred=raw_train_masks[lofar_broken_indexes],
    #                    figsize=(100,200)
    #                    )
    good_index = np.ones(7500, dtype=bool)
    good_index[lofar_broken_indexes] = False
    f_name = 'LOFAR_Full_RFI_dataset.pkl'
    pickle.dump([raw_train_data[good_index], raw_train_masks[good_index], raw_test_data, raw_test_masks], open(f_name, 'wb'), protocol=4)
