import numpy as np
import os
import pandas as pd
#from utils.plotting import save_scatter
from utils import get_lofar_data, get_hera_data
from utils import get_patches, reconstruct
import matplotlib.pyplot as plt
import copy


class DataStats:
    def __init__(self, data, patch_size=None, dir_path='./', name='stats'):

        if data is None:
            patch_size = None

        if patch_size is not None:
            p_size = (1, patch_size, patch_size, 1)
            s_size = (1, patch_size, patch_size, 1)
            rate = (1, 1, 1, 1)
            data = get_patches(data, p_size, s_size, rate, 'VALID')

        self.data = data
        self.patch_size = patch_size
        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.name = name
        self.df = None

        temp_dict = data_stats(np.array([0, 1]))
        self.stats_dict = {}
        for key in temp_dict.keys():
            self.stats_dict[key] = []

    def main(self):
        self.calc_stats()
        self.save_csv()
        self.save_agg_csv()

    def calc_stats(self):
        for d in zip(self.data):
            image_dict = data_stats(d)
            self.append_dict(image_dict)
        self.df = pd.DataFrame.from_dict(self.stats_dict)

    # def calc_stats_in_chunks(self, chunk_size=100):
    #     for d, m in zip(self.data, self.masks):
    #         image_dict = data_masks_stats(d, m)
    #         self.append_dict(image_dict)
    #     self.df = pd.DataFrame.from_dict(self.stats_dict)

    def append_dict(self, image_dict):
        for key in image_dict.keys():
            self.stats_dict[key].append(image_dict[key])

    def save_csv(self, query=None):
        file = os.path.join(self.dir_path, self.get_file_name(agg=False, query=query))
        if query is not None:
            df = self.df.query(query)
        else:
            df = self.df
        df.to_csv(file)

    def load_csv(self):
        file = os.path.join(self.dir_path, self.get_file_name(agg=False, query=None))
        self.df = pd.read_csv(file)

    def save_agg_csv(self, query=None):
        file = os.path.join(self.dir_path, self.get_file_name(agg=True, query=query))
        if query is not None:
            df = self.df.query(query)
        else:
            df = self.df
        df = pd.DataFrame(aggregate(df))
        df.to_csv(file)

    def save_scatter(self, xaxis, yaxis, query=None):
        file_name = self.get_file_name(agg=False, query=query, csv=False)
        file_name += f'_{xaxis}_{yaxis}'
        if query is not None:
            df = self.df.query(query)
        else:
            df = self.df
        x = list(df[xaxis])
        y = list(df[yaxis])
        # save_scatter(x, y, dir_path=self.dir_path, file_name=file_name,
        #              xlabel=xaxis, ylabel=yaxis, title=None)
        #-------------
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.scatter(x, y, s=2)
        ax.set_xlabel(xaxis, fontsize=20)
        ax.tick_params(axis='x', labelsize=20, which='major')
        ax.set_ylabel(yaxis, fontsize=20)
        ax.tick_params(axis='y', labelsize=20, which='major')
        file = os.path.join(self.dir_path, f'{file_name}.png')
        fig.savefig(file)
        print(f'Saved {file_name}.png')
        plt.close('all')

    def get_file_name(self, agg, query=None, csv=True):
        file_name = self.name
        if self.patch_size is not None:
            file_name += f'_p{self.patch_size}'
        if query is not None:
            file_name += f'_{query}'
        if agg:
            file_name += '_agg'
        if csv:
            file_name += '.csv'
        return file_name


def aggregate(dict_or_df):
    _min, _max, mean, std = {'agg': 'min'}, {'agg': 'max'}, {'agg': 'mean'}, {'agg': 'std'},
    for key in dict_or_df.keys():
        _min[key] = np.min(dict_or_df[key])
        _max[key] = np.max(dict_or_df[key])
        mean[key] = np.mean(dict_or_df[key])
        std[key] = np.std(dict_or_df[key])
    return _min, _max, mean, std


def data_stats(data):

    _data = np.abs(data)
    mean = np.mean(_data)
    std = np.std(_data)
    std_over_mean = std/mean
    min_ = np.min(_data)
    max_ = np.max(_data)
    perc_10 = np.percentile(_data, 10)
    perc_50 = np.percentile(_data, 50)
    perc_90 = np.percentile(_data, 90)

    return dict(
        mean=mean,
        std=std,
        std_over_mean=std_over_mean,
        min=min_,
        max=max_,
        perc_10=perc_10,
        perc_50=perc_50,
        perc_90=perc_90,
    )


# if __name__ == '__main__':