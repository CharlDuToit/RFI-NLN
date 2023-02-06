import numpy as np
import os
import pandas as pd
#from utils.plotting import save_scatter
from utils import get_lofar_data, get_hera_data
from utils import get_patches, reconstruct
import matplotlib.pyplot as plt
import copy

class DataStats:
    def __init__(self, data, masks, patch_size=None, dir_path='./', name='stats'):

        if data is None or masks is None:
            patch_size = None

        if patch_size is not None:
            p_size = (1, patch_size, patch_size, 1)
            s_size = (1, patch_size, patch_size, 1)
            rate = (1, 1, 1, 1)
            data = get_patches(data, p_size, s_size, rate, 'VALID')
            masks = get_patches(masks.astype('float'), p_size, s_size, rate, 'VALID').astype(bool)
            #print('Created patches')

        self.data = data
        self.masks = masks
        self.patch_size = patch_size
        self.dir_path = dir_path
        self.name = name
        self.df = None

        temp_dict = data_masks_stats(np.array([0, 1]), np.array([True, False]))
        self.stats_dict = {}
        for key in temp_dict.keys():
            self.stats_dict[key] = []

    def calc_stats(self):
        for d, m in zip(self.data, self.masks):
            image_dict = data_masks_stats(d, m)
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

    def fit_hyperbola(self, xaxis, yaxis, query=None):
        from scipy.optimize import curve_fit
        def func(x, g, a, b, c):
            return c + 1.0 / (g* x**3 + a * x*x + b*x )

        #def func(x, a, b):
            #return  1 / (a * x*x + b*x)

        x = self.df[xaxis]
        y = self.df[yaxis]
        indexes = np.argsort(x)
        x = x[indexes]
        y = y[indexes]

        popt, pcov = curve_fit(func, x, y)
        print(popt)

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.scatter(x, y, s=1, c='red')
        #ax.plot(x, func(x, *popt))
        #ax.plot(x, func(x, -0.25))
        #ax.plot(x, func(x, 0.00, -0.2, -0.9, -0.03 )) # LOFAR rfi_min
        ax.plot(x, func(x, 0.00, 0.02, 0.01, 0.2 )) # LOFAR nonrfi_max
        # 1 ,

        #ax.set_ylim(-5, 0) # rfi min
        ax.set_ylim(0, 60)
        #ax.set_xlim(0, 10)

        file_name = self.get_file_name(agg=False, query=query, csv=False)
        file_name += f'_{xaxis}_{yaxis}_FITTED'
        file = os.path.join(self.dir_path, f'{file_name}.png')
        fig.savefig(file)
        print(f'Saved {file_name}.png')
        #plt.close('all')



def aggregate(dict_or_df):
    _min, _max, mean, std = {'agg': 'min'}, {'agg': 'max'}, {'agg': 'mean'}, {'agg': 'std'},
    for key in dict_or_df.keys():
        _min[key] = np.min(dict_or_df[key])
        _max[key] = np.max(dict_or_df[key])
        mean[key] = np.mean(dict_or_df[key])
        std[key] = np.std(dict_or_df[key])
    return _min, _max, mean, std



def data_masks_stats(data, masks):
    rfi = data[masks]
    nonrfi = data[np.invert(masks)]

    mean = data.mean()
    std = data.std()
    std_over_mean = std/mean
    rfi_ratio = masks.mean()

    rfi_mean = rfi.mean() if rfi_ratio > 0.0 else 0.0
    rfi_std = rfi.std() if rfi_ratio > 0.0 else 0.0
    rfi_min = rfi.min() if rfi_ratio > 0.0 else 0.0
    rfi_max = rfi.max() if rfi_ratio > 0.0 else 0.0
    rfi_min_data_percentile = (data < rfi_min).mean() if rfi_ratio > 0.0 else 0.0
    rfi_min_stds_from_mean = (rfi_min - mean) / std if rfi_ratio > 0.0 else 0.0
    rfi_min_means_from_mean = (rfi_min - mean) / mean if rfi_ratio > 0.0 else 0.0
    rfi_min_div_mean = rfi_min/mean if rfi_ratio > 0.0 else 0.0
    rfi_max_over_rfi_min = rfi_max / rfi_min if rfi_min > 0.0 else 0.0

    rfi_perc10 = np.percentile(rfi, 10)
    rfi_perc10_data_percentile = (data < rfi_perc10).mean() if rfi_ratio > 0.0 else 0.0
    rfi_perc10_stds_from_mean = (rfi_perc10 - mean) / std if rfi_ratio > 0.0 else 0.0
    rfi_perc10_over_rfi_min = rfi_perc10 / rfi_min if rfi_min > 0.0 else 0.0


    nonrfi_mean = nonrfi.mean() if rfi_ratio < 1.0 else 0.0
    nonrfi_std = nonrfi.std() if rfi_ratio < 1.0 else 0.0
    nonrfi_min = nonrfi.min() if rfi_ratio < 1.0 else 0.0
    nonrfi_max = nonrfi.max() if rfi_ratio < 1.0 else 0.0
    nonrfi_max_data_percentile = (data < nonrfi_max).mean() if rfi_ratio < 1.0 else 0.0
    nonrfi_max_stds_from_mean = (nonrfi_max - mean) / std if rfi_ratio < 1.0 else 0.0
    nonrfi_max_means_from_mean = (nonrfi_max - mean) / mean if rfi_ratio < 1.0 else 0.0
    nonrfi_max_div_mean = nonrfi_max/mean if rfi_ratio < 1.0 else 0.0
    nonrfi_max_over_nonrfi_min = nonrfi_max / nonrfi_min if nonrfi_min > 0.0 else 0.0

    nonrfi_perc90 = np.percentile(nonrfi, 90)
    nonrfi_perc90_data_percentile = (data < nonrfi_perc90).mean() if rfi_ratio > 0.0 else 0.0
    nonrfi_perc90_stds_from_mean = (nonrfi_perc90 - mean) / std if rfi_ratio > 0.0 else 0.0
    nonrfi_max_over_nonrfi_perc90 = rfi_perc10 / rfi_min if rfi_min > 0.0 else 0.0

    nonrfi_max_over_rfi_min = nonrfi_max/rfi_min if rfi_min > 0.0 else 0.0
    overlap_ratio = np.logical_and((rfi_min < data), (data < nonrfi_max)).mean() if 1.0 > rfi_ratio > 0.0 else 0.0
    rfi_overlap_ratio = (rfi < nonrfi_max).mean() if 1.0 > rfi_ratio > 0.0 else 0.0 # what ratio of rfi is in overlap
    nonrfi_overlap_ratio = (rfi_min < nonrfi).mean() if 1.0 > rfi_ratio > 0.0 else 0.0 # what ratio of nonrfi is in overlap

    nonrfi_perc90_minus_rfi_perc10 = nonrfi_perc90_data_percentile - rfi_perc10_data_percentile
    nonrfi_perc90_over_rfi_perc10 = nonrfi_perc90/rfi_perc10 if rfi_perc10 > 0.0 else 0.0

    return dict(
        mean=mean,
        std=std,
        std_over_mean=std_over_mean,
        rfi_ratio=rfi_ratio,
        rfi_mean=rfi_mean,
        rfi_std=rfi_std,
        rfi_min=rfi_min,
        rfi_max=rfi_max,
        rfi_min_data_percentile=rfi_min_data_percentile,
        rfi_min_stds_from_mean=rfi_min_stds_from_mean,
        rfi_min_means_from_mean=rfi_min_means_from_mean,
        rfi_min_div_mean=rfi_min_div_mean,
        rfi_max_over_rfi_min = rfi_max_over_rfi_min,

        rfi_perc10=rfi_perc10,
        rfi_perc10_data_percentile=rfi_perc10_data_percentile,
        rfi_perc10_stds_from_mean=rfi_perc10_stds_from_mean,
        rfi_perc10_over_rfi_min=rfi_perc10_over_rfi_min,

        nonrfi_mean=nonrfi_mean,
        nonrfi_std=nonrfi_std,
        nonrfi_min=nonrfi_min,
        nonrfi_max=nonrfi_max,
        nonrfi_max_data_percentile=nonrfi_max_data_percentile,
        nonrfi_max_stds_from_mean=nonrfi_max_stds_from_mean,
        nonrfi_max_means_from_mean=nonrfi_max_means_from_mean,
        nonrfi_max_div_mean=nonrfi_max_div_mean,
        nonrfi_max_over_nonrfi_min=nonrfi_max_over_nonrfi_min,

        nonrfi_perc90=nonrfi_perc90,
        nonrfi_perc90_data_percentile=nonrfi_perc90_data_percentile,
        nonrfi_perc90_stds_from_mean=nonrfi_perc90_stds_from_mean,
        nonrfi_max_over_nonrfi_perc90=nonrfi_max_over_nonrfi_perc90,

        nonrfi_max_over_rfi_min=nonrfi_max_over_rfi_min,
        overlap_ratio=overlap_ratio,
        rfi_overlap_ratio=rfi_overlap_ratio,
        nonrfi_overlap_ratio=nonrfi_overlap_ratio,

        nonrfi_perc90_minus_rfi_perc10=nonrfi_perc90_minus_rfi_perc10,
        nonrfi_perc90_over_rfi_perc10=nonrfi_perc90_over_rfi_perc10
    )


if __name__ == '__main__':

    data_path = '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'
    (raw_train_data, raw_train_masks, raw_test_data, raw_test_masks) = get_lofar_data(data_path)
    #dir_path = './lofar_test_patch_512'
    dir_path = './lofar_train_patch_512'

    print(raw_train_data.shape)
    print(raw_test_data.shape)

    # -----------------------------------------
    files = []
    for i in range(0,7500,500):
        data_stats = DataStats(raw_train_data[i:i+500, ...],
                               raw_train_masks[i:i+500, ...],
                               dir_path=dir_path,
                               patch_size=512,
                               name=f'lofar_train_{i}')
        data_stats.calc_stats()
        data_stats.save_csv()
        files.append(os.path.join(dir_path, data_stats.get_file_name(agg=False, csv=True)))
    df = pd.read_csv(files[0])
    for f in files[1:]:
        df = pd.concat([df, pd.read_csv(f)], ignore_index=True)
    data_stats = DataStats(None, None, dir_path=dir_path, patch_size=512, name=f'lofar_train')
    data_stats.df = df
    # -----------------------------------------

    # Entire dataset
    #data_stats = DataStats(raw_test_data, raw_test_masks, dir_path=dir_path, patch_size=512, name='lofar_test')
    #data_stats = DataStats(raw_train_data, raw_train_masks, dir_path=dir_path, patch_size=512, name='lofar_train')
    #data_stats = DataStats(None, None, dir_path='./hera_train_patch_512', patch_size=None, name='hera_train_p512')
    #data_stats = DataStats(None, None, dir_path='./lofar_train_patch_512', patch_size=None, name='lofar_train')
    #data_stats.load_csv()

    #data_stats.fit_hyperbola('std_over_mean', 'nonrfi_max_stds_from_mean')
    #data_stats.fit_hyperbola('std_over_mean', 'rfi_min_stds_from_mean')

    #data_stats.save_scatter('std_over_mean', 'rfi_ratio')

    #data_stats.calc_stats()

    data_stats.save_csv()
    data_stats.save_agg_csv()
    data_stats.save_csv(query='rfi_ratio>0.0')
    data_stats.save_agg_csv(query='rfi_ratio>0.0')

    try:
        data_stats.save_csv(query='rfi_ratio==0.0')
        data_stats.save_agg_csv(query='rfi_ratio==0.0')
    except:
        print('e1')

    try:
        data_stats.save_csv(query='rfi_ratio==1.0')
        data_stats.save_agg_csv(query='rfi_ratio==1.0')
    except:
        print('e2')

    query = 'rfi_ratio>0.0 and rfi_ratio<1.0 and overlap_ratio==0.0'
    data_stats.save_csv(query=query)
    data_stats.save_agg_csv(query=query)

    query = 'rfi_ratio>0.0 and rfi_ratio<1.0 and overlap_ratio>0.0'
    data_stats.save_csv(query=query)
    data_stats.save_agg_csv(query=query)

    query = 'rfi_ratio>0.0 and rfi_ratio<1.0 and overlap_ratio>0.0 and rfi_min > 0.0 and rfi_perc10 > 0.0'
    data_stats.save_csv(query=query)
    data_stats.save_agg_csv(query=query)

    # print('Saved CSVs')
    #
    # data_stats = DataStats(None, None, dir_path='./lofar_train_patch_64', patch_size=None, name='lofar_train')
    # data_stats.load_csv()
    #
    data_stats.save_scatter('std_over_mean', 'nonrfi_max_data_percentile', query=query)
    data_stats.save_scatter('overlap_ratio', 'rfi_min_data_percentile', query=query)
    data_stats.save_scatter('rfi_ratio', 'nonrfi_max_data_percentile', query=query)

    data_stats.save_scatter('std_over_mean', 'nonrfi_max_stds_from_mean', query=query)
    data_stats.save_scatter('std_over_mean', 'rfi_min_stds_from_mean', query=query)
    data_stats.save_scatter('overlap_ratio', 'nonrfi_max_stds_from_mean', query=query)
    data_stats.save_scatter('overlap_ratio', 'rfi_min_stds_from_mean', query=query)

    data_stats.save_scatter('rfi_ratio', 'nonrfi_max_stds_from_mean', query=query)
    data_stats.save_scatter('rfi_ratio', 'rfi_min_stds_from_mean', query=query)

    data_stats.save_scatter('mean', 'rfi_min_div_mean', query=query)
    data_stats.save_scatter('mean', 'nonrfi_max_div_mean', query=query)

    data_stats.save_scatter('mean', 'nonrfi_max_means_from_mean', query=query)
    data_stats.save_scatter('mean', 'rfi_min_means_from_mean', query=query)

    data_stats.save_scatter('std_over_mean', 'nonrfi_max_means_from_mean', query=query)
    data_stats.save_scatter('std_over_mean', 'rfi_min_means_from_mean', query=query)

    data_stats.save_scatter('std_over_mean', 'nonrfi_perc90_stds_from_mean', query=query)
    data_stats.save_scatter('std_over_mean', 'rfi_perc10_stds_from_mean', query=query)

    data_stats.save_scatter('std_over_mean', 'nonrfi_perc90_data_percentile', query=query)
    data_stats.save_scatter('std_over_mean', 'rfi_perc10_data_percentile', query=query)

    data_stats.save_scatter('std_over_mean', 'nonrfi_max_over_nonrfi_perc90', query=query)
    data_stats.save_scatter('std_over_mean', 'rfi_perc10_over_rfi_min', query=query)

    #data_stats.save_scatter('std_over_mean', 'nonrfi_max_over_nonrfi_perc90', query=query)
    #data_stats.save_scatter('std_over_mean', 'rfi_perc10_over_nonrfi_perc90', query=query)



    #
    # # Not useful scatters
    # # data_stats.save_scatter('mean', 'rfi_min_data_percentile', query='rfi_ratio>0.0 and rfi_overlap_ratio>0.0')
    # # data_stats.save_scatter('mean', 'nonrfi_max_data_percentile', query='rfi_ratio>0.0 and rfi_overlap_ratio>0.0')
    # # data_stats.save_scatter('overlap_ratio', 'nonrfi_max_data_percentile', query='rfi_ratio>0.0 and rfi_overlap_ratio>0.0')
    # # data_stats.save_scatter('rfi_ratio', 'rfi_min_data_percentile', query='rfi_ratio>0.0 and rfi_overlap_ratio>0.0')
    # # data_stats.save_scatter('std_over_mean', 'rfi_min_data_percentile', query='rfi_ratio>0.0 and rfi_overlap_ratio>0.0')
