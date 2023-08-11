import numpy as np
import os
import pandas as pd
# from utils.plotting import save_scatter
from utils import get_lofar_data, get_hera_data
from utils import get_patches, reconstruct, get_multichannel_patches, scale, replace_zeros_with_min
from utils import save_scatter
import matplotlib.pyplot as plt
import copy

from sklearn.metrics import (roc_curve,
                             auc,
                             f1_score,
                             recall_score,
                             precision_score,
                             accuracy_score,
                             average_precision_score,
                             jaccard_score,
                             roc_auc_score,
                             precision_recall_curve)


class ThresholdStats:
    def __init__(self,
                 data,
                 masks,
                 log=True,
                 scaling='image',
                 thresholding='image',
                 threshold_type='magnitude',
                 only_threshold_rfi_regions=False,
                 min_rfi_ratio=0.0,
                 num_thresholds=1000,
                 patch_size=64,
                 cell_size=8,
                 dir_path='./',
                 name='stats'):
        """

        Parameters
        ----------
        data
        masks
        scaling: image, patch, cell or box
        thresholding: image, patch, cell or box
        threshold_type: magnitude or std
        only_threshold_rfi_regions: only apply thresholding to regions that actually have RFI in them
        min_rfi_ratio: min rfi ratio for which to apply thresholding in a region
        num_thresholds: number of thresholds to apply, linearly spaced
        patch_size: 64 or 32 typically
        cell_size: 8 typically
        dir_path
        name
        """

        # ======================== Parameter validation =====================================

        if threshold_type not in ('magnitude', 'std'):
            raise ValueError('Incorrect threshold_type parameter')
        if scaling not in ('image', 'patch', 'cell', 'None'):
            raise ValueError('Incorrect scaling parameter')
        if thresholding not in ('image', 'patch', 'cell'):
            raise ValueError('Incorrect scaling parameter')
        if threshold_type == 'magnitude' and scaling == 'None':
            raise ValueError('Cannot threshold by magnitude if no scaling is applied')
        if scaling == 'patch' and thresholding == 'image':
            raise ValueError('Cannot scale per patch and apply threshold per image')
        if scaling == 'cell' and thresholding in ('image', 'patch'):
            raise ValueError('Cannot scale per cell and apply threshold per image or patch')

        # ======================== END Parameter validation =====================================

        thresholds = None
        if data is None or masks is None:  # So we want to read .csv files that have already been created
            print('data or masks is None, cannot compute masks')
            # patch_size = None  # still need it for naming files
            # cell_size = None
        else:
            data[..., 0] = replace_zeros_with_min(data[..., 0])  # (N_im, width, height, channels)

            if log:
                data[..., 0] = np.log(data[..., 0])

            if scaling == 'None':
                print('No scaling applied, using visibility magnitudes')

            if scaling == 'image':
                data[..., 0] = scale(data[..., 0], scale_per_image=True)

            if scaling == 'patch' or thresholding == 'patch':
                print('Creating patches')
                data = get_multichannel_patches(data, patch_size, patch_size, patch_size, patch_size)
                masks = get_multichannel_patches(masks, patch_size, patch_size, patch_size, patch_size)
                if scaling == 'patch':
                    data[..., 0] = scale(data[..., 0], scale_per_image=True)

            if scaling == 'cell' or thresholding == 'cell':
                print('Creating cells')
                data = get_multichannel_patches(data, cell_size, cell_size, cell_size, cell_size)
                masks = get_multichannel_patches(masks, cell_size, cell_size, cell_size, cell_size)
                if scaling == 'cell':
                    data[..., 0] = scale(data[..., 0], scale_per_image=True)

            # if only_threshold_rfi_regions:

            if threshold_type == 'magnitude':
                thresholds = np.linspace(0.0, 1.0, num_thresholds)
            if threshold_type == 'std':
                # thresholds = np.linspace(0.0, 1.0, num_thresholds)
                stds = np.std(data, axis=(1, 2, 3))
                means = np.mean(data, axis=(1, 2, 3))
                maxes = np.max(data, axis=(1, 2, 3))
                mins = np.min(data, axis=(1, 2, 3))
                stds_from_mean_low = np.min((mins - means) / stds)
                stds_from_mean_high = np.max((maxes - means) / stds)
                thresholds = np.linspace(stds_from_mean_low, stds_from_mean_high, num_thresholds)

        self.thresholds = thresholds
        # self.masks_inferred = None

        self.data = data
        self.masks = masks
        self.log = log
        self.patch_size = patch_size
        self.scaling = scaling
        self.thresholding = thresholding
        self.threshold_type = threshold_type
        self.num_thresholds = num_thresholds
        self.only_threshold_rfi_regions = only_threshold_rfi_regions
        self.cell_size = cell_size
        self.min_rfi_ratio = min_rfi_ratio
        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        self.name = name

        # Remove ?
        self.df = None
        temp_dict = data_masks_stats(np.array([0, 1]), np.array([True, False]))
        self.stats_dict = {}
        for key in temp_dict.keys():
            self.stats_dict[key] = []

    def main(self):
        self.calc_stats()
        self.save_csv()
        self.save_agg_csv()

    def save_per_region_scatter(self):
        # c = c[np.any(b, axis=(1,2))]
        i = 0
        for im, mask in zip(self.data, self.masks):
            if self.only_threshold_rfi_regions:
                if np.mean(mask) <= self.min_rfi_ratio:
                    continue

            rfi_stds_from_mean, nonrfi_stds_from_mean = ThresholdStats.image_mask_to_rfi_nonrfi_stds_from_mean(im, mask)
            # list_of_list_of_x = [magnitudes for rfi, magnitudes for nonrfi]
            # list_of_list_of_y = [stds_from_mean for rfi, stds_from_mean for nonrfi]
            save_scatter(
                [rfi_stds_from_mean[:, 0], nonrfi_stds_from_mean[:, 0]],
                [rfi_stds_from_mean[:, 1]+0.2, nonrfi_stds_from_mean[:, 1]],
                labels=['rfi', 'nonrfi'],
                dir_path=self.dir_path,
                file_name=f'{self.get_file_name()}_{i}',
                size=1,
                xlabel='magnitude',
                ylabel='stds from mean'
            )
            i += 1

    @staticmethod
    def calc_precision_scores(y_true, y_pred, thresholds):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        precision_scores = [precision_score(y_true, y_pred > thr) for thr in thresholds]
        return np.array(precision_scores)

    @staticmethod
    def calc_recall_scores(y_true, y_pred, thresholds):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        precision_scores = [recall_score(y_true, y_pred > thr) for thr in thresholds]
        return np.array(precision_scores)

    @staticmethod
    def stds_from_mean_to_thresholds(one_image, stds_from_mean):
        mean = np.mean(one_image)
        std = np.std(one_image)
        thresholds = stds_from_mean * std + mean
        return thresholds

    @staticmethod
    def thresholds_to_stds_from_mean(one_image, thresholds):
        mean = np.mean(one_image)
        std = np.std(one_image)
        stds_from_mean = (thresholds - mean)/std
        return stds_from_mean

    @staticmethod
    def image_mask_to_rfi_nonrfi_stds_from_mean(image, mask):
        # Used for scatter plotsm
        mask = mask.flatten().astype(bool)
        image = image.flatten()
        std = np.std(image)
        mean = np.mean(image)
        stds_from_mean = (image - mean) / std
        rfi = np.column_stack([image[mask], stds_from_mean[mask]])
        nonrfi = np.column_stack([image[~mask], stds_from_mean[~mask]])
        return rfi, nonrfi

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
        # -------------
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

    def get_file_name(self, agg=False, query=None, csv=False):
        file_name = self.name
        file_name += f'_log-{self.log}'
        file_name += f'_s-{self.scaling}'
        file_name += f'_t-{self.thresholding}'
        file_name += f'_tt-{self.threshold_type}'
        file_name += f'_otrfi-{self.only_threshold_rfi_regions}'
        file_name += f'_mrfi-{self.min_rfi_ratio}'
        file_name += f'_p{self.patch_size}'
        file_name += f'_c{self.cell_size}'

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


def data_masks_stats(data, masks):
    rfi = data[masks]
    nonrfi = data[np.invert(masks)]

    mean = data.mean()
    std = data.std()
    std_over_mean = std / mean
    rfi_ratio = masks.mean()

    rfi_mean = rfi.mean() if rfi_ratio > 0.0 else 0.0
    rfi_std = rfi.std() if rfi_ratio > 0.0 else 0.0
    rfi_min = rfi.min() if rfi_ratio > 0.0 else 0.0
    rfi_max = rfi.max() if rfi_ratio > 0.0 else 0.0
    rfi_min_data_percentile = (data < rfi_min).mean() if rfi_ratio > 0.0 else 0.0
    rfi_min_stds_from_mean = (rfi_min - mean) / std if rfi_ratio > 0.0 else 0.0
    rfi_min_means_from_mean = (rfi_min - mean) / mean if rfi_ratio > 0.0 else 0.0
    rfi_min_div_mean = rfi_min / mean if rfi_ratio > 0.0 else 0.0
    rfi_max_over_rfi_min = rfi_max / rfi_min if rfi_min > 0.0 else 0.0

    rfi_perc10 = np.percentile(rfi, 10) if rfi_ratio > 0.0 else 0.0
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
    nonrfi_max_div_mean = nonrfi_max / mean if rfi_ratio < 1.0 else 0.0
    nonrfi_max_over_nonrfi_min = nonrfi_max / nonrfi_min if nonrfi_min > 0.0 else 0.0

    nonrfi_perc90 = np.percentile(nonrfi, 90) if rfi_ratio < 1.0 else 0.0
    nonrfi_perc90_data_percentile = (data < nonrfi_perc90).mean() if rfi_ratio < 1.0 else 0.0
    nonrfi_perc90_stds_from_mean = (nonrfi_perc90 - mean) / std if rfi_ratio < 1.0 else 0.0
    nonrfi_max_over_nonrfi_perc90 = rfi_perc10 / rfi_min if rfi_min > 0.0 else 0.0

    nonrfi_max_over_rfi_min = nonrfi_max / rfi_min if rfi_min > 0.0 else 0.0
    overlap_ratio = np.logical_and((rfi_min < data), (data < nonrfi_max)).mean() if 1.0 > rfi_ratio > 0.0 else 0.0
    rfi_overlap_ratio = (rfi < nonrfi_max).mean() if 1.0 > rfi_ratio > 0.0 else 0.0  # what ratio of rfi is in overlap
    nonrfi_overlap_ratio = (
                rfi_min < nonrfi).mean() if 1.0 > rfi_ratio > 0.0 else 0.0  # what ratio of nonrfi is in overlap

    nonrfi_perc90_minus_rfi_perc10 = nonrfi_perc90_data_percentile - rfi_perc10_data_percentile
    nonrfi_perc90_over_rfi_perc10 = nonrfi_perc90 / rfi_perc10 if rfi_perc10 > 0.0 else 0.0

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
        rfi_max_over_rfi_min=rfi_max_over_rfi_min,

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


def main():
    from utils.raw_data import load_raw_data
    train_data, train_masks, test_data, test_masks = load_raw_data(
        data_name='HERA_CHARL',
        data_path='/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/',
        # data_path='/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/',
        lofar_subset='all')
    thresh_stats = ThresholdStats(test_data[0:108],
                                  test_masks[0:108],
                                  dir_path='./thresh_hera_test/log_cell_8',
                                  log=True,
                                  scaling='cell',
                                  thresholding='cell',
                                  threshold_type='std',
                                  only_threshold_rfi_regions=True,
                                  min_rfi_ratio=0.0,
                                  patch_size=64,
                                  cell_size=8,
                                  num_thresholds=100,
                                  name='lofar_test')
    thresh_stats.save_per_region_scatter()


if __name__ == '__main__':
    main()
