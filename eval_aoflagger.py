from utils import main_args, load_raw_data, to_dict, flag_data, aof_recall_precision_f1_fpr, save_csv, Namespace, \
    save_image_masks_batches
import time
import datetime, pickle

import numpy as np
from h5py import File
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score
from scipy.io import savemat
import pandas as pd
import matplotlib as plt
import os

from utils import save_image_masks_masksinferred_batches, flag_data


def main():
    ns = Namespace(data_name='HERA_CHARL_AOF',
                   data_path='/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/',
                   rfi_threshold=None,
                   seed='AOF_HERA_CHARL_20_july',
                   output_path='./outputs'
                   )
    kwargs = to_dict(ns)

    # '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/'
    # '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'

    train_data, train_masks, test_data, test_masks = load_raw_data(**kwargs)

    subset = 'test'
    if subset == 'test':
        data = test_data
        masks = test_masks
    else:
        data = train_data
        masks = train_masks

    for rfi_threshold in (10,):
        print('rfi_threshold : ', rfi_threshold)

        kwargs['rfi_threshold'] = rfi_threshold

        start = time.time()
        masks_aof = flag_data(data, **kwargs)
        infer_time = time.time() - start

        # image_time = infer_time / data.shape[0]
        # recall, precision, f1, fpr = aof_recall_precision_f1_fpr(masks, masks_aof)
        # print('    f1: ', f1)
        #
        # results = dict(subset=subset,
        #                image_time=image_time,
        #                f1=f1,
        #                precision=precision,
        #                recall=recall,
        #                fpr=fpr)
        # csv_kwargs = {**kwargs, **results}
        # save_csv(results_dict=csv_kwargs, **kwargs)

        # save_image_masks_batches(dir_path='./outputs/AOF/'+kwargs['data_name']+f'/{rfi_threshold}',
        #                          data=test_data,
        #                          masks=masks_aof,
        #                          batch_size=28)

    # Save to pickle
    # f_name = '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/HERA_AOF_{}_all.pkl'.format(
    #     datetime.datetime.now().strftime("%d-%m-%Y"))
    # pickle.dump([data, masks_aof, test_data, test_masks], open(f_name, 'wb'), protocol=4)
    # print('{} saved!'.format(f_name))
    from scipy.io import savemat

    savemat('HERA_CHARL_aof_inferred.mat',
            {
                'data': data[..., 0:1],
                'masks': masks[..., 0:1],
                'pred': masks_aof[..., 0:1]
            }
            )


def main3():
    ns = Namespace(data_name='LOFAR',
                   data_path='/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/',
                   rfi_threshold=None,
                   seed='maahh',
                   output_path='./outputs'
                   )
    kwargs = to_dict(ns)

    # '/home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/'
    # '/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/'

    train_data, train_masks, test_data, test_masks = load_raw_data(**kwargs)
    indexes = np.argsort(test_masks.mean(axis=(1, 2, 3)))
    test_data = test_data[indexes]
    test_masks = test_masks[indexes]

    print(test_data.shape[0])
    im_dict = {f'rfi_ratio': test_masks.mean(axis=(1, 2, 3))}
    df = pd.DataFrame.from_dict(im_dict)
    df.to_csv('./pickle_images/rfi_ratio.csv')

    save_image_masks_batches(dir_path='./pickle_images',
                             data=np.log(test_data),
                             masks=test_masks,
                             batch_size=20)


def main2():
    hf = File('/home/ee487519/PycharmProjects/lofar_test_LTA/LOFAR_RFI_test_data_with_aoflags.h5', 'r')
    aoflags = np.array([aof > 31 for aof in hf['aoflags']])
    hand_labels = np.array([h > 0 for h in hf['hand_labels']])
    test_data = np.log(hf['test_data'])

    indexes = np.argsort(hand_labels.mean(axis=(1, 2, 3)))
    test_data = test_data[indexes]
    hand_labels = hand_labels[indexes]
    aoflags = aoflags[indexes]

    # hf['test_data'][index, ..., 0], hf['hand_labels'][index, ..., 0], hf['aoflags'][index, ..., 0] > 31

    f1_list = [f1_score(hand.flatten(), aof.flatten()) for hand, aof in zip(hand_labels, aoflags)]
    print(np.argmax(f1_list), np.max(f1_list))
    print(np.argmin(f1_list), np.min(f1_list))

    im_dict = {f'f1': f1_list,
               'rfi_ratio': hand_labels.mean(axis=(1, 2, 3))}
    df = pd.DataFrame.from_dict(im_dict)
    df.to_csv('./image_f1_ratio.csv')

    save_image_masks_masksinferred_batches('./hd5_images', test_data, hand_labels, aoflags)


def load_aof(dataset='LOFAR'):
    from scipy.io import loadmat
    if dataset == 'LOFAR':
        hf = File('/home/ee487519/PycharmProjects/lofar_test_LTA/LOFAR_RFI_test_data_with_aoflags.h5', 'r')
        aoflags = np.array([aof > 31 for aof in hf['aoflags']])
        hand_labels = np.array([h > 0 for h in hf['hand_labels']])
        test_data = np.array(hf['test_data'])
    if dataset == 'HERA_CHARL':
        hera_aof_mat = loadmat('./HERA_CHARL_aof_inferred.mat')
        test_data = hera_aof_mat['data']
        hand_labels = hera_aof_mat['masks'] > 0
        aoflags = hera_aof_mat['pred'] > 0
        # /home/ee487519/DatasetsAndConfig/Generated/HERA_Charl/HERA_AOF_20-07-2023_all.pkl


    indexes = np.argsort(hand_labels.mean(axis=(1, 2, 3)))
    test_data = test_data[indexes]
    hand_labels = hand_labels[indexes]
    aoflags = aoflags[indexes]

    return test_data[..., 0], hand_labels[..., 0], aoflags[..., 0]


def load_R5(dataset='LOFAR'):
    from scipy.io import loadmat
    if dataset == 'LOFAR':
        R5 = loadmat(
            '/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/RNET5/rfi/proud-malamute-of-abstract-intensity/inferred.mat')
    if dataset == 'HERA_CHARL':
        R5 = loadmat(
            '/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/RNET5/rfi/burrowing-tricky-swan-of-wealth/inferred.mat')

    # indexes = np.argsort(hand_labels.mean(axis=(1,2,3)))
    # test_data = test_data[indexes]
    # hand_labels = hand_labels[indexes]
    # aoflags = aoflags[indexes]

    return R5['data'][..., 0], R5['masks'][..., 0] > 0.0, R5['pred'][..., 0]


def score_scatter(dataset='LOFAR', xaxis='rfi_overlap_ratio', yaxis = 'recall'):
    # We assume that all data points for an image has the same x coordinate
    # i.e. the xaxis is not dependant on aoflagger or RNET5, but rather the
    # that stats of the data itself
    if dataset == 'LOFAR':
        df = pd.read_csv('./LOFAR_R5_aof_image_scores_per_threshold.csv')
    else:
        df = pd.read_csv('./HERA_CHARL_R5_aof_image_scores_per_threshold.csv')
    from utils import save_scatter

    # xaxis = 'rfi_std_over_rfi_mean'
    # yaxis = 'fpr'
    # rfi_std_over_rfi_mean
    # rfi_overlap_ratio


    list_of_list_of_x = []
    list_of_list_of_y = []

    thresholds = (0.5, 0.01, 0.00001, 'aof')
    for thr in thresholds:
        list_of_list_of_x.append(np.array(df[xaxis]))
        list_of_list_of_y.append(np.array(df[f'{yaxis}_{thr}']))

    list_of_list_of_x_lines = []
    list_of_list_of_y_lines = []
    aof_index = []
    for i in range(len(df)):
        row = df.iloc[i]
        x = [row[xaxis]] * len(thresholds)
        y = [row[f'{yaxis}_{thr}'] for thr in thresholds]
        # for thr in thresholds:
        list_of_list_of_x_lines.append(x)
        list_of_list_of_y_lines.append(y)
        indexes = list(np.argsort(y))
        aof_index.append(indexes.index(3))
    print(f'The average {yaxis} value for {thresholds} are {np.array(list_of_list_of_y_lines).mean(axis=(0,))}')

    # LOFAR
    # The average recall value for (0.5, 0.01, 1e-05, 'aof') are [0.4964493  0.52751475 0.59143885 0.51886528]
    # The average precision value for (0.5, 0.01, 1e-05, 'aof') are [0.76456138 0.70309519 0.45407936 0.49386522]
    # The average fpr value for (0.5, 0.01, 1e-05, 'aof') are [0.0014422  0.00192719 0.0044241  0.00354363]
    # print(f'When thresholds: {thresholds} are aranged in ascending order by the value of {yaxis}, then aof has index {np.mean(aof_index)} on average. ')
    # When thresholds: (0.5, 0.01, 1e-05, 'aof') are aranged in ascending order by the value of recall, then aof has index 1.238532110091743 on average.
    # When thresholds: (0.5, 0.01, 1e-05, 'aof') are aranged in ascending order by the value of fpr, then aof has index 2.2110091743119265 on average.

    save_scatter(list_of_list_of_x,
                 list_of_list_of_y,
                 list_of_list_of_x_lines=list_of_list_of_x_lines,
                 list_of_list_of_y_lines=list_of_list_of_y_lines,
                 labels_scatter=thresholds,
                 layout_rect=(0.15, 0.1, 1.0, 1.0),
                 size=80,
                 figsize=(20, 15),
                 linewidth=1,
                 # RFI ratio
                 # logx=True,
                 # xlim_bottom=4e-4,
                 # RFI overlap ratio
                 # xlim_bottom=0.55,
                 # rfi std over rfi mean

                 logy=False,
                 dir_path=f'./{dataset}_R5_aof_scatter',
                 file_name=f'{xaxis}_{yaxis}_scatter',
                 xlabel=xaxis.replace('_', ' ').replace('rfi', 'RFI'),
                 ylabel=yaxis,  # 'Recall',
                 title=None,
                 axis_fontsize=55,
                 xtick_size=55,
                 ytick_size=55,
                 legend_fontsize=55,

                 show_legend=True,
                 legendspacing=0.0,
                 legend_borderpad=0.0,
                 legend_title='Threshold',
                 #legend_bbox=None,
                 # legend_loc='upper left',
                 )


def R5_aof_scores_loop(dataset='LOFAR'):
    data, masks, aoflags = load_aof(dataset=dataset)

    # ----------------------
    # Stats dict
    # im_dict = {'rfi_ratio': masks.mean(axis=(1, 2))}

    temp_dict = data_masks_stats(np.array([0, 1]), np.array([True, False]))
    im_dict = {}
    for key in temp_dict.keys():
        im_dict[key] = []

    for d, m in zip(data, masks):
        image_dict = data_masks_stats(d, m)
        for key in image_dict.keys():
            im_dict[key].append(image_dict[key])
       #  self.df = pd.DataFrame.from_dict(self.stats_dict)

    # ----------------------
    # AOF:
    recall_precision_f1_fpr_list = np.array(
        [aof_recall_precision_f1_fpr(mask, aof) for mask, aof in zip(masks, aoflags)])

    aof_dict = {
        f'recall_aof': recall_precision_f1_fpr_list[..., 0],
        f'precision_aof': recall_precision_f1_fpr_list[..., 1],
        f'f1_aof': recall_precision_f1_fpr_list[..., 2],
        f'fpr_aof': recall_precision_f1_fpr_list[..., 3],
    }
    im_dict = {**im_dict, **aof_dict}

    # ----------------------
    # R5
    data, masks, R5_predicted = load_R5(dataset=dataset)
    for thr in (0.5, 0.01, 0.00001):
        recall_precision_f1_fpr_list = np.array([aof_recall_precision_f1_fpr(mask, r5 > thr) for mask, r5 in zip(masks, R5_predicted)])

        thr_dict = {
            f'recall_{thr}': recall_precision_f1_fpr_list[..., 0],
            f'precision_{thr}': recall_precision_f1_fpr_list[..., 1],
            f'f1_{thr}': recall_precision_f1_fpr_list[..., 2],
            f'fpr_{thr}': recall_precision_f1_fpr_list[..., 3],
        }
        im_dict = {**im_dict, **thr_dict}

    # ------------------
    df = pd.DataFrame.from_dict(im_dict)
    df.to_csv(f'./{dataset}_R5_aof_image_scores_per_threshold.csv')

def gen_LOFAR_R5_aof_conf_images():
    # /home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/RNET5/rfi/proud-malamute-of-abstract-intensity/inferred.mat
    #                 'data': data_recon,
    #                 'masks': masks_recon,
    #                 'pred': inferred_masks_recon
    from utils import save_confusion_image, save_waterfall

    indexes = [6, 16, 71, 75]
    indexes = [107]

    # R5 = loadmat('/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/RNET5/rfi/proud-malamute-of-abstract-intensity/inferred.mat')
    data, masks, R5_predicted = load_R5()
    data, masks, aoflags = load_aof()

    R5_predicted = R5_predicted > 0.5
    masks = masks > 0.0
    for d, m, aof, pred, ind in zip(data[indexes], masks[indexes], aoflags[indexes], R5_predicted[indexes], indexes):
        fontsize = 65
        figsize = (20, 19)

        # increase: more left space
        # increase: more bottom space
        # decrease: more right space
        # decreate: more top space
        save_waterfall(d,
                       layout_rect=(0.15, -0.12, 0.96, 1.2),  # fontsize 65 # figsize = (20, 19)
                       # xlabel='Frequency',
                       # ylabel='Time',
                       xlabel='Frequency Bins',
                       ylabel='Time Bins',
                       show=False,
                       log=True,
                       n_xticks=5,
                       n_yticks=5,
                       show_ticks=True,
                       #show_ticks=False,
                       # show_ytick_labels=False,
                       # show_xtick_labels=False,
                       legendspacing=0,
                       legend_borderpad=0,
                       legend_bbox=(1.3, 0.5),
                       dir_path='LOFAR_conf_images',
                       file_name=f'{ind}_data',
                       axis_fontsize=fontsize,
                       xtick_size=fontsize,
                       ytick_size=fontsize,
                       show_legend=False,
                       legend_fontsize=fontsize,
                       figsize=figsize,
                       # x_range=(100,105),
                       # y_range=(-10,10)
                       )
        legend_bbox_dict = {
            6: (0.9, 0.5),
            16: (0.45, 0.3),
            71: (0.65, 0.3),
            75: (0.3, 0.3),
        }

        save_confusion_image(d, m, m,
                             layout_rect=(0.15, -0.12, 0.96, 1.2),  # fontsize 65 # figsize = (20, 19)
                             # xlabel='Frequency Bins',
                             xlabel='Frequency',
                             ylabel='Time',
                             # ylabel='',
                             show=False,
                             log=True,
                             n_xticks=5,
                             n_yticks=5,
                             #show_ticks=True,
                             show_ticks=False,
                             # show_ytick_labels=False,
                             # show_ytick_labels=False,
                             # show_xtick_labels=False,
                             black_TN=True,
                             legend_fontsize=fontsize,
                             legendspacing=0,
                             legend_borderpad=0,
                             show_legend=False,
                             legend_bbox=None,# legend_bbox_dict[ind],
                             dir_path='LOFAR_conf_images',
                             file_name=f'{ind}_aof',
                             axis_fontsize=fontsize,
                             xtick_size=fontsize,
                             ytick_size=fontsize,

                             figsize=figsize,
                             # x_range=(100,105),
                             # y_range=(-10,10)
                             )
        # exit()
        legend_bbox_dict = {
            6: (0.9, 0.5),
            16: (0.45, 0.3),
            71: (0.65, 0.3),
            75: (0.3, 0.3),
        }
        save_confusion_image(d, m, pred,
                             layout_rect=(0.15, -0.12, 0.96, 1.2),  # fontsize 65 # figsize = (20, 19)
                             xlabel='Frequency Bins',
                             # ylabel='Time Bins',
                             ylabel='',
                             show=False,
                             log=True,
                             n_xticks=5,
                             n_yticks=5,
                             show_ticks=True,
                             show_ytick_labels=True,
                             black_TN=True,
                             show_legend=True,
                             legend_fontsize=fontsize,
                             legendspacing=0,
                             legend_borderpad=0,
                             legend_bbox=legend_bbox_dict[ind],  # (1.1, 0.5),
                             dir_path='LOFAR_conf_images',
                             file_name=f'{ind}_R5',
                             axis_fontsize=fontsize,
                             xtick_size=fontsize,
                             ytick_size=fontsize,

                             figsize=figsize,
                             # x_range=(100,105),
                             # y_range=(-10,10)
                             )


def gen_HERA_CHARL_R5_aof_conf_images():
    # /home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/RNET5/rfi/proud-malamute-of-abstract-intensity/inferred.mat
    #                 'data': data_recon,
    #                 'masks': masks_recon,
    #                 'pred': inferred_masks_recon
    from utils import save_confusion_image, save_waterfall

    # indexes = [0, 15, 50 ,201, ] # aof worst, aof best, r5 worst, r5 best
    indexes = [65, 266] # aof worst, aof best, r5 worst, r5 best

    # R5 = loadmat('/home/ee487519/PycharmProjects/RFI-NLN-HPC/downloads/RNET5/rfi/proud-malamute-of-abstract-intensity/inferred.mat')
    data, masks, R5_predicted = load_R5(dataset='HERA_CHARL')
    data, masks, aoflags = load_aof(dataset='HERA_CHARL')

    R5_predicted = R5_predicted > 0.5
    # masks = masks > 0.0
    for d, m, aof, pred, ind in zip(data[indexes], masks[indexes], aoflags[indexes], R5_predicted[indexes], indexes):
        fontsize = 65
        figsize = (20, 19)

        # increase: more left space
        # increase: more bottom space
        # decrease: more right space
        # decreate: more top space
        save_waterfall(d,
                       layout_rect=(0.15, -0.12, 0.96, 1.2),  # fontsize 65 # figsize = (20, 19)
                       xlabel='Frequency Bins',
                       ylabel='Time Bins',
                       show=False,
                       log=True,
                       n_xticks=5,
                       n_yticks=5,
                       show_ticks=True,
                       legendspacing=0,
                       legend_borderpad=0,
                       legend_bbox=(1.3, 0.5),
                       dir_path='HERA_CHARL_conf_images',
                       file_name=f'{ind}_data',
                       axis_fontsize=fontsize,
                       xtick_size=fontsize,
                       ytick_size=fontsize,
                       show_legend=False,
                       legend_fontsize=fontsize,
                       figsize=figsize,
                       # x_range=(100,105),
                       # y_range=(-10,10)
                       )

        save_confusion_image(d, m, aof,
                             layout_rect=(0.15, -0.12, 0.96, 1.2),  # fontsize 65 # figsize = (20, 19)
                             xlabel='Frequency Bins',
                             # ylabel='Time Bins',
                             ylabel='',
                             show=False,
                             log=True,
                             n_xticks=5,
                             n_yticks=5,
                             show_ticks=True,
                             show_ytick_labels=True,
                             black_TN=True,
                             show_legend=False,
                             legend_fontsize=fontsize,
                             legendspacing=0,
                             legend_borderpad=0,
                             legend_bbox=(1.3, 0.5),
                             dir_path='HERA_CHARL_conf_images',
                             file_name=f'{ind}_aof',
                             axis_fontsize=fontsize,
                             xtick_size=fontsize,
                             ytick_size=fontsize,

                             figsize=figsize,
                             # x_range=(100,105),
                             # y_range=(-10,10)
                             )
        legend_bbox_dict = {
            6: (0.9, 0.5),
            16: (0.45, 0.3),
            71: (0.65, 0.3),
            75: (0.3, 0.3),
        }
        save_confusion_image(d, m, pred,
                             layout_rect=(0.15, -0.12, 0.96, 1.2),  # fontsize 65 # figsize = (20, 19)
                             xlabel='Frequency Bins',
                             # ylabel='Time Bins',
                             ylabel='',
                             show=False,
                             log=True,
                             n_xticks=5,
                             n_yticks=5,
                             show_ticks=True,
                             show_ytick_labels=True,
                             black_TN=True,
                             show_legend=True,
                             legend_fontsize=fontsize,
                             legendspacing=0,
                             legend_borderpad=0,
                             # legend_bbox=legend_bbox_dict[ind],  # (1.1, 0.5),
                             dir_path='HERA_CHARL_conf_images',
                             file_name=f'{ind}_R5',
                             axis_fontsize=fontsize,
                             xtick_size=fontsize,
                             ytick_size=fontsize,

                             figsize=figsize,
                             # x_range=(100,105),
                             # y_range=(-10,10)
                             )


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
    rfi_std_over_rfi_mean = rfi_std/rfi_mean if rfi_ratio > 0.0 else 0.0

    rfi_min_data_percentile = (data < rfi_min).mean() if rfi_ratio > 0.0 else 0.0
    rfi_min_stds_from_mean = (rfi_min - mean) / std if rfi_ratio > 0.0 else 0.0
    rfi_min_means_from_mean = (rfi_min - mean) / mean if rfi_ratio > 0.0 else 0.0
    rfi_min_div_mean = rfi_min/mean if rfi_ratio > 0.0 else 0.0
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
    nonrfi_max_div_mean = nonrfi_max/mean if rfi_ratio < 1.0 else 0.0
    nonrfi_max_over_nonrfi_min = nonrfi_max / nonrfi_min if nonrfi_min > 0.0 else 0.0
    nonrfi_std_over_nonrfi_mean = nonrfi_std/nonrfi_mean if rfi_ratio < 1.0 else 0.0


    rfi_mean_over_nonrfi_mean = rfi_mean/nonrfi_mean if 1.0 > rfi_ratio > 0.0 else 0.0

    nonrfi_perc90 = np.percentile(nonrfi, 90) if rfi_ratio < 1.0 else 0.0
    nonrfi_perc90_data_percentile = (data < nonrfi_perc90).mean() if rfi_ratio < 1.0 else 0.0
    nonrfi_perc90_stds_from_mean = (nonrfi_perc90 - mean) / std if rfi_ratio < 1.0 else 0.0
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
        rfi_std_over_rfi_mean=rfi_std_over_rfi_mean,
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
        nonrfi_std_over_nonrfi_mean=nonrfi_std_over_nonrfi_mean,
        nonrfi_max_data_percentile=nonrfi_max_data_percentile,
        nonrfi_max_stds_from_mean=nonrfi_max_stds_from_mean,
        nonrfi_max_means_from_mean=nonrfi_max_means_from_mean,
        nonrfi_max_div_mean=nonrfi_max_div_mean,
        nonrfi_max_over_nonrfi_min=nonrfi_max_over_nonrfi_min,

        rfi_mean_over_nonrfi_mean=rfi_mean_over_nonrfi_mean,

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
    main()

    # R5_aof_scores_loop(dataset='HERA_CHARL')
    # R5_aof_scores_loop(dataset='LOFAR')

    #gen_HERA_CHARL_R5_aof_conf_images()
    gen_LOFAR_R5_aof_conf_images()

    score_scatter(yaxis='recall', xaxis='rfi_overlap_ratio', dataset='HERA_CHARL')
    score_scatter(yaxis='fpr', xaxis='rfi_overlap_ratio', dataset='HERA_CHARL')
    score_scatter(yaxis='precision', xaxis='rfi_overlap_ratio', dataset='HERA_CHARL')

    # score_scatter(xaxis='recall', yaxis='rfi_std_over_rfi_mean', dataset='LOFAR')


    # df = pd.read_csv('./HERA_CHARL_R5_aof_image_scores_per_threshold.csv')
    # df_aof = df[['f1_aof', 'f1_0.5']].sort_values('f1_aof')
    # #            aof        R5
    # # worst: 0 = 0.03682    0.85566
    # # best: 15 = 0.99949    0.92308
    # df_r5 = df[['f1_aof', 'f1_0.5']].sort_values('f1_0.5')
    # #            aof        R5
    # # worst: 50 = 0.13694    0.23380
    # # best: 201 = 0.92258    0.98675
    # a = 5

