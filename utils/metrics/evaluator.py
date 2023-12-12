from .inference import infer_fcn
from utils import reconstruct, rfi_ratio_split
from utils import save_data_masks_inferred, save_image_masks_masksinferred_batches
from utils import rfi_ratio_indexes
from utils import save_lines
from .segmentation_metrics import auroc, auprc, f1, precision, accuracy, recall, \
    recall_prec_f1_fpr_tn_fp_fn_fp, conf_matrix  # , prec_recall_vals, fpr_tpr_vals, conf_matrix
from .dynamic_threshold import range_rounding
import numpy as np
import pandas as pd
from utils import model_dir, rfi_file, unshuffle, save_scatter
import time
import os
from scipy.io import savemat

from sklearn.metrics import (roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score,
                             accuracy_score, confusion_matrix)
import tensorflow as tf


def infer_and_get_f1_and_save(model, data, masks, raw_input_shape, patch_x, patch_y,
                              data_subset='test',
                              batch_size=64,
                              shuffle_seed=None, shuffle_patches=False,
                              **kwargs):
    if data is None or masks is None:
        return {}

    # ------------------------------------ Infer ------------------------------------
    masks_inferred = infer_fcn(model, data, batch_size=batch_size)
    masks_inferred = np.clip(masks_inferred, 0.0, 1.0)

    # ------------------------------------ Unshuffle and Reconstruct ------------------------------------
    if shuffle_patches:
        data, masks, masks_inferred = unshuffle(shuffle_seed, data, masks, masks_inferred)
    # n_p = patches_per_image * images_per_epoch
    # reconstruct_and_save_images(data[:n_p], masks[:n_p], masks_inferred[:n_p], **kwargs)
    data_recon = reconstruct(data, raw_input_shape, patch_x, patch_y, None, None)
    masks_recon = reconstruct(masks, raw_input_shape, patch_x, patch_y, None, None)
    inferred_masks_recon = reconstruct(masks_inferred, raw_input_shape, patch_x, patch_y, None, None)

    indexes = np.argsort(masks_recon.mean(axis=(1,2,3)))
    data_recon = data_recon[indexes]
    masks_recon = masks_recon[indexes]
    inferred_masks_recon = inferred_masks_recon[indexes]

    # ------------------------------------ Save data  ------------------------------------
    savemat(os.path.join(model_dir(**kwargs), 'inferred.mat'),
            {
                'data': data_recon,
                'masks': masks_recon,
                'pred': inferred_masks_recon
            }
            )

    # ------------------------------------ Save images  ------------------------------------
    _dir = os.path.join(model_dir(**kwargs), 'images')
    save_image_masks_masksinferred_batches(_dir, data_recon, masks_recon, inferred_masks_recon)

    # ------------------------------------ Calculate per image F1  ------------------------------------
    # reconstruct_and_save_images(data, masks, masks_inferred, **kwargs)
    images_f1 = [f1(m, mi) for m, mi in zip(masks_recon, inferred_masks_recon)]
    worst_f1_index = np.argmin(images_f1)
    best_f1_index = np.argmax(images_f1)
    worst_f1 = images_f1[worst_f1_index]
    best_f1 = images_f1[best_f1_index]
    print(worst_f1_index, worst_f1, best_f1_index, best_f1)

    # # ------------------------------------ Save image CSV ------------------------------------
    im_dict = {
        f'{data_subset}_f1': images_f1,
        f'{data_subset}_rfi_ratio': masks_recon.mean(axis=(1,2,3)),

    }
    df = pd.DataFrame.from_dict(im_dict)
    df.to_csv(os.path.join(model_dir(**kwargs), 'image_rfiratio.csv'))

    return {  # (
        f'{data_subset}_worst_f1_index': worst_f1_index,
        f'{data_subset}_worst_f1': worst_f1,
        f'{data_subset}_best_f1_index': best_f1_index,
        f'{data_subset}_best_f1': best_f1,
    }


def infer_and_get_metrics(model, data, masks, patches_per_image, data_subset='train', save_images=False, batch_size=64,
                          images_per_epoch=10, shuffle_seed=None, shuffle_patches=False,
                          rfi_split_ratio=0.01, calc_auc=True, **kwargs):
    if data is None or masks is None:
        return {}

    # ------------------------------------ Infer ------------------------------------
    start = time.time()
    masks_inferred = infer_fcn(model, data, batch_size=batch_size)
    masks_inferred = np.clip(masks_inferred, 0.0, 1.0)
    infer_time = time.time() - start
    time_patch = infer_time / len(data)  # per patch, might be ==2 if tuple is based in data
    time_image = time_patch * patches_per_image

    # masks_inferred = tf.nn.max_pool2d(masks_inferred, ksize=2, strides=1, padding='SAME').numpy()

    # ------------------------------------ Save images ------------------------------------
    if save_images:
        n_p = patches_per_image * images_per_epoch
        if not shuffle_patches:
            reconstruct_and_save_images(data[:n_p], masks[:n_p], masks_inferred[:n_p], **kwargs)
        else:
            data, masks, masks_inferred = unshuffle(shuffle_seed, data, masks, masks_inferred)
            reconstruct_and_save_images(data[:n_p], masks[:n_p], masks_inferred[:n_p], **kwargs)

    # ------------------------------------ Calc patch metrics ------------------------------------
    # patch_f1 = [f1(m, mi) for m, mi in zip(masks, masks_inferred)]
    true_rfi_ratio = np.array([np.mean(m) for m in masks]) + 1e-4
    # inferred_rfi_ratio = np.array([np.mean(m) for m in masks_inferred]) + 1e-4
    # max_inferred = np.array([np.max(m) for m in masks_inferred])
    #
    # # ------------------------------------ Save patch CSV ------------------------------------
    # patch_dict = {
    #     f'{data_subset}_true_rfi_ratio': true_rfi_ratio,
    #     f'{data_subset}_inferred_rfi_ratio': inferred_rfi_ratio,
    #     f'{data_subset}_f1': patch_f1,
    #     f'{data_subset}_max_inferred': max_inferred,
    # }
    # df = pd.DataFrame.from_dict(patch_dict)
    # file = rfi_file(data_subset=data_subset, **kwargs)
    # df.to_csv(file)
    #
    # # ------------------------------------ Save patch F1 vs RFI ratio scatter ------------------------------------
    # _dir = model_dir(**kwargs)
    # save_scatter(true_rfi_ratio, patch_f1, size=1, logx=True, dir_path=_dir, xlabel='RFI ratio',
    #              ylabel='F1-score', file_name=f'{data_subset}_rfi_f1')

    # ------------------------------------ Calculate metrics------------------------------------
    # _f1 = f1(masks, masks_inferred)
    _accuracy = accuracy(masks, masks_inferred)
    _recall = recall(masks, masks_inferred)
    _precision = precision(masks, masks_inferred)
    _f1 = 2 * _recall * _precision / (_precision + _recall)
    # TN, FP, FN, TP = conf_matrix(masks, masks_inferred, thr=0.5)

    masks_inferred = masks_inferred.flatten()
    masks = masks.flatten()

    # _accuracy = accuracy_score(masks, masks_inferred > 0.5)
    # _recall, _precision, fpr_half, _f1, TN, FP, FN, TP = recall_prec_f1_fpr_tn_fp_fn_fp(masks, masks_inferred)

    #  clever rounding
    masks_inferred = range_rounding(masks_inferred)

    _dir = model_dir(**kwargs)
    if calc_auc:
        fpr, tpr, thr = roc_curve(masks, masks_inferred)
        save_lines(fpr, tpr, linewidth=1, size=5, scatter=True, xlabel='FPR', ylabel='TPR',
                   file_name=f'{data_subset}_fpr_tpr_curve_rr', dir_path=_dir)
        _auroc = auc(fpr, tpr)

        prec, recall_, threshold = precision_recall_curve(masks, masks_inferred)
        save_lines(recall_, prec, linewidth=1, size=5, scatter=True, xlabel='Recall', ylabel='Precision',
                   file_name=f'{data_subset}_prec_recall_curve_rr', dir_path=_dir)
        _auprc = auc(recall_, prec)
    else:
        fpr = None
        tpr = None
        thr = None
        prec = None
        recall_ = None
        threshold = None
        _auroc = None
        _auprc = None

    # ------------------------------------ Calculate F1 score for low and high rfi ratios -------------------------
    # rfi_split_ratio = 0.01

    # lo_ind = true_rfi_ratio < rfi_split_ratio
    # hi_ind = true_rfi_ratio >= rfi_split_ratio
    #
    # f1_lo = f1_score(masks[lo_ind], masks_inferred[lo_ind] > 0.5)
    # f1_hi = f1_score(masks[hi_ind], masks_inferred[hi_ind] > 0.5)

    print(data_subset, ' time: ', time.time() - start)

    # ------------------------------------ Returns metrics dict ------------------------------------
    return {  # (
        f'{data_subset}_auroc_new': _auroc,
        f'{data_subset}_auprc_new': _auprc,
        f'{data_subset}_f1_new': _f1,
        #f'{data_subset}_f1_low': f1_lo,
        #f'{data_subset}_f1_high': f1_hi,
        f'{data_subset}_accuracy': _accuracy,
        f'{data_subset}_recall_new': _recall,
        f'{data_subset}_precision_new': _precision,
        # f'{data_subset}_fpr_new': fpr_half,

        # f'{data_subset}_TN': TN,
        # f'{data_subset}_FP': FP,
        # f'{data_subset}_FN': FN,
        # f'{data_subset}_TP': TP,

        f'{data_subset}_fpr_vals': list(fpr),
        f'{data_subset}_tpr_vals': list(tpr),
        f'{data_subset}_fpr_tpr_thr_vals': list(thr),
        f'{data_subset}_prec_vals': list(prec),
        f'{data_subset}_recall_vals': list(recall_),
        f'{data_subset}_prec_recall_thr_vals': list(threshold),
        'rfi_split_ratio': rfi_split_ratio,
        'time_image': time_image,
        'time_patch': time_patch
    }


def infer_and_get_metrics_separate(model, data, masks, patches_per_image, data_subset='train', save_images=True,
                                   batch_size=64, images_per_epoch=10, shuffle_seed=None, shuffle_patches=False,
                                   rfi_split_ratio=0.01, **kwargs):
    if data is None or masks is None:
        return {}

    model_low = model[0]
    model_high = model[1]

    masks_inferred = np.empty(masks.shape, dtype='float32')
    lo_ind, hi_ind = rfi_ratio_indexes(masks, rfi_split_ratio)

    # ------------------------------------ Infer Low ------------------------------------
    start = time.time()
    masks_inferred[lo_ind] = infer_fcn(model_low, data[lo_ind], batch_size=batch_size)
    infer_time = time.time() - start
    time_patch = infer_time / len(data)  # per patch, might be ==2 if tuple is based in data
    time_image = time_patch * patches_per_image

    # ------------------------------------ Infer High ------------------------------------
    masks_inferred[hi_ind] = infer_fcn(model_high, data[hi_ind], batch_size=batch_size)

    masks_inferred = np.clip(masks_inferred, 0.0, 1.0)
    # ------------------------------------ Save images ------------------------------------
    if save_images:
        n_p = patches_per_image * images_per_epoch
        if shuffle_patches:
            data, masks, masks_inferred = unshuffle(shuffle_seed, data, masks, masks_inferred)
        reconstruct_and_save_images(data[:n_p], masks[:n_p], masks_inferred[:n_p], **kwargs)

    # ------------------------------------ Calc patch metrics ------------------------------------
    patch_f1 = [f1(m, mi) for m, mi in zip(masks, masks_inferred)]
    true_rfi_ratio = np.array([np.mean(m) for m in masks]) + 1e-4
    inferred_rfi_ratio = np.array([np.mean(m) for m in masks_inferred]) + 1e-4
    max_inferred = np.array([np.max(m) for m in masks_inferred])

    # ------------------------------------ Save patch CSV ------------------------------------
    patch_dict = {
        f'{data_subset}_true_rfi_ratio': true_rfi_ratio,
        f'{data_subset}_inferred_rfi_ratio': inferred_rfi_ratio,
        f'{data_subset}_f1': patch_f1,
        f'{data_subset}_max_inferred': max_inferred,
    }
    df = pd.DataFrame.from_dict(patch_dict)
    file = rfi_file(data_subset=data_subset, **kwargs)
    df.to_csv(file)

    # ------------------------------------ Save patch F1 vs RFI ratio scatter ------------------------------------
    _dir = model_dir(**kwargs)
    try:
        save_scatter(true_rfi_ratio, patch_f1, size=1, logx=True, dir_path=_dir, xlabel='RFI ratio',
                     ylabel='F1-score', file_name=f'{data_subset}_rfi_f1')
    except:
        pass

    # ------------------------------------ Calculate metrics------------------------------------
    _auroc = auroc(masks, masks_inferred)
    _auprc = auprc(masks, masks_inferred)
    _f1 = f1(masks, masks_inferred)
    _accuracy = accuracy(masks, masks_inferred)
    _recall = recall(masks, masks_inferred)
    _precision = precision(masks, masks_inferred)

    # ------------------------------------ Calculate F1 score for low and high rfi ratios -------------------------

    f1_lo = f1(masks[lo_ind], masks_inferred[lo_ind])
    f1_hi = f1(masks[hi_ind], masks_inferred[hi_ind])

    # ------------------------------------ Returns metrics dict ------------------------------------
    return {  # (
        f'{data_subset}_auroc': _auroc,
        f'{data_subset}_auprc': _auprc,
        f'{data_subset}_f1': _f1,
        f'{data_subset}_f1_low': f1_lo,
        f'{data_subset}_f1_high': f1_hi,
        f'{data_subset}_accuracy': _accuracy,
        f'{data_subset}_recall': _recall,
        f'{data_subset}_precision': _precision,
        # 'rfi_split_ratio': rfi_split_ratio,
        'time_image': time_image,
        'time_patch': time_patch
    }


def reconstruct_and_save_images(data, masks, inferred_masks, raw_input_shape, patch_x, patch_y, **kwargs):
    try:
        data_recon = reconstruct(data, raw_input_shape, patch_x, patch_y, None, None)
        masks_recon = reconstruct(masks, raw_input_shape, patch_x, patch_y, None, None)
        inferred_masks_recon = reconstruct(inferred_masks, raw_input_shape, patch_x, patch_y, None, None)
        _dir = model_dir(**kwargs)
        save_data_masks_inferred(_dir, data_recon, masks_recon, inferred_masks_recon)
        save_data_masks_inferred(_dir, data_recon, masks_recon, inferred_masks_recon, thresh=0.5)
    except:
        pass


def infer_and_get_curves(model, data, masks, patches_per_image, save_images=False, batch_size=64,
                         images_per_epoch=10, shuffle_seed=None, shuffle_patches=False,
                         data_subset='test',
                         **kwargs):
    start = time.time()

    verbose = False
    if verbose: print('Evaluating: ', data_subset)

    if data is None or masks is None:
        return {}

    # ------------------------------------ Infer ------------------------------------
    if verbose: print('   Infering...')
    masks_inferred = infer_fcn(model, data, batch_size=batch_size)
    masks_inferred = np.clip(masks_inferred, 0.0, 1.0)

    # ------------------------------------ Save images ------------------------------------
    if save_images:
        n_p = patches_per_image * images_per_epoch
        if not shuffle_patches:
            reconstruct_and_save_images(data[:n_p], masks[:n_p], masks_inferred[:n_p], **kwargs)
        else:
            data, masks, masks_inferred = unshuffle(shuffle_seed, data, masks, masks_inferred)
            reconstruct_and_save_images(data[:n_p], masks[:n_p], masks_inferred[:n_p], **kwargs)

    # ------------------------------------ Calculate metrics------------------------------------
    _dir = model_dir(**kwargs)

    # flatten
    masks_inferred = masks_inferred.flatten()
    masks = masks.flatten()

    _recall, _precision, fpr_half, _f1, TN, FP, FN, TP = 0, 0, 0, 0, 0, 0, 0, 0
    if data_subset != 'train':
        if verbose: print('   Calculating conf matrix')
        _recall, _precision, fpr_half, _f1, TN, FP, FN, TP = recall_prec_f1_fpr_tn_fp_fn_fp(masks, masks_inferred)

    #  clever rounding
    masks_inferred = range_rounding(masks_inferred)

    #
    if verbose: print('   Calculating roc')
    fpr, tpr, thr = roc_curve(masks, masks_inferred)
    save_lines(fpr, tpr, linewidth=1, size=5, scatter=True, xlabel='FPR', ylabel='TPR',
               file_name=f'{data_subset}_fpr_tpr_curve_rr', dir_path=_dir)
    if verbose: print('   Calculating area under roc')
    auroc_ = auc(fpr, tpr)
    # print(len(thr)) # test 6 328 392, val 44577

    prec, recall_, threshold = precision_recall_curve(masks, masks_inferred)
    save_lines(recall_, prec, linewidth=1, size=5, scatter=True, xlabel='Recall', ylabel='Precision',
               file_name=f'{data_subset}_prec_recall_curve_rr', dir_path=_dir)
    auprc_ = auc(recall_, prec)

    print(data_subset, ' time: ', time.time() - start)

    # ------------------------------------ Returns metrics dict ------------------------------------
    return {  # (
        f'{data_subset}_tpr_vals': list(tpr),
        f'{data_subset}_fpr_vals': list(fpr),
        f'{data_subset}_fpr_tpr_thr_vals': list(thr),
        f'{data_subset}_prec_vals': list(prec),
        f'{data_subset}_recall_vals': list(recall_),
        f'{data_subset}_prec_recall_thr_vals': list(threshold),

        f'{data_subset}_auprc_new': auprc_,
        f'{data_subset}_auroc_new': auroc_,

        f'{data_subset}_f1_new': _f1,
        f'{data_subset}_recall_new': _recall,
        f'{data_subset}_precision_new': _precision,
        f'{data_subset}_fpr_new': fpr_half,
        f'{data_subset}_TN': TN,
        f'{data_subset}_FP': FP,
        f'{data_subset}_FN': FN,
        f'{data_subset}_TP': TP,

    }


def evaluate(model,
             train_data,
             train_masks,
             # train_rfi_ratio,
             val_data,
             val_masks,
             # val_rfi_ratio,
             test_data,
             test_masks,
             # test_rfi_ratio,
             rfi_set,
             calc_train_val_auc,
             **kwargs
             ):
    if rfi_set == 'separate':
        return evaluate_separate(model,
                                 train_data,
                                 train_masks,
                                 val_data,
                                 val_masks,
                                 test_data,
                                 test_masks,
                                 **kwargs
                                 )

    start = time.time()
    print('Evaluating model')
    train_metrics = infer_and_get_metrics(model, train_data, train_masks, data_subset='train',
                                          calc_auc=calc_train_val_auc, **kwargs)
    val_metrics = infer_and_get_metrics(model, val_data, val_masks, data_subset='val', calc_auc=calc_train_val_auc,
                                        **kwargs)
    test_metrics = infer_and_get_metrics(model, test_data, test_masks, data_subset='test', calc_auc=True,
                                         save_images=True, **kwargs)

    generalization_metrics = {}
    if test_metrics:
        test_f1 = 'test_f1_new'
        print(f'Test F1: {test_metrics[test_f1]}')
        if calc_train_val_auc:
            generalization_metrics['train_auprc_over_test_auprc_new'] = train_metrics['train_auprc_new'] / test_metrics[
                'test_auprc_new']
    if calc_train_val_auc:
        if val_metrics:
            generalization_metrics['train_auprc_over_val_auprc_new'] = train_metrics['train_auprc_new'] / val_metrics[
                'val_auprc_new']
        if val_metrics and test_metrics:
            generalization_metrics['val_auprc_over_test_auprc_new'] = val_metrics['val_auprc_new'] / test_metrics['test_auprc_new']
    print(f'Evaluation time: {time.time() - start}s')

    return {
        **train_metrics,
        **val_metrics,
        **test_metrics,
        **generalization_metrics,
    }


def evaluate_curves(model,
                    test_data,
                    test_masks,
                    val_data,
                    val_masks,
                    train_data,
                    train_masks,
                    **kwargs
                    ):
    start = time.time()
    print('Evaluating model')

    test_metrics = infer_and_get_curves(model, test_data, test_masks, save_images=False, data_subset='test', **kwargs)
    val_metrics = infer_and_get_curves(model, val_data, val_masks, save_images=False, data_subset='val', **kwargs)
    train_metrics = infer_and_get_curves(model, train_data, train_masks, save_images=False, data_subset='train',
                                         **kwargs)

    generalization_metrics = dict()
    generalization_metrics['train_auprc_over_test_auprc_new'] = train_metrics['train_auprc_new'] / test_metrics[
        'test_auprc_new']
    generalization_metrics['train_auprc_over_val_auprc_new'] = train_metrics['train_auprc_new'] / val_metrics[
        'val_auprc_new']

    print(f'Evaluation time: {time.time() - start}s')
    return {**generalization_metrics, **train_metrics, **test_metrics, **val_metrics}


def evaluate_separate(model,
                      train_data,
                      train_masks,
                      # train_rfi_ratio,
                      val_data,
                      val_masks,
                      # val_rfi_ratio,
                      test_data,
                      test_masks,
                      # test_rfi_ratio,
                      **kwargs
                      ):
    start = time.time()
    print('Evaluating model')
    train_metrics = infer_and_get_metrics_separate(model, train_data, train_masks, data_subset='train', **kwargs)
    val_metrics = infer_and_get_metrics_separate(model, val_data, val_masks, data_subset='val', **kwargs)
    test_metrics = infer_and_get_metrics_separate(model, test_data, test_masks, data_subset='test', **kwargs)

    if test_metrics:
        test_f1 = 'test_f1'
        print(f'Test F1: {test_metrics[test_f1]}')
    print(f'Evaluation time: {time.time() - start}s')

    return {
        **train_metrics,
        **val_metrics,
        **test_metrics,
    }
