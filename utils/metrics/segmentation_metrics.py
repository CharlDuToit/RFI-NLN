import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (roc_curve,
                             auc,
                             f1_score,
                             recall_score,
                             precision_score,
                             accuracy_score,
                             average_precision_score,
                             jaccard_score,
                             roc_auc_score,
                             confusion_matrix,
                             precision_recall_curve)
from utils.data import *


def aof_recall_precision_f1_fpr(y_true, y_pred):
    """
    y_pred is assumed to be binary or exclusively 0s and 1s
    """
    # aconf_mat[0,0] = # TN
    # conf_mat[0, 1] = # FP
    # conf_mat[1, 0] = # FN
    # conf_mat[1,1] =  # TP

    conf_mat = confusion_matrix(y_true.flatten(), y_pred.flatten())
    recall_ = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[1, 0])
    precision_ = conf_mat[1, 1] / (conf_mat[1, 1] + conf_mat[0, 1])
    fpr = conf_mat[0, 1] / (conf_mat[0, 0] + conf_mat[0, 1])
    f1_ = 2 * precision_ * recall_ / (precision_ + recall_)

    return recall_, precision_, f1_, fpr


def recall(y_true, y_pred, threshold=0.5):
    return recall_score(y_true.flatten(), y_pred.flatten() > threshold)


def accuracy(y_true, y_pred, threshold=0.5):
    return accuracy_score(y_true.flatten(), y_pred.flatten() > threshold)


def precision(y_true, y_pred, threshold=0.5):
    return precision_score(y_true.flatten(), y_pred.flatten() > threshold)


def f1(y_true, y_pred, threshold=0.5):
    return f1_score(y_true.flatten(), y_pred.flatten() > threshold)


def auroc(y_true, y_pred):
    fpr, tpr, thr = roc_curve(y_true.flatten() > 0, np.round(y_pred.flatten(), 3))
    return auc(fpr, tpr)


def auprc(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true.flatten() > 0, np.round(y_pred.flatten(), 3))
    return auc(recall, precision)


def fpr_tpr_vals(y_true, y_pred):
    fpr, tpr, thr = roc_curve(y_true.flatten() > 0, np.round(y_pred.flatten(), 3))
    return fpr, tpr, thr
    # return auc(fpr, tpr)


def prec_recall_vals(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true.flatten() > 0, np.round(y_pred.flatten(), 3))
    return precision, recall, thresholds
    # return auc(recall, precision)


def conf_matrix(y_true, y_pred, thr=0.5):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    mat = confusion_matrix(y_true.flatten(), y_pred.flatten() >= thr)

    TN = mat[0, 0]
    FP = mat[0, 1]
    FN = mat[1, 0]
    TP = mat[1, 1]

    return TN, FP, FN, TP


def top_f1(y_true, y_pred, retun_f1_and_thrsh=False):
    precision, recall, thresholds = precision_recall_curve(y_true.flatten() > 0, y_pred.flatten())

    f1_scores = 2 * recall * precision / (recall + precision)
    best_f1 = np.nanmax(f1_scores)
    best_thresh = thresholds[np.nanargmax(f1_scores)]
    if retun_f1_and_thrsh:
        return best_f1, best_thresh
    else:
        return best_f1


def get_metrics(test_masks_recon, test_masks_orig_recon, error_recon):
    fpr, tpr, thr = roc_curve(test_masks_orig_recon.flatten() > 0,
                              error_recon.flatten())
    true_auroc = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(test_masks_orig_recon.flatten() > 0,
                                                           error_recon.flatten())
    true_auprc = auc(recall, precision)

    f1_scores = 2 * recall * precision / (recall + precision)
    true_f1 = np.max(f1_scores)

    return -1, true_auroc, -1, true_auprc, -1, true_f1
