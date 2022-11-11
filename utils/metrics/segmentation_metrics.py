import tensorflow as tf
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (roc_curve,
                             auc,
                             f1_score,
                             accuracy_score,
                             average_precision_score,
                             jaccard_score,
                             roc_auc_score,
                             precision_recall_curve)
from utils.data import *


def f1(y_true, y_pred, threshold=0.5):
    return f1_score(y_true.flatten(), y_pred.flatten() > threshold)


def auroc(y_true,y_pred):
    fpr, tpr, thr = roc_curve(y_true.flatten() > 0, y_pred.flatten())
    return auc(fpr, tpr)


def auprc(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true.flatten() > 0, y_pred.flatten())
    return auc(recall, precision)


def top_f1(y_true, y_pred, retun_f1_and_thrsh = False):
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


def get_dists(neighbours_dist, raw_input_shape, patch_x, patch_y):
    """
        Reconstruct distance vector to original dimensions when using patches

        Parameters
        ----------
        neighbours_dist (np.array): Vector of per neighbour distances
        args (Namespace): cmd_args 

        Returns
        -------
        dists (np.array): reconstructed patches if necessary

    """
    patches = raw_input_shape[0] > patch_x or raw_input_shape[1] > patch_y
    dists = np.mean(neighbours_dist, axis=tuple(range(1, neighbours_dist.ndim)))
    if patches:
        #dists = np.array([[d] * args.patch_x ** 2 for i, d in enumerate(dists)]).reshape(len(dists), args.patch_x, args.patch_y)
        dists = np.array([[d] * patch_x * patch_y for i, d in enumerate(dists)]).reshape(len(dists), patch_x, patch_y)

        # dists_recon = reconstruct(np.expand_dims(dists, axis=-1), args)
        dists_recon = reconstruct(np.expand_dims(dists, axis=-1), raw_input_shape, patch_x, patch_y)
        return dists_recon
    else:
        return dists
