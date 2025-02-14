import tensorflow as tf
import numpy as np
from time import time
import pickle
from sklearn import metrics, neighbors
#import faiss
from inference import infer, get_error
from utils.data import reconstruct, reconstruct_latent_patches, scale, get_dists, nln
import itertools
import warnings

warnings.filterwarnings('ignore')



def get_nln_errors(model,
                   model_type,
                   z_query,
                   z,
                   test_images,
                   x_hat_train,
                   neighbours_idx,
                   neighbour_mask,
                   args):
    """
        Calculates the NLN error for either frNN or KNN

        Parameters
        ----------
        model (tuple): the model used
        model_type (str): the type of model (AE,VAE,...)
        z_query (np.array): test set latent space 
        z (np.array): training set latent space
        test_images (np.array): test data
        x_hat_train (np.array): reconstruction of training data
        neighbour_idx (np.array): indexes of nearest neighbours 
        neighbours_mask (np.array): boolean mask of nearest neighbours array
        args (Namespace)
        
        Returns
        -------
        error (np.array) : nln error per patch (if using) 

    """

    x_hat = infer(model[0], test_images, args, 'AE')
    test_images_stacked = np.stack([test_images] * neighbours_idx.shape[-1], axis=1)
    neighbours = x_hat_train[neighbours_idx]

    if ((model_type == 'AE') or
            (model_type == 'AE_SSIM') or
            (model_type == 'AAE') or
            (model_type == 'DAE') or
            (model_type == 'VAE')):

        error_nln = get_error(model_type, test_images_stacked, neighbours, mean=False)
        error = np.mean(error_nln, axis=1)  # nanmean for frNN

        error_recon = get_error(model_type, test_images, x_hat, mean=False)

        error[neighbour_mask] = error_recon[neighbour_mask]

    elif model_type == 'DAE_disc':
        disc_x_hat_train = infer(model[1], x_hat_train, args, 'disc')
        disc_x = infer(model[1], test_images, args, 'disc')
        disc_x_hat = infer(model[1], x_hat, args, 'disc')

        disc_x_stacked = np.stack([disc_x] * neighbours_idx.shape[-1], axis=1)
        disc_neighbours = disc_x_hat_train[neighbours_idx]

        error_nln = get_error(model_type,
                              test_images_stacked,
                              neighbours,
                              d_x=disc_x_stacked,
                              d_x_hat=disc_neighbours,
                              mean=False)
        error = np.mean(error_nln, axis=1)  # nanmean for frNN

        error_recon = get_error(model_type,
                                test_images,
                                x_hat,
                                d_x=disc_x,
                                d_x_hat=disc_x_hat,
                                mean=False)

        error[neighbour_mask] = error_recon[neighbour_mask]


    elif model_type == 'GANomaly':
        x_hat = infer(model[0], test_images, args, 'AE')
        z_hat = infer(model[2], x_hat, args, 'encoder')

        z_neighbours = z[neighbours_idx]
        z_hat_stacked = np.stack([z_hat] * neighbours_idx.shape[-1], axis=1)

        error_nln = get_error(model_type,
                              test_images_stacked,
                              neighbours,
                              z=z_neighbours,
                              z_hat=z_hat_stacked,
                              mean=False)

        error = np.mean(error_nln, axis=1)  # nanmean for frNN

        error_recon = get_error(model_type,
                                test_images,
                                x_hat,
                                z=z_query,
                                z_hat=z_hat,
                                mean=False)

        error[neighbour_mask] = error_recon[neighbour_mask]

    return error


# HAS NO USAGES
def get_nln_metrics(model,
                    train_images,
                    test_images,
                    test_labels,
                    model_type,
                    args):
    """
        Calculates the NLN metrics for either frNN or KNN

        Parameters
        ----------
        model (tuple): the model used
        train_images (np.array): images from training set 
        test_images (np.array): testing images
        test_labels (np.array): labels of testing images
        model_type (str): the type of model (AE,VAE,...)
        args (Namespace): the argumenets from cmd_args
        
        Returns
        -------
        max_auc (float32): maximum auroc for the given KNN
        max_neighbours (float32): optimal number of neighbours for AUROC
        max_radius (float32): optimal radius size for frNN
        max_dists_auc (float32): optimal distance based auroc  
        max_sum_auc (float32): optimal auroc from adding distance and nln 
        max_mul_auc (float32): optimal auroc from multiplying dists and nln
        x_hat (np.array): Reconstructions of test data
        x_hat_train (np.array): Reconstructions of training data 
        neighbours_idx (np.array): Indexes of neigbours in z_train
        neighbours_dist (np.array): Latent distances for each neighbour

    """

    z_query = infer(model[0].encoder, test_images, args, 'encoder')
    z = infer(model[0].encoder, train_images, args, 'encoder')
    x_hat_train = infer(model[0], train_images, args, 'AE')
    x_hat = infer(model[0], test_images, args, 'AE')

    d, max_auc, max_neighbours, max_radius, index_counter = {}, 0, 0, 0, 0
    dists_auc, sum_auc, mul_auc = [], [], []

    for n_bour in args.neighbours:
        if args.algorithm == 'knn':
            t = time()
            neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z,
                                                                               z_query,
                                                                               x_hat_train,
                                                                               args.algorithm,
                                                                               n_bour,
                                                                               radius=None)
            error = get_nln_errors(model,
                                   model_type,
                                   z_query,
                                   z,
                                   test_images,
                                   x_hat_train,
                                   neighbours_idx,
                                   neighbour_mask,
                                   args)
            if args.patches:
                if error.ndim == 4:
                    # error, test_labels_ = reconstruct(error, args, test_labels)
                    error, test_labels_ = reconstruct(error, args.raw_input_shape, args.patch_x, args.patch_y,
                                                      args.anomaly_class, test_labels)

                else:
                    # error, test_labels_ = reconstruct_latent_patches(error, args, test_labels)
                    error, test_labels_ = reconstruct_latent_patches(error, args.raw_input_shape, args.patch_x,
                                                                     args.patch_y, args.anomaly_class, test_labels)

            recon_error = get_error('AE', test_images, x_hat, mean=False)

            if args.patches:
                # recon_error, test_labels_ = reconstruct(recon_error, args, test_labels)
                error, test_labels_ = reconstruct(recon_error, args.raw_input_shape, args.patch_x, args.patch_y,
                                                  args.anomaly_class, test_labels)
            else:
                recon_error, test_labels_ = recon_error, test_labels

            recon_error = scale(np.nanmean(recon_error, axis=tuple(range(1, recon_error.ndim))), scale_per_image=False)

            error = scale(np.nanmean(error, axis=tuple(range(1, error.ndim))), scale_per_image=False)
            dists = scale(get_dists(neighbours_dist, args), scale_per_image=False)

            if args.anomaly_type == 'MISO':
                add = metrics.roc_auc_score(test_labels_ == args.anomaly_class, error + dists + recon_error)
                mul = metrics.roc_auc_score(test_labels_ == args.anomaly_class, 0.3 * error + 0.7 * dists)
                dists = metrics.roc_auc_score(test_labels_ == args.anomaly_class, dists)
            else:
                add = metrics.roc_auc_score(test_labels_ != args.anomaly_class, error + dists + recon_error)
                mul = metrics.roc_auc_score(test_labels_ != args.anomaly_class, 0.3 * error + 0.7 * dists)
                dists = metrics.roc_auc_score(test_labels_ != args.anomaly_class, dists)

            dists_auc.append(dists)
            sum_auc.append(add)
            mul_auc.append(mul)

            temp_args = [error, test_labels_, args.anomaly_class, args.neighbours,
                         [float('nan')], n_bour, float('nan'), max_auc, max_neighbours,
                         max_radius, index_counter, d, t, dists_auc[-1], sum_auc[-1], mul_auc[-1], args.anomaly_type]

            (max_auc, max_neighbours,
             max_radius, index_counter, d) = get_max_score(temp_args)

    with open('outputs/{}/{}/{}/latent_scores.pkl'.format(model_type, args.anomaly_class, args.model_name), 'wb') as f:
        pickle.dump(d, f)

    return max_auc, max_neighbours, max_radius, np.max(dists_auc), np.max(sum_auc), np.max(
        mul_auc), x_hat, x_hat_train, neighbours_idx, neighbours_dist


def get_max_score(args):
    """
        Find the maximum AUROC score for the model 

        Parameters
        ----------
        args (dictionary): arguments  shown below

        Returns
        -------
        max_auc (float32): maximum detection auroc 
        max_neighbours (float32): maximum KNN
        max_radius (float32): maximum radius for frNN
        index_counter (float32): index for maximum
        d (float32): dictionary of optimal vals 

    """
    (error, test_labels, anomaly, n_bours,
     radius, n_bour, rad, max_auc,
     max_neighbours, max_radius,
     index_counter, d, t, dist_auc, sum_auc, mul_auc, anomaly_type) = args

    if anomaly_type == 'MISO':
        fpr, tpr, thr = metrics.roc_curve(test_labels == anomaly, error)

        a_u_c = metrics.roc_auc_score(test_labels == anomaly, error)

    else:
        fpr, tpr, thr = metrics.roc_curve(test_labels != anomaly, error)

        a_u_c = metrics.roc_auc_score(test_labels != anomaly, error)

    if a_u_c > max_auc: max_auc = a_u_c; max_neighbours = n_bour;max_radius = rad;

    d[index_counter] = [n_bour, rad, a_u_c, dist_auc, sum_auc, mul_auc]
    index_counter += 1
    print("{}/{} auc-roc = {}, dist-auc = {}, sum-auc ={}, mul-acu ={} time elapsed = {}s".format(index_counter,
                                                                                                  len(n_bours) * len(
                                                                                                      radius),
                                                                                                  a_u_c,
                                                                                                  dist_auc,
                                                                                                  sum_auc,
                                                                                                  mul_auc,
                                                                                                  time() - t))

    return max_auc, max_neighbours, max_radius, index_counter, d

