import numpy as np
import faiss
from .patches import reconstruct

def nln(z, z_query, x_hat_train, algorithm, neighbours, radius=None):
    """
        Calculates the nearest neighbours using either frNN or KNN

        Parameters
        ----------
        z (np.array): training set latent space vector
        z_query (np.array): test set latent space vector
        x_hat_train (np.array): reconstruction of training data
        algorithm (str): KNN or frNN
        neighbours (int): number of neighbours
        radius (double): Optional, the frnn radius

        Returns
        -------
        neighbours_dist (np.array) : latent neigbour distance vector
        neighbours_idx (np.array): index of neigbours in z
        x_hat_train (np.array): reconstruction of training data, adjusted during frNN
        neighbour_mask (np.array): used for frNN to determine if a sample has neighbours

    """
    if algorithm == 'knn':
        index = faiss.IndexFlatL2(z.shape[1])
        index.add(z.astype(np.float32))
        neighbours_dist, neighbours_idx = index.search(z_query.astype(np.float32), neighbours)
        neighbour_mask = np.zeros([len(neighbours_idx)], dtype=bool)

    return neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask


def get_dists(neighbours_dist, args):
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

    dists = np.mean(neighbours_dist, axis=tuple(range(1, neighbours_dist.ndim)))

    if args.patches:
        n_patches = args.raw_input_shape[1] // args.patch_x
        srt, fnnsh = 0, n_patches ** 2
        dists_recon = []
        for i in range(0, len(dists), n_patches ** 2):
            dists_recon.append(np.max(dists[srt:fnnsh]))  ## USING MAX
            srt = fnnsh
            fnnsh += n_patches ** 2

        return np.array(dists_recon)
    else:
        return dists


def get_dists_recon(neighbours_dist, raw_input_shape, patch_x, patch_y):
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


def combine(nln_error, dists, alpha, combine_std_min=None, **kwargs):
    # alpha between 0.0 and 1.0
    # LOFAR optimal alpha = 0.66
    # HERA optimal alpha = 0.10

    # d in dists is a list of dists for every image
    if combine_std_min is None:
        combined_recon = nln_error * np.array([d > np.percentile(d, alpha * 100) for d in dists])
    else:
        nln_error_clipped = np.clip(nln_error,
                                    nln_error.mean() + nln_error.std() * combine_std_min,
                                    1.0)
        combined_recon = nln_error_clipped * np.array([d > np.percentile(d, alpha * 100) for d in dists])
    combined_recon = np.nan_to_num(combined_recon)
    return combined_recon


def get_normal_data(data, masks):
    if data is None or masks is None:
        return None, None
    axis = data.shape[1:]
    normal_data = data[np.invert(np.any(masks, axis=axis))]
    labels = ['normal'] * len(normal_data)
    return normal_data, labels


def get_labels(masks, anomaly_class, **kwargs):
    if masks is None:
        return None
    labels = np.empty(len(masks), dtype='object')
    labels[np.any(masks, axis=(1, 2, 3))] = anomaly_class
    labels[np.invert(np.any(masks, axis=(1, 2, 3)))] = 'normal'
    return labels

