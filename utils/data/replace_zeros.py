import numpy as np


def replace_zeros_with_min(images):
    # images  : ( N_im, widith, height, channels )
    # For every image in images, replaces all zeros with minimum non-zero value
    for i in range(len(images)):
        mi = np.min(images[i][np.nonzero(images[i])])
        images[i][np.where(images[i] == 0)] = mi
    return images
