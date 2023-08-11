import numpy as np
import matplotlib.pyplot as plt
from utils import get_patches, reconstruct
import os
from scipy.io import loadmat


def modulate(data,
             patch_size,

             stds_over_means_factor=1.0,
             clip_stds_over_means=True,
             stds_over_means_max=2,
             stds_over_means_min=0.5,

             patch_over_means_factor=1.0,
             clip_patch_over_means=True,
             patch_over_means_min=0.8,
             patch_over_means_max=2,
             display=True, ):
    if len(data.shape) == 2:
        data = np.expand_dims(data, (0,-1))

    recon_shape = data.shape

    # Get patches
    p_size = (1, patch_size, patch_size, 1)
    s_size = (1, patch_size, patch_size, 1)
    rate = (1, 1, 1, 1)
    patches = get_patches(data, p_size, s_size, rate, 'VALID')
    n_patches = patches.shape[0]

    # Get std/mean for every patch
    patch_means = np.mean(patches, axis=(1,2,3))
    patch_stds = np.std(patches, axis=(1,2,3))
    patch_stds_over_means = patch_stds / patch_means
    patch_stds_over_means = stds_over_means_factor * patch_stds_over_means / np.mean(patch_stds_over_means)

    # Clip each std/mean
    if clip_stds_over_means:
        patch_stds_over_means = np.clip(patch_stds_over_means, stds_over_means_min, stds_over_means_max)

    # Multiply each patch of ones by its std/mean i.e. low resolution thresholds
    patches_stds_over_means = np.ones(patches.shape)
    for i in range(n_patches):
        patches_stds_over_means[i, ...] = patches_stds_over_means[i, ...] * patch_stds_over_means[i]

    # Divide every sample in patch by its patch mean i.e. high resolution thresholds
    patches_over_means = np.empty(patches.shape)
    for i in range(n_patches):
        patches_over_means[i, ...] = patches[i, ...] / patch_means[i]
    patches_over_means = patch_over_means_factor * patches_over_means

    # Clip high resolution thresholds
    if clip_patch_over_means:
        patches_over_means = np.clip(patches_over_means, patch_over_means_min, patch_over_means_max)

    # Multiply low and high resoluton thresholds
    patches_thresholds = np.multiply(patches_stds_over_means, patches_over_means)
    thresholds_recon = reconstruct(patches_thresholds, recon_shape[1:3], patch_size, patch_size)

    if display:
        recon = reconstruct(patches_stds_over_means, recon_shape[1:3], patch_size, patch_size )
        display_thresh(recon, 'stds_over_means', figsize=(10,10))

        recon = reconstruct(patches_over_means, recon_shape[1:3], patch_size, patch_size )
        display_thresh(recon, 'patch_over_means', figsize=(40,40))

        display_thresh(thresholds_recon, 'modulated_thresholds', figsize=(40,40))

    return thresholds_recon

def display_thresh(thresh_image, file_name='name', figsize=(40,40), thresh=None):
    if len(thresh_image.shape) == 4:
        thresh_image = thresh_image[0, ..., 0]
    fig, ax = plt.subplots(figsize=figsize)
    if thresh is None:
        im = ax.imshow(thresh_image)
    else:
        im = ax.imshow(thresh_image > thresh )
    plt.colorbar(im , ax=ax, location='right', orientation='vertical')
    if file_name[-4:] != '.png':
        file_name += '.png'
    fig.savefig(f'./{file_name}')

def load_data(dir_path, file_name):
    file = os.path.join(dir_path, file_name)
    if file[-4:] != '.mat':
        file += '.mat'
    return loadmat(file)['sbdata']


if __name__ == '__main__':
    dir_path = '/home/ee487519/PycharmProjects/correlator/'
    file_name = 'ant_fft_000_094_t4096_f4096'
    data = load_data(dir_path, file_name)/1e3
    thresholds = modulate(data, 512)

    dir_path = '/home/ee487519/PycharmProjects/RFI-NLN/outputs/infer_test/RNET/rfi/illustrious-poodle-of-stimulating-joviality/inferred/ant_fft_000_094_t4096_f4096/'
    dir_path += 'p-0.012_m-0.0157/'
    dir_path += 'masks_inferred.mat'

    masks_inferred = loadmat(dir_path)['masks_inferred']

    new_masks = np.multiply(masks_inferred, thresholds)

    display_thresh(new_masks, 'new_masks')
    display_thresh(new_masks, 'new_masks_thresh0.5', thresh=0.5)
