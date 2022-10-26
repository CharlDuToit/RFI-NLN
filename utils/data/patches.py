import tensorflow as tf
import numpy as np
from .defaults import sizes
from model_config import *

def get_patches(x,
                #y,
                p_size,
                s_size,
                rate,
                padding,
                batch_size=1024):
    """
        This function gets reformated image patches with the reshaped labels
        Note: If y is the mask, then we perform logic to get labels from patches

        x (np.array) images
        y (np.array) labels
        p_size (list) patch size
        s_size (list) stride size
        rate (list) subsampling rate after getting patches
    """
    # scaling_factor = (x.shape[1] // p_size[1]) ** 2
    scaling_factor = (x.shape[1] // p_size[1]) * (x.shape[2] // p_size[2])
    output = np.empty([x.shape[0] * scaling_factor, p_size[1], p_size[2], x.shape[-1]], dtype='float32')

    strt, fnnsh = 0, np.minimum(batch_size, x.shape[0])
    output_start, output_fnnsh = 0, np.minimum(batch_size * scaling_factor, output.shape[0])

    for i in range(0, len(x), batch_size):
        x_out = tf.image.extract_patches(images=x[strt:fnnsh, ...],
                                         sizes=p_size,
                                         strides=s_size,
                                         rates=rate,
                                         padding=padding).numpy()

        x_patches = np.reshape(x_out, (x_out.shape[0] * x_out.shape[1] * x_out.shape[2],
                                       p_size[1],
                                       p_size[2],
                                       x.shape[-1]))

        output[output_start:output_fnnsh, ...] = x_patches

        strt = fnnsh
        fnnsh += batch_size
        output_start = output_fnnsh
        output_fnnsh += BATCH_SIZE * scaling_factor

    #if y is not None:
    #    y_labels = np.array([[label] * x_out.shape[1] * x_out.shape[2] for label in y]).flatten()
    #    return x_patches, y_labels
    #else:
        #return output
    return output


def reconstruct_old(patches, args, labels=None):
    """
        Reconstructs the original training/testing images from the patches 
        NOTE: does not work on patches where stride!=patch_size or when data has been central cropped

        Charl: this only works on square data
        
        Parameters
        ----------
        patches (np.array) array of patches generated by get_patches() 
        args (Namespace): the argumenets from cmd_args
        labels (np.array) array of labels of arranged according to patches

        Returns
        -------
        np.array, (optional) np.array

    """
    t = patches.transpose(0, 2, 1, 3)
    n_patches = sizes[str(args.data)] // args.patch_x
    recon = np.empty(
        [patches.shape[0] // n_patches ** 2, args.patch_x * n_patches, args.patch_y * n_patches, patches.shape[-1]])

    start, counter, indx, b = 0, 0, 0, []

    for i in range(n_patches, patches.shape[0] + 1, n_patches):
        b.append(
            np.reshape(np.stack(t[start:i, ...], axis=0), (n_patches * args.patch_x, args.patch_x, patches.shape[-1])))
        start = i
        counter += 1
        if counter == n_patches:
            recon[indx, ...] = np.hstack(b)
            indx += 1
            counter, b = 0, []

    if labels is not None:
        start, end, labels_recon = 0, n_patches ** 2, []

        for i in range(0, labels.shape[0], n_patches ** 2):
            if args.anomaly_class in labels[start:end]:
                labels_recon.append(str(args.anomaly_class))
            else:
                labels_recon.append('non_anomalous')

            start = end
            end += n_patches ** 2
        return recon.transpose(0, 2, 1, 3), np.array(labels_recon)

    else:
        return recon.transpose(0, 2, 1, 3)


def reconstruct(patches, new_shape, patch_x, patch_y, anomaly_class=None, labels=None):
    """
    patch_stride never used
    """
    t = patches.transpose(0, 2, 1, 3)
    n_patches_x = new_shape[0] // patch_x
    n_patches_y = new_shape[1] // patch_y
    recon = np.empty(
        [patches.shape[0] // (n_patches_x * n_patches_y), patch_y * n_patches_y, patch_x * n_patches_x,
         patches.shape[-1]])
    start, counter, indx, b = 0, 0, 0, []

    #  original image = 1,128,512,1
    #  input_shape = 128,512,1
    #  patch_x = 32
    #  n_patches_x = 4
    #    n_patches_y = 64
    #    patch_y = 8
    #  patches.shape = (256, 32, 8, 1)
    #  t.shape = (256, 8, 32, 1)
    # i = 64, 128... 256, . i.e. 4 iterations
    for i in range(n_patches_y, patches.shape[0] + 1, n_patches_y):
        b.append(
            np.reshape(t[start:i, ...], (n_patches_y * patch_y, patch_x, patches.shape[-1])))

        start = i
        counter += 1
        if counter == n_patches_x:
            recon[indx, ...] = np.hstack(b)  # stacks along axis=1
            # recon[0] = (128, 64*8, 1) = (128, 512, 1)
            indx += 1
            counter, b = 0, []

    if labels is not None and anomaly_class is not None:
        start, end, labels_recon = 0, n_patches_x * n_patches_y, []

        for i in range(0, labels.shape[0], n_patches_x * n_patches_y):
            if anomaly_class in labels[start:end]:
                labels_recon.append(str(anomaly_class))
            else:
                labels_recon.append('normal')

            start = end
            end += n_patches_x * n_patches_y
        return recon.transpose(0, 2, 1, 3), np.array(labels_recon)
    else:
        return recon.transpose(0, 2, 1, 3)


def reconstruct_latent_patches(patches, new_shape, patch_x, patch_y, anomaly_class=None, labels=None):
#def reconstruct_latent_patches(patches, args, labels=None):
    """
        Reconstruction method for feature consistent autoencoding

        Parameters
        ----------
        patches (np.array): patches correspodning to the latent projection 

        Returns
        -------
        np.array, (optional) np.array
    """

    n_patches_x = new_shape[0] // patch_x
    n_patches_y = new_shape[1] // patch_y
    recon = np.empty([patches.shape[0] // n_patches_x*n_patches_y,  n_patches_x*n_patches_y, patches.shape[-1]])

    start, end, labels_recon = 0, n_patches_x * n_patches_y, []

    for j, i in enumerate(range(0, patches.shape[0], n_patches_x * n_patches_y)):
        recon[j, ...] = patches[start:end, ...]
        start = end
        end += n_patches_x * n_patches_y

    if labels is not None:
        start, end, labels_recon = 0, n_patches_x * n_patches_y, []
        for i in range(0, labels.shape[0], n_patches_x * n_patches_y):
            if anomaly_class in labels[start:end]:
                labels_recon.append(str(anomaly_class))
            else:
                labels_recon.append('normal')

            start = end
            end += n_patches_x * n_patches_y
        return recon, np.array(labels_recon)
    else:
        return recon

def reconstruct_latent_patches_old(patches, args, labels=None):
    """
        Reconstruction method for feature consistent autoencoding

        Parameters
        ----------
        patches (np.array): patches correspodning to the latent projection

        Returns
        -------
        np.array, (optional) np.array
    """

    n_patches = sizes[str(args.data)] // args.patch_x
    recon = np.empty([patches.shape[0] // n_patches ** 2, n_patches ** 2, patches.shape[-1]])

    start, end, labels_recon = 0, n_patches ** 2, []

    for j, i in enumerate(range(0, patches.shape[0], n_patches ** 2)):
        recon[j, ...] = patches[start:end, ...]
        start = end
        end += n_patches ** 2

    if labels is not None:
        start, end, labels_recon = 0, n_patches ** 2, []

        for i in range(0, labels.shape[0], n_patches ** 2):
            if args.anomaly_class in labels[start:end]:
                labels_recon.append(str(args.anomaly_class))
            else:
                labels_recon.append('normal')

            start = end
            end += n_patches ** 2
        return recon, np.array(labels_recon)

    else:
        return recon


# No references to get_patched_dataset
"""
def get_patched_dataset(train_images,
                        train_labels,
                        test_images,
                        test_labels,
                        test_masks=None,
                        p_size=(1, 4, 4, 1),
                        s_size=(1, 4, 4, 1),
                        rate=(1, 1, 1, 1),
                        padding='VALID',
                        central_crop=False):

        This function returns the training and testing set in patch form. 
        Note: If test_masks is specified then labels generated from the masks, 
              otherwise labels are based off the original labels. 

        train_images (np.array) training images 
        train_labels (np.array) training labels 
        test_images  (np.array) test images 
        test_labels  (np.array) test labels 
        p_size (list) patch size
        s_size (list) stride size 
        rate (list) subsampling rate after getting patches 
        padding (str) ...

    if central_crop:
        train_images = tf.image.central_crop(train_images, 0.7).numpy()

    train_patches, train_labels_p = get_patches(train_images,
                                                train_labels,
                                                p_size,
                                                s_size,
                                                rate,
                                                padding)

    test_patches, test_labels_p = get_patches(test_images,
                                              test_labels,
                                              p_size,
                                              s_size,
                                              rate,
                                              padding)
    if test_masks is not None:
        test_masks_patches, _ = get_patches(test_masks,
                                            test_labels,
                                            p_size,
                                            s_size,
                                            rate,
                                            padding)

        return train_patches, train_labels_p, test_patches, test_labels_p, test_masks_patches

    else:  # test_masks is None
        return train_patches, train_labels_p, test_patches, test_labels_p
"""
