import tensorflow as tf
import numpy as np
#from .defaults import sizes

def get_patches(x,
                #y,
                p_size,
                s_size,
                rate,
                padding,
                batch_size=512):
    """
        This function gets reformated image patches with the reshaped labels
        Note: If y is the mask, then we perform logic to get labels from patches

        x (np.array) images of shape (N, x,y, 1)
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
        output_fnnsh += batch_size * scaling_factor

    #if y is not None:
    #    y_labels = np.array([[label] * x_out.shape[1] * x_out.shape[2] for label in y]).flatten()
    #    return x_patches, y_labels
    #else:
        #return output
    return output


def num_patches_per_image(image_shape, patch_x, patch_y, **kwargs):
    if len(image_shape) == 3 or len(image_shape) == 2: # x,y, channels
        return (image_shape[0] // patch_x) * (image_shape[1] // patch_y)
    elif len(image_shape) == 4: # N, x, y, channels
        return (image_shape[1] // patch_x) * (image_shape[2] // patch_y)
    else:
        raise ValueError('image_shape has wrong number of dimensions')


def get_multichannel_patches(data_or_masks, patch_x, patch_y, patch_stride_x, patch_stride_y, **kwargs):
    """

    Parameters
    ----------
    data_or_masks: ndarray of shape (N,x,y, n_chan)
    patch_x
    patch_y
    patch_stride_x
    patch_stride_y
    kwargs

    Returns
    -------
    patches of dtype float32

    """
    if data_or_masks is None:
        return None
    scaling_factor = (data_or_masks.shape[1] // patch_x) * (data_or_masks.shape[2] // patch_y)
    ret_patches = np.empty([data_or_masks.shape[0] * scaling_factor,
                            patch_x,
                            patch_y,
                            data_or_masks.shape[-1]], dtype='float32')
    p_size = (1, patch_x, patch_y, 1)
    s_size = (1, patch_stride_x, patch_stride_y, 1)
    rate = (1, 1, 1, 1)
    for ch in range(data_or_masks.shape[-1]):
        channel_data = np.expand_dims(data_or_masks[..., ch], -1)
        if data_or_masks.dtype == np.dtype('bool'):
            ret_patches[..., ch] = np.squeeze(get_patches(channel_data.astype('int'), p_size, s_size, rate, 'VALID').astype('bool'), axis=-1)
        else:
            ret_patches[..., ch] = np.squeeze(get_patches(channel_data, p_size, s_size, rate, 'VALID'), axis=-1)
    return ret_patches

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