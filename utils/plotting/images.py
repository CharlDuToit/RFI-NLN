from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc 
#from utils.metrics import *
#from inference import infer, get_error
import os
import numpy as np
#import pandas as pd


def save_image_masks_batches(dir_path, data, masks, batch_size=20, figsize=(20, 60)):
    for i in range(0, len(data), batch_size):
        strt = i
        fnsh = np.minimum(strt + batch_size, len(data))
        d = data[strt:fnsh, ..., 0]
        m = masks[strt:fnsh, ..., 0]

        fig, ax = plt.subplots(len(d), 2, figsize=figsize)

        ax[0, 0].title.set_text('Input')
        ax[0, 1].title.set_text('True Mask')
        for j in range(len(d)):
            ax[j, 0].imshow(d[j])
            ax[j, 1].imshow(m[j])

        plt.tight_layout()
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        fig.savefig(os.path.join(dir_path, f'image_masks_{strt}'))

        plt.close('all')


def save_image_batches_grid(dir_path, data, grid_size=10, figsize=(50, 50)):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for i in range(0, len(data), grid_size*grid_size):
        strt = i
        fnsh = np.minimum(strt + grid_size*grid_size, len(data))

        if data.shape[-1] in (1, 2):
            d = data[strt:fnsh, ..., 0]
        else:
            d = data[strt:fnsh, ...]

        fig, ax = plt.subplots(grid_size, grid_size, figsize=figsize)

        # ax[0].title.set_text('Input')
        for j in range(len(d)):
            ax[j // grid_size, j % grid_size].imshow(d[j])

        plt.tight_layout()

        fig.savefig(os.path.join(dir_path, f'image_{strt}'))

        plt.close('all')


def save_image_batches(dir_path, data, batch_size=20, figsize=(20, 60)):
    for i in range(0, len(data), batch_size):
        strt = i
        fnsh = np.minimum(strt + batch_size, len(data))

        if data.shape[-1] in (1, 2):
            d = data[strt:fnsh, ..., 0]
        else:
            d = data[strt:fnsh, ...]

        fig, ax = plt.subplots(len(d), 1, figsize=figsize)

        ax[0].title.set_text('Input')
        for j in range(len(d)):
            ax[j].imshow(d[j])

        plt.tight_layout()
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        fig.savefig(os.path.join(dir_path, f'image_{strt}'))

        plt.close('all')


def save_data_masks(dir_path, data, true_masks, figsize=(10, 20)):
    # masks_inferred = self.infer(data)
    fig, ax = plt.subplots(len(data), 2, figsize=figsize)

    if len(data) == 1:
        ax[0].title.set_text('Input')
        ax[1].title.set_text('True Mask')
        ax[0].imshow(data[0, ..., 0])
        ax[1].imshow(true_masks[0, ..., 0])
    else:
        ax[0, 0].title.set_text('Input')
        ax[0, 1].title.set_text('True Mask')
        for i in range(len(data)):
            ax[i, 0].imshow(data[i, ..., 0])
            ax[i, 1].imshow(true_masks[i, ..., 0])

    plt.tight_layout()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    fig.savefig('{}/data_true_masks_image.png'.format(dir_path))

    plt.close('all')


def save_data_inferred(dir_path, data, masks_inferred, thresh=-1, figsize=(10,20)):
    # masks_inferred = self.infer(data)
    fig, ax = plt.subplots(len(data), 2, figsize=figsize)

    if 0.0 < thresh < 1.0:
        masks_inferred = masks_inferred > thresh

    if len(data) == 1:
        ax[0].title.set_text('Input')
        ax[1].title.set_text('Mask Inferred')
        ax[0].imshow(data[0, ..., 0])
        ax[1].imshow(masks_inferred[0, ..., 0])
    else:
        ax[0, 0].title.set_text('Input')
        ax[0, 1].title.set_text('Mask Inferred')
        for i in range(len(data)):
            ax[i, 0].imshow(data[i, ..., 0])
            ax[i, 1].imshow(masks_inferred[i, ..., 0])

    #fig, ax = plt.subplots(np.maximum(len(data), 2), 2, figsize=figsize)

    plt.tight_layout()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if 0.0 < thresh < 1.0:
        fig.savefig('{}/data_masks_image_thresh_{:.2f}.png'.format(dir_path, thresh))
    else:
        fig.savefig('{}/data_masks_image.png'.format(dir_path))

    plt.close('all')

def save_data_masks_inferred(dir_path, data, masks, masks_inferred, epoch=-1, thresh=-1, model_type=None, figsize=(10, 20)):
    # masks_inferred = self.infer(data)
    fig, ax = plt.subplots(len(data), 4, figsize=figsize)

    if 0.0 < thresh < 1.0:
        masks_inferred = masks_inferred > thresh

    if len(data) == 1:
        ax[0].title.set_text('Input')
        ax[1].title.set_text('Mask')
        ax[2].title.set_text('Mask Inferred')
        ax[3].title.set_text('Absolute Error')
        ax[0].imshow(data[0, ..., 0])
        ax[1].imshow(masks[0, ..., 0])
        ax[2].imshow(masks_inferred[0, ..., 0])
        ax[3].imshow(np.absolute(masks[0, ..., 0] - masks_inferred[0, ..., 0]))

    else:
        ax[0, 0].title.set_text('Input')
        ax[0, 1].title.set_text('Mask')
        ax[0, 2].title.set_text('Mask Inferred')
        ax[0, 3].title.set_text('Absolute Error')

        for i in range(len(data)):
            ax[i, 0].imshow(data[i, ..., 0])
            ax[i, 1].imshow(masks[i, ..., 0])
            ax[i, 2].imshow(masks_inferred[i, ..., 0])
            ax[i, 3].imshow(np.absolute(masks[i, ..., 0] - masks_inferred[i, ..., 0]))

    plt.tight_layout()

    if not model_type:
        model_type = ''
    else:
        model_type += '_'

    if not os.path.exists(os.path.join(dir_path, 'epochs')):
        os.makedirs(os.path.join(dir_path, 'epochs'))
    if epoch > -1:
        fig.savefig('{}/epochs/{}epoch_{:04d}_image.png'.format(dir_path, model_type, epoch))
    else:
        if 0.0 < thresh < 1.0:
            fig.savefig('{}/{}data_masks_image_thresh_{:.2f}.png'.format(dir_path, model_type, thresh))
        else:
            fig.savefig('{}/{}data_masks_image.png'.format(dir_path, model_type))

    plt.close('all')


def save_data_cells_boundingboxes(dir_path, data, cells, cells_inferred, epoch=-1, thresh=0.25, model_type=None, figsize=(10, 20)):
    from utils import images_cells_to_images_bounding_boxes, draw_images_bounding_boxes
    # from utils import draw_images_bounding_boxes, images_cells_to_images_bounding_boxes_v3

    # true_cells = unnormalize_images_cells(data.shape[1:], cells)
    # pred_cells = unnormalize_images_cells(data.shape[1:], cells_inferred)

    true_bboxes = images_cells_to_images_bounding_boxes(data.shape[1:], cells, thresh)
    pred_bboxes = images_cells_to_images_bounding_boxes(data.shape[1:], cells_inferred, thresh)

    # true_bboxes = images_cells_to_images_bounding_boxes_v3(data.shape[1:], cells, 0.25)
    # pred_bboxes = images_cells_to_images_bounding_boxes_v3(data.shape[1:], cells_inferred, 0.25)

    empty_im = np.zeros(data.shape)
    true_images = draw_images_bounding_boxes(empty_im, true_bboxes)
    pred_images = draw_images_bounding_boxes(empty_im, pred_bboxes)

    # masks_inferred = self.infer(data)
    fig, ax = plt.subplots(len(data), 3, figsize=figsize)

    if len(data) == 1:
        ax[0].title.set_text('Input')
        ax[1].title.set_text('True Bounding boxes')
        ax[2].title.set_text('Inferred Bounding boxes')
        ax[0].imshow(data[0, ..., 0])
        ax[1].imshow(true_images[0, ...])
        ax[2].imshow(pred_images[0, ...])

    else:
        ax[0, 0].title.set_text('Input')
        ax[0, 1].title.set_text('True Bounding boxes')
        ax[0, 2].title.set_text('Inferred Bounding boxes')

        for i in range(len(data)):
            ax[i, 0].imshow(data[i, ..., 0])
            ax[i, 1].imshow(true_images[i, ...])
            ax[i, 2].imshow(pred_images[i, ...])

    plt.tight_layout()

    if not model_type:
        model_type = ''
    else:
        model_type += '_'

    if not os.path.exists(os.path.join(dir_path, 'epochs')):
        os.makedirs(os.path.join(dir_path, 'epochs'))
    if epoch > -1:
        fig.savefig('{}/epochs/{}epoch_{:04d}_image.png'.format(dir_path, model_type, epoch))
    else:
        if 0.0 < thresh < 1.0:
            fig.savefig('{}/{}data_masks_image_thresh_{:.2f}.png'.format(dir_path, model_type, thresh))
        else:
            fig.savefig('{}/{}data_masks_image.png'.format(dir_path, model_type))

    plt.close('all')



def save_data_inferred_ae(dir_path, data, data_inferred, epoch=-1):
    fig, ax = plt.subplots(len(data), 3, figsize=(10, 20))

    ax[0, 0].title.set_text('Input')
    ax[0, 1].title.set_text('AE Output')
    ax[0, 2].title.set_text('Absolute Error')

    for i in range(len(data)):
        ax[i, 0].imshow(data[i, ..., 0])
        ax[i, 1].imshow(data_inferred[i, ..., 0])
        ax[i, 2].imshow(np.absolute(data[i, ..., 0] - data_inferred[i, ..., 0]))

    plt.tight_layout()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if epoch > -1:
        fig.savefig('{}/epochs/epoch_{:04d}_image.png'.format(dir_path, epoch))
    else:
        fig.savefig('{}/input_output_image.png'.format(dir_path))

    plt.close('all')

def save_data_nln_dists_combined(dir_path, neighbours, alpha, data, masks, x_hat, ae_error, nln_error, dists, combined):

    fig, axs = plt.subplots(10, 7, figsize=(10, 8))
    axs[0, 0].set_title('Input', fontsize=5)
    axs[0, 1].set_title('Mask', fontsize=5)  # test_mask
    axs[0, 2].set_title('AE Output', fontsize=5)  # test_image - x_hat
    axs[0, 3].set_title('AE Error', fontsize=5)  # test_image - x_hat
    axs[0, 4].set_title('NLN Error', fontsize=5)
    axs[0, 5].set_title('Dists', fontsize=5)
    axs[0, 6].set_title('Combined', fontsize=5)

    for i in range(len(data)):
        axs[i, 0].imshow(data[i, ..., 0].astype(np.float32), vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        axs[i, 1].imshow(masks[i, ..., 0].astype(np.float32), vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        axs[i, 2].imshow(x_hat[i, ..., 0].astype(np.float32), vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        axs[i, 3].imshow(ae_error[i, ..., 0].astype(np.float32), vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        axs[i, 4].imshow(nln_error[i, ..., 0].astype(np.float32), interpolation='nearest', aspect='auto')
        axs[i, 5].imshow(dists[i, ..., 0].astype(np.float32), vmin=0, vmax=1, interpolation='nearest', aspect='auto')
        axs[i, 6].imshow(combined[i, ..., 0].astype(np.float32), vmin=0, vmax=1, interpolation='nearest', aspect='auto')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.savefig('{}/neighbours_{}_alpha_{}.png'.format(dir_path, neighbours, alpha), dpi=300)


def save_data_masks_dknn(dir_path, data_recon, masks_recon, dists_recon):
    fig, ax = plt.subplots(len(data_recon), 3, figsize=(10, 20))
    #fig, ax = plt.subplots(len(data_recon), 4, figsize=(10, 20))

    ax[0, 0].title.set_text('Input')
    ax[0, 1].title.set_text('Mask')
    ax[0, 2].title.set_text('Dists Recon')
    #ax[0, 3].title.set_text('Mask minus Output')

    for i in range(len(data_recon)):
        ax[i, 0].imshow(data_recon[i, ..., 0])
        ax[i, 1].imshow(masks_recon[i, ..., 0])
        ax[i, 2].imshow(dists_recon[i, ..., 0])
        #ax[i, 3].imshow(masks_recon[i, ..., 0] - dists_recon[i, ..., 0])

    plt.tight_layout()
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    fig.savefig('{}/data_masks_image.png'.format(dir_path))
    plt.close('all')

def generate_and_save_images(model, epoch, test_input, model_type, args):
    """
        Shows input vs output plot for AE while trainging
        
        model (tf.keras.Model): model
        epoch (int): current epoch number 
        test_input (np.array): testing images input
        name (str): model name
        args (Namespace): arguments from cmd_args

    """
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    if model_type== 'VAE' or model_type == 'VAEGAN':
        mean, logvar = model.encoder(test_input,vae=True)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
    elif model_type== 'BIGAN':
        predictions = test_input
    else:
        predictions = model(test_input, training=False)
    predictions = predictions.numpy().astype(np.float32)
    fig = plt.figure(figsize=(10,10))

    for i in range(predictions.shape[0]):
      plt.subplot(5, 5, i+1)
      if predictions.shape[-1] == 1:#1 channel only
          plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)

      if predictions.shape[-1] == 3: #RGB
          plt.imshow(predictions[i,...], vmin=0, vmax=1)
      plt.axis('off')
    
    if not os.path.exists('outputs/{}/{}/{}/epochs/'.format(model_type,
                                                            args.anomaly_class,
                                                            args.model_name)):

        os.makedirs('outputs/{}/{}/{}/epochs/'.format(model_type,
                                                      args.anomaly_class,
                                                      args.model_name))

    plt.tight_layout()
    plt.savefig('outputs/{}/{}/{}/epochs/image_at_epoch_{:04d}.png'.format(model_type,
                                                                           args.anomaly_class,
                                                                           args.model_name,
                                                                           epoch))
    plt.close('all')


