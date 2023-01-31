from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve,auc 
#from utils.metrics import *
from inference import infer, get_error
import os
import numpy as np
import pandas as pd



def save_data_inferred(dir_path, data, masks_inferred, thresh=-1, figsize=(10,20)):
    # masks_inferred = self.infer(data)
    fig, ax = plt.subplots(len(data), 2, figsize=figsize)
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

def save_data_masks_inferred(dir_path, data, masks, masks_inferred, epoch=-1, thresh=-1, figsize=(10, 20)):
    # masks_inferred = self.infer(data)
    fig, ax = plt.subplots(len(data), 4, figsize=figsize)

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
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    if epoch > -1:
        fig.savefig('{}/epochs/epoch_{:04d}_image.png'.format(dir_path, epoch))
    else:
        if 0.0 < thresh < 1.0:
            fig.savefig('{}/data_masks_image_thresh_{:.2f}.png'.format(dir_path, thresh))
        else:
            fig.savefig('{}/data_masks_image.png'.format(dir_path))

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


# NO USAGES FOUND
def save_training_curves(model, args, test_images, test_labels, model_type):
    """
        Shows input vs output for each class for AE after training  
        
        model (tf.keras.Model): model
        args (Namespace): arguments from cmd_args
        test_images (np.array): testing images input
        test_labels (np.array): labels testing images input
        name (str): model name

    """
    model_output = infer(model[0], test_images, args, 'AE').astype(np.float32)
    error = get_error('AE',test_images, model_output).astype(np.float32)
    test_images = test_images.astype(np.float32)
    df = pd.DataFrame(columns=['Reconstruction'])

    labels = pd.unique(test_labels)
    fig, ax = plt.subplots(len(labels),3,figsize=(10,20))
    
    ax[0,1].title.set_text('Model Output')
    ax[0,2].title.set_text('Input - Output')

    for i,lbl in enumerate(labels):
        df.loc[lbl] =  error[test_labels==lbl].mean()
        ind = np.where(test_labels==lbl)[0][0]
        if test_images.shape[-1] == 1: #Mag only
            ax[i,0].imshow(test_images[ind,...,0]);
            ax[i,1].imshow(model_output[ind,...,0]);  
            ax[i,2].imshow(test_images[ind,...,0] - model_output[ind,...,0]);
        if test_images.shape[-1] == 3: #RGB
            ax[i,0].imshow(test_images[ind,...],vmin =0, vmax=1);
            ax[i,1].imshow(model_output[ind,...],vmin=0, vmax=1);  
            ax[i,2].imshow(test_images[ind,...] - model_output[ind,...],vmin=0, vmax=1);

        ax[i,0].title.set_text(lbl)
        ax[i,1].title.set_text(error[ind].mean())
    if not os.path.exists('outputs/{}/{}/{}'.format(model_type,
                                                    args.anomaly_class,
                                                    args.model_name)):

        os.makedirs('outputs/{}/{}/{}'.format(model_type,
                                              args.anomaly_class,
                                              args.model_name))
    plt.suptitle('Anomaly = {}'.format(args.anomaly_class))
    plt.tight_layout()
    fig.savefig('outputs/{}/{}/{}/io.png'.format(model_type,
                                                 args.anomaly_class,
                                                 args.model_name))
    plt.close('all')

    error = (error - np.min(error))/(np.max(error) - np.min(error))
    fpr, tpr, _ = roc_curve(test_labels==args.anomaly_class, error)
    a = auc(fpr,tpr)

    plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % a)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('outputs/{}/{}/{}/hist.png'.format(model_type,
                                                   args.anomaly_class,
                                                   args.model_name))

    df.to_csv('outputs/{}/{}/{}/data.csv'.format(model_type,
                                                 args.anomaly_class,
                                                 args.model_name))


# def save_flops_metric(dir_path, flops, metrics, labels, params, file_name='flops_metric', ylabel='score'):
#     fig = plt.figure(figsize=(10, 10))
#     for flop, metric, label, p in zip(flops, metrics, labels, params):
#         #flop /= 1e9
#         s = 500 * (p/1e6 - 0.5)/15 + 50
#         #plt.scatter(flop, metric, label=label, s=s)
#         plt.scatter(flop, metric, s=s)
#         #plt.text(flop*0.9, metric, label)
#         #text_label = label + f'\n {round(p/1e6,3)}'
#         text_label = label
#         plt.text(flop*0.9, metric, text_label, fontsize=15)
#         #plt.text(flop * 0.9, metric-0.01, round(p/1e6, 3))
#         #plt.scatter(flop, f1, label=label, s=p) We need to rescale p first
#
#     plt.xscale('log')
#     #plt.legend(fontsize=20)
#     plt.xlabel('FLOPs', fontsize=30)
#     plt.tick_params(axis='x', labelsize=20)
#     plt.ylabel(ylabel, fontsize=30)
#     plt.tick_params(axis='y', labelsize=20)
#     plt.tight_layout()
#     if not os.path.exists(dir_path):
#         os.makedirs(dir_path)
#     plt.savefig('{}/{}.png'.format(dir_path, file_name))
#     plt.close('all')




