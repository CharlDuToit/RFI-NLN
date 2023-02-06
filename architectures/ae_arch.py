from .generic_architecture import GenericArchitecture

from utils import (get_nln_metrics,
                           #save_metrics_csv,
                           evaluate_performance,
                           #save_results_csv,
                           #nln,
                            save_csv,
                           get_nln_errors,)
                           #get_dists)
from utils.profiling import (num_trainable_params,
                             num_non_trainable_params,
                             get_flops)
from utils import nln, get_dists
from utils.training import print_epoch, save_checkpoint_to_path
from utils import save_epochs_curve, save_data_masks_inferred, save_data_inferred_ae, save_data_nln_dists_combined
from data_collection import DataCollection

from utils.data import patches
import time
import os
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf


class AEArchitecture(GenericArchitecture):

    def __init__(self, model, args, checkpoint='None'):
        # model must be an autoencoder with property model.encoder
        super(AEArchitecture, self).__init__(model, args)
        self.loss_func = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.lr = 1e-4
        self.loss = 'mse'
        if isinstance(model, (tuple, list)):
            self.model = model[0]  # Autoencoder
        else:
            self.model = model
        if not os.path.exists(self.dir_path + '/inferred'):
            os.makedirs(self.dir_path + '/inferred')

    def train(self, data_collection: DataCollection):

        normal_train_data_dataset = tf.data.Dataset.from_tensor_slices(
            data_collection.normal_train_data).shuffle(self.buffer_size, seed=42).batch(self.batch_size)

        epoch_losses = []
        train_start = time.time()
        for epoch in range(self.epochs):
            start = time.time()
            epoch_loss = 0.0
            for image_batch in normal_train_data_dataset:
                loss = self.train_step(image_batch, image_batch)  # output must strive to input
                epoch_loss += loss

            epoch_loss /= len(normal_train_data_dataset)  # divide by number of batches
            epoch_losses.append(epoch_loss)
            with open(self.dir_path + '/epoch_losses.txt', 'a+') as f:
                f.write(f'{epoch_loss}\n')

            inds = random.sample(range(len(data_collection.normal_train_data)), self.num_samples)  # 10 random images
            data = data_collection.normal_train_data[inds]
            data_inferred = self.infer(data)
            self.save_data_images_ae(data,
                                     data_inferred,
                                     epoch)
            self.save_checkpoint(epoch)
            self.print_epoch(epoch, time.time() - start, epoch_loss, 'loss')

        print(f'Total training time: {(time.time() - train_start) // 60} min')
        self.save_checkpoint()
        self.save_training_metrics_image(epoch_losses, f'{self.model_type} loss')
        # self.save_summary()

    def save_data_images_ae(self, data, data_out, epoch=-1):
        save_data_inferred_ae(self.dir_path, data, data_out, epoch)

    def save_data_nln_dists_combined(self, neighbours, alpha, data, masks, x_hat, ae_error, nln_error, dists, combined):
        save_data_nln_dists_combined(self.dir_path+'/inferred', neighbours, alpha, data, masks, x_hat, ae_error,
                                     nln_error, dists, combined)

    # should still do the job of encoder and decoder (autoencoder)
    def infer(self, data):
        # assume self.model is not a list type e.g. (ae, disc) and that the model has only one output
        # i.e. len(model.outputs) == 1
        # data can be in patches or not

        data_tensor = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        output = np.empty(data.shape, dtype=np.float32)
        #output = np.empty([len(data)] + self.model.outputs[0].shape[1:], dtype=np.float32)
        strt, fnnsh = 0, 0
        for batch in data_tensor:
            fnnsh += len(batch)
            output[strt:fnnsh, ...] = self.model(batch, training=False).numpy()  # .astype(np.float32)
            strt = fnnsh

        output[output == np.inf] = np.finfo(output.dtype).max
        return output

    # to access encoder
    def infer_latent_dims(self, data):
        data_tensor = tf.data.Dataset.from_tensor_slices(data).batch(self.batch_size)
        output = np.empty( (len(data), self.latent_dim), dtype=np.float32)
        #output = np.empty([len(data)] + self.model.encoder.shape[1:], dtype=np.float32)
        strt, fnnsh = 0, 0
        for batch in data_tensor:
            fnnsh += len(batch)
            output[strt:fnnsh, ...] = self.model.encoder(batch, training=False).numpy()  # .astype(np.float32)
            strt = fnnsh

        return output

    def get_nln_errors(self, test_data, x_hat_train, ae_test_error, neighbours_idx, neighbour_mask):
        test_data_stacked = np.stack([test_data] * neighbours_idx.shape[-1], axis=1)
        neighbours = x_hat_train[neighbours_idx]

        error_nln = np.absolute(test_data_stacked - neighbours)
        error_nln = np.mean(error_nln, axis=1)  # nanmean for frNN
        error_nln[neighbour_mask] = ae_test_error[neighbour_mask]  # no effect for all False mask

        return error_nln

    def nln(self, z_train, z_test, n_neighbours):
        neighbours_dist, neighbours_idx, x_hat_train, neighbour_mask = nln(z_train,
                                                                           z_test,
                                                                           None,
                                                                           self.algorithm,
                                                                           n_neighbours)
        return neighbours_dist, neighbours_idx, neighbour_mask

    def evaluate_and_save(self, dc: DataCollection):
        # Dictionaries to be saved
        arch_dict = self.to_dict()
        data_dict = dc.to_dict()

        # Inference
        z_train = self.infer_latent_dims(dc.normal_train_data)
        z_test = self.infer_latent_dims(dc.test_data)
        x_hat_train = self.infer(dc.normal_train_data)
        x_hat_test = self.infer(dc.test_data)

        # Error
        ae_test_error = np.absolute(dc.test_data - x_hat_test)

        # Reconstruction
        ae_test_error_recon, test_labels_recon = dc.reconstruct(ae_test_error, dc.test_labels)
        test_masks_recon = dc.reconstruct(dc.test_masks)

        # Reconstruction for saving images
        test_data_recon = dc.reconstruct(dc.test_data)
        x_hat_test_recon = dc.reconstruct(x_hat_test)

        # Recon indexes to save to image
        inds = random.sample(range(len(ae_test_error_recon)), self.num_samples)  # 10 random images

        # AE Metrics
        ae_auroc, ae_auprc, ae_f1 = self.get_metrics(test_masks_recon, ae_test_error_recon)
        ae_dict = {'ae_auroc': ae_auroc, 'ae_auprc' : ae_auprc, 'ae_f1' : ae_f1}

        # ---------------------------------------------------------------------------------------
        # neighbours
        for neighbour in self.neighbours:

            # NLN
            neighbours_dist, neighbours_idx, neighbour_mask = self.nln(z_train, z_test, neighbour)
            nln_error = self.get_nln_errors(dc.test_data,
                                            x_hat_train,
                                            ae_test_error,
                                            neighbours_idx,
                                            neighbour_mask)
            # Reconstruct
            if dc.patches:
                if nln_error.ndim == 4:
                    nln_error_recon = dc.reconstruct(nln_error)
                else:
                    nln_error_recon = dc.reconstruct_latent_patches(nln_error)
            else:
                nln_error_recon = nln_error

            # NLN metrics
            nln_auroc, nln_auprc, nln_f1 = self.get_metrics(test_masks_recon, nln_error_recon)
            nln_dict = {'nln_auroc': nln_auroc, 'nln_auprc': nln_auprc, 'nln_f1': nln_f1}

            # Dists metrics
            dists_recon = dc.get_dists(neighbours_dist)
            dists_auroc, dists_auprc, dists_f1 = self.get_metrics(test_masks_recon, dists_recon)
            dists_dict = {'dists_auroc': dists_auroc, 'dists_auprc': dists_auprc, 'dists_f1': dists_f1}

            # ---------------------------------------------------------------------------------------
            # alphas
            for alpha in self.alphas: # alpha is scalar between 0.0 and 1.0

                # Combine NLN and dists
                combined_recon = dc.combine(nln_error_recon, dists_recon, alpha)

                # Combined metrics
                combined_auroc, combined_auprc, combined_f1 = self.get_metrics(test_masks_recon, combined_recon)
                combined_dict = {'combined_auroc': combined_auroc,
                                 'combined_auprc': combined_auprc,
                                 'combined_f1': combined_f1}

                # Save results to CSV
                results = {**ae_dict, **nln_dict, **dists_dict, **combined_dict, **arch_dict, **data_dict,
                           'alpha': alpha,
                           'neighbour': neighbour,
                           'time_patch': None,  # time_patch,
                           'time_image': None,  # time_image,
                           'flops_patch': None,  # flops_patch,
                           'flops_image': None,  # flops_image,
                           }
                save_csv(dc.data_name, dc.seed, results)

                # Save inferred images
                self.save_data_nln_dists_combined(neighbour,
                                                  alpha,
                                                  test_data_recon[inds],
                                                  test_masks_recon[inds],
                                                  x_hat_test_recon[inds],
                                                  ae_test_error_recon[inds],
                                                  nln_error_recon[inds],
                                                  dists_recon[inds],
                                                  combined_recon[inds])

        # Save solution info
        #self.save_summary()
        self.save_solution_config({**arch_dict, **data_dict, 'alphas': self.alphas, 'neighbours': self.neighbours})

    # def load_checkpoint(self):

    # def to_dict(self):