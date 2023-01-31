from utils.flagging import flag_data
from utils.metrics import get_dists
from utils.data.processor import process
from utils.data.patches import get_patches, reconstruct, reconstruct_latent_patches

#import copy
import numpy as np
import os
import errno
import pickle
from scipy.io import loadmat

import tensorflow as tf


class DataCollection:

    def __init__(self,
                 args,
                 std_minus=0.0,
                 std_plus=0.0,
                 combine_min_std_plus=None,
                 flag_test_data=False,
                 generate_normal_data=False,
                 clip_per_image=True,
                 scale_per_image=True,
                 log=True,
                 hyp_split=0.2):

        self.generate_normal_data = generate_normal_data
        self.hyp_split = hyp_split

        self.combine_min_std_plus = combine_min_std_plus
        self.std_minus = std_minus
        self.std_plus = std_plus
        if std_minus == 0.0 and std_plus == 0.0:
            self.clip_per_image = False
        else:
            self.clip_per_image = clip_per_image
        self.log = log
        self.clip_per_image = clip_per_image
        self.scale_per_image = scale_per_image

        self.input_channels = args.input_channels
        self.rfi_threshold = args.rfi_threshold
        if self.rfi_threshold is None:
            self.flag_test_data = False
        else:
            self.flag_test_data = flag_test_data

        self.seed = args.seed
        #self.raw_input_shape = args.raw_input_shape
        self.raw_input_shape = None  # will be determined from load_raw_data
        self.input_shape = None  # will be determined from load_raw_data and preprocess
        self.patches = args.patches
        self.patch_y = args.patch_y
        self.patch_stride_y = args.patch_stride_y
        self.patch_x = args.patch_x
        self.patch_stride_x = args.patch_stride_x
        self.anomaly_class = args.anomaly_class
        self.anomaly_type = args.anomaly_type
        self.data_name = args.data  # SHOULD ONLY BE USED IN load_raw_data(), to_dict() and flag_data()
        self.rfi = args.rfi  # hera rfi to exclude
        self.data_path = args.data_path
        self.lofar_subset = args.lofar_subset
        self.limit = args.limit

        self.raw_train_data = None
        self.raw_train_masks = None
        self.raw_test_masks = None
        self.raw_test_data = None

        self.train_data = None
        self.train_masks = None
        self.train_labels = None  # ['normal', 'rfi', 'normal'....]

        self.normal_train_data = None  # non contaminated images
        self.normal_train_labels = None  # ['normal', 'normal', 'normal' ...]

        self.test_data = None
        self.test_masks = None
        self.test_labels = None

        self.hyp_data = None
        self.hyp_masks = None
        self.hyp_labels = None

        self.normal_hyp_data = None
        self.normal_hyp_labels = None
        # self.test_masks_orig = None  # not generated via aoflagger

    def load_raw_data(self):
        """populates raw train data and masks and raw test data and masks"""

        # ======================================================
        if self.data_name in ['ant_fft_000_094_t4032_f4096',
                         'ant_fft_000_094_t4096_f4096',
                         'ant_fft_000_094_t8128_f8192',
                         'ant_fft_000_094_t12160_f16384']:
            file = os.path.join(self.data_path, f'{self.data_name}.mat')
            self.raw_train_data = np.expand_dims(loadmat(file)['sbdata'], (0, -1))/1e3
            #self.raw_train_data = np.expand_dims(loadmat(file)['sbdata'][0:512, 0:512], (0, -1))/1e3
            #print(self.raw_train_data.shape)
            self.raw_train_masks = None
            self.raw_test_data = None
            self.raw_test_masks = None
        # ======================================================
        if self.data_name == 'ASTRON_0':
            file = os.path.join(self.data_path, 'dyn_spectra_000_094.mat')
            data = loadmat(file)
            # ac000, ac094, vis000_094
            self.raw_train_data = np.empty((3,100,300,1), dtype='float32')
            self.raw_train_data[0, ...] = np.expand_dims(data['ac000'].astype('float32'),  -1)
            self.raw_train_data[1, ...] = np.expand_dims(data['ac094'].astype('float32'),  -1)
            self.raw_train_data[2, ...] = np.abs(np.expand_dims(data['vis000_094'].astype('float32'),  -1))
            self.raw_train_masks = None
            self.raw_test_data = None
            self.raw_test_masks = None

        # ======================================================
        if self.data_name == 'LOFAR':
            full_file = file = 'LOFAR_Full_RFI_dataset.pkl'
            if self.lofar_subset is None or self.lofar_subset == 'full':
                file = full_file
            if self.lofar_subset == 'L629174':
                file = 'L629174_RFI_dataset.pkl'
            if self.lofar_subset == 'L631961':
                file = 'L631961_RFI_dataset.pkl'

            if os.path.exists(os.path.join(self.data_path, file)):
                print(os.path.join(self.data_path, file) + ' Loading')
                with open('{}/{}'.format(self.data_path, file), 'rb') as f:
                    (self.raw_train_data, self.raw_train_masks, self.raw_test_data, self.raw_test_masks) = pickle.load(
                        f)
                    #self.raw_input_shape = self.raw_train_data.shape[1:]
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        os.path.join(self.data_path, file))
        # ======================================================
        if self.data_name in ['HERA', 'HERA_PHASE']:
            if self.data_name == 'HERA':
                date = '04-03-2022'
            elif self.data_name == 'HERA_PHASE':
                date = '07-11-2022'

            if self.rfi is not None:
                rfi_models = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']
                (_, _, test_data,
                 test_masks) = np.load('{}/HERA_{}_{}.pkl'.format(self.data_path, date, self.rfi), allow_pickle=True)

                rfi_models.remove(self.rfi)

                (train_data,
                 train_masks, _, _) = np.load('{}/HERA_{}_{}.pkl'.format(self.data_path, date, '-'.join(rfi_models)),
                                              allow_pickle=True)

            else:
                (train_data, train_masks,
                 test_data, test_masks) = np.load('{}/HERA_{}_all.pkl'.format(self.data_path, date),
                                                  allow_pickle=True)
            train_data[train_data == np.inf] = np.finfo(train_data.dtype).max
            test_data[test_data == np.inf] = np.finfo(test_data.dtype).max

            self.raw_train_data = train_data.astype('float32')
            self.raw_test_data = test_data.astype('float32')
            self.raw_train_masks = train_masks
            self.raw_test_masks = test_masks

        # ======================================================
        self.raw_input_shape = self.raw_train_data.shape[1:]

        # After these blocks: self.raw_input_shape[-1] = self.input_channels
        if self.raw_input_shape[-1] != self.input_channels:
            print(f'raw channels: {self.raw_input_shape[-1]}, args channels: {self.input_channels}')
        if self.input_channels < 1:
            print('args channels less than 1, using raw channels')
            self.input_channels = self.raw_input_shape[-1]
        elif self.raw_input_shape[-1] < self.input_channels:
            print('args channels more than raw channels, using raw channels')
            self.input_channels = self.raw_input_shape[-1]
        elif self.raw_input_shape[-1] > self.input_channels:
            print(f'extracting first {self.input_channels} channels from data')
            self.raw_train_data = self.raw_train_data[..., 0:self.input_channels]
            self.raw_test_data = self.raw_test_data[..., 0:self.input_channels]
            self.raw_train_masks = self.raw_train_masks[..., 0:self.input_channels]
            self.raw_test_masks = self.raw_test_masks[..., 0:self.input_channels]
            self.raw_input_shape = self.raw_train_data.shape[1:]

    def preprocess(self):
        train_data = self.raw_train_data
        test_data = self.raw_test_data
        train_masks = self.raw_train_masks
        test_masks = self.raw_test_masks

        hyp_data = None
        hyp_masks = None
        hyp_labels = None
        train_labels = None
        test_labels = None
        normal_hyp_data = None
        normal_hyp_labels = None

        if self.limit is not None:
            train_indx = np.random.permutation(len(train_data))[:self.limit]  # lets keep the seed random
            train_data = train_data[train_indx]
            if train_masks is not None:
                train_masks = train_masks[train_indx]

        # test_masks_orig = copy.deepcopy(test_masks)
        if self.rfi_threshold is not None:
            train_masks = flag_data(train_data[..., 0], self.data_name, self.rfi_threshold)
            train_masks = np.expand_dims(train_masks, axis=-1)
            if self.flag_test_data:
                test_masks = flag_data(test_data[..., 0], self.data_name, self.rfi_threshold)
                test_masks = np.expand_dims(test_masks, axis=-1)

        # hyperparam data
        if 0.0 < self.hyp_split < 1.0:
            n_hyp = int(len(train_data) * self.hyp_split)
            rand_indx = np.random.RandomState(seed=42).permutation(len(train_data))  # always same indx
            hyp_indx = rand_indx[:n_hyp]
            train_indx = rand_indx[n_hyp:]
            hyp_data = train_data[hyp_indx]
            hyp_masks = train_masks[hyp_indx]
            train_data = train_data[train_indx]
            train_masks = train_masks[train_indx]

        # clip
        #if self.clip_per_image is False:
        if test_data is not None:
            test_data[..., 0] = self.get_clipped_by_minrfi_maxnonrfi(test_data[..., 0], test_masks[..., 0])
        if train_data is not None:
            train_data[..., 0] = self.get_clipped_by_minrfi_maxnonrfi(train_data[..., 0], train_masks[..., 0])
        if hyp_data is not None:
            hyp_data[..., 0] = self.get_clipped_by_minrfi_maxnonrfi(hyp_data[..., 0], hyp_masks[..., 0])


        # log
        if self.log:
            if test_data is not None:
                test_data[..., 0] = np.log(test_data[..., 0])
            if train_data is not None:
                train_data[..., 0] = np.log(train_data[..., 0])
            if hyp_data is not None:
                hyp_data[..., 0] = np.log(hyp_data[..., 0])

        # scale
        #if self.scale:
        if test_data is not None:
            test_data[..., 0] = self.rescale(test_data[..., 0])
        if train_data is not None:
            train_data[..., 0] = self.rescale(train_data[..., 0])
        if hyp_data is not None:
            hyp_data[..., 0] = self.rescale(hyp_data[..., 0])

        # patches
        if self.patches:
            train_data = self.get_patches(train_data)
            if train_masks is not None:
                train_masks = self.get_patches(train_masks)
            if test_data is not None:
                test_data = self.get_patches(test_data)
            if test_masks is not None:
                test_masks = self.get_patches(test_masks)
            if hyp_data is not None:
                hyp_data = self.get_patches(hyp_data)
            if hyp_masks is not None:
                hyp_masks = self.get_patches(hyp_masks)
        else:
            self.patch_x = self.raw_input_shape[0]
            self.patch_y = self.raw_input_shape[1]

        # labels
        if train_masks is not None:
            train_labels = self.get_labels(train_masks)
        if test_masks is not None:
            test_labels = self.get_labels(test_masks)
        if hyp_masks is not None:
            hyp_labels = self.get_labels(hyp_masks)

        # normal data
        if self.generate_normal_data:
            normal_train_data, normal_train_labels = self.get_normal_data(train_data, train_masks)
            self.normal_train_data = normal_train_data
            self.normal_train_labels = normal_train_labels

            normal_hyp_data, normal_hyp_labels = self.get_normal_data(hyp_data, hyp_masks)
            self.normal_hyp_data = normal_hyp_data
            self.normal_hyp_labels = normal_hyp_labels

        # transfer variables to .self
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_data = train_data
        self.train_masks = train_masks
        self.test_data = test_data
        self.test_masks = test_masks

        self.hyp_data = hyp_data
        self.hyp_masks = hyp_masks
        self.hyp_labels = hyp_labels

        self.input_shape = self.train_data.shape[1:]

        # Free up memory
        self.raw_train_data = None
        self.raw_test_data = None
        self.raw_test_masks = None
        self.raw_train_masks = None
        #train_normal_ratio = len(normal_train_data) / len(train_data)
        #hyp_normal_ratio = len(normal_hyp_data) / len(hyp_data)

    def get_dists(self, neighbours_dist):
        return get_dists(neighbours_dist, self.raw_input_shape, self.patch_x, self.patch_y)

    def combine(self, nln_error, dists, alpha):
        # alpha between 0.0 and 1.0
        # LOFAR optimal alpha = 0.66
        # HERA optimal alpha = 0.10
        if self.combine_min_std_plus is None:
            combined_recon = nln_error * np.array([d > np.percentile(d, alpha * 100) for d in dists])
        else:
            nln_error_clipped = np.clip(nln_error,
                                        nln_error.mean() + nln_error.std() * np.abs(self.combine_min_std_plus),
                                        1.0)
            combined_recon = nln_error_clipped * np.array([d > np.percentile(d, alpha * 100) for d in dists])
        combined_recon = np.nan_to_num(combined_recon)
        return combined_recon

    def patches_per_image(self):
        return (self.raw_input_shape[0] // self.patch_x) * (self.raw_input_shape[1] // self.patch_y)

    def rescale(self, data):
        if data is None:
            return None
        return process(data, per_image=self.scale_per_image)

    def get_normal_data(self, data, masks):
        if data is None or masks is None:
            return None
        normal_data = data[np.invert(np.any(masks, axis=(1, 2, 3)))]
        labels = ['normal'] * len(normal_data)
        return normal_data, labels

    def get_labels(self, masks):
        if masks is None:
            return None
        labels = np.empty(len(masks), dtype='object')
        labels[np.any(masks, axis=(1, 2, 3))] = self.anomaly_class
        labels[np.invert(np.any(masks, axis=(1, 2, 3)))] = 'normal'
        return labels

    def get_clipped_by_minrfi_maxnonrfi(self, data, masks):
        if data is None or masks is None:
            return None
        if self.clip_per_image is False:
            nonrfi_max = np.max(data[~masks])
            rfi_min = np.min(data[masks])
            if nonrfi_max > rfi_min:
                return np.clip(data, nonrfi_max, rfi_min)
            else:
                return np.clip(data, rfi_min, nonrfi_max)
        else:
            result = np.empty(data.shape, dtype=data.dtype)
            for i in range(len(data)):
                nonrfi_max = np.max(data[i][~masks[i]])
                rfi_min = np.min(data[i][masks[i]])
                if nonrfi_max > rfi_min:
                    result[i] = np.clip(data[i], rfi_min, nonrfi_max)
                else:
                    result[i] = np.clip(data[i], nonrfi_max, rfi_min)
            return result

    def get_clipped(self, data, masks):
        if data is None:
            return None
        if masks is None:
            return self.get_clipped_data(data)
        # mean and std of nonrfi
        #_max = np.mean(data[np.invert(masks)]) + np.abs(self.std_plus) * np.std(data[np.invert(masks)])
        _max = np.mean(data[np.invert(masks)]) + self.std_plus * np.std(data[np.invert(masks)])
        #_min = np.absolute(np.mean(data[np.invert(masks)]) - np.abs(self.std_minus) * np.std(data[np.invert(masks)]))
        _min = np.mean(data[np.invert(masks)]) - np.abs(self.std_minus) * np.std(data[np.invert(masks)])
        return np.clip(data, _min, _max)

    def get_clipped_data(self, data):
        if data is None:
            return None
        std = np.std(data)
        mean = np.mean(data)
        #_max = mean + np.abs(self.std_plus) * std
        _max = mean + self.std_plus * std
        #_min = np.absolute(mean - np.abs(self.std_minus) * std)
        _min = mean - np.abs(self.std_minus) * std
        return np.clip(data, _min, _max)

    def flag_data(self, data):
        if data is None:
            return None
        train_masks = flag_data(data, self.data_name, self.rfi_threshold)
        return np.expand_dims(train_masks, axis=-1)

    def get_patches(self, data_or_masks):
        if data_or_masks is None:
            return None
        scaling_factor = (data_or_masks.shape[1] // self.patch_x) * (data_or_masks.shape[2] // self.patch_y)
        ret_patches = np.empty([data_or_masks.shape[0] * scaling_factor,
                                self.patch_x,
                                self.patch_y,
                                data_or_masks.shape[-1]], dtype='float32')
        for ch in range(data_or_masks.shape[-1]):
            p_size = (1, self.patch_x, self.patch_y, 1)
            s_size = (1, self.patch_stride_x, self.patch_stride_y, 1)
            rate = (1, 1, 1, 1)
            channel_data = np.expand_dims(data_or_masks[..., ch], -1)
            if data_or_masks.dtype == np.dtype('bool'):
                ret_patches[..., ch] = np.squeeze(get_patches(channel_data.astype('int'), p_size, s_size, rate, 'VALID').astype('bool'), axis=-1)
            else:
                ret_patches[..., ch] = np.squeeze(get_patches(channel_data, p_size, s_size, rate, 'VALID'), axis=-1)
        return ret_patches
        # p_size = (1, self.patch_x, self.patch_y, data_or_masks.shape[-1])
        # s_size = (1, self.patch_stride_x, self.patch_stride_y, 1)
        # rate = (1, 1, 1, 1)
        # if data_or_masks.dtype == np.dtype('bool'):
        #     return get_patches(data_or_masks.astype('int'), p_size, s_size, rate, 'VALID').astype('bool')
        # else:
        #     return get_patches(data_or_masks, p_size, s_size, rate, 'VALID')

    def reconstruct(self, patches, labels=None):
        return reconstruct(patches, self.raw_input_shape, self.patch_x, self.patch_y, self.anomaly_class, labels)

    def reconstruct_latent_patches(self, patches, labels=None):
        reconstruct_latent_patches(patches, self.raw_input_shape, self.patch_x, self.patch_y, self.anomaly_class,
                                   labels)

    #def load(self):
    #    self.load_raw_data()
    #    self.preprocess()

    def get_datasets(self, val_split=0.2, use_hyp_data=False, use_normal_data=False, buffer_size=2**10, batch_size=64, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**16)
        if not use_hyp_data:
            if use_normal_data:
                data = self.normal_train_data
                masks = None
            else:
                data = self.train_data
                masks = self.train_masks
        else:
            if use_normal_data:
                data = self.normal_hyp_data
                masks = None
            else:
                data = self.hyp_data
                masks = self.hyp_masks

        train_indx = [i for i in range(len(data))]
        val_indx = []
        val_mask_dataset = None
        val_data_dataset = None
        if 0.0 < val_split < 1.0:
            n_val = int(len(data) * val_split)
            #rand_indx = np.random.permutation(len(data))  # random seed
            rand_indx = np.random.RandomState(seed=seed).permutation(len(data))  # random or provided seed
            val_indx = rand_indx[:n_val]
            train_indx = rand_indx[n_val:]
            # shuffle seed can be fixed, since np randomstate has a random or provided seed
            val_data_dataset = tf.data.Dataset.from_tensor_slices(
                data[val_indx]).shuffle(buffer_size, seed=42).batch(batch_size)
            if not use_normal_data:
                val_mask_dataset = tf.data.Dataset.from_tensor_slices(
                    masks[val_indx].astype('float32')).shuffle(buffer_size, seed=42).batch(batch_size)

        train_data_dataset = tf.data.Dataset.from_tensor_slices(
            data[train_indx]).shuffle(buffer_size, seed=42).batch(batch_size)

        train_mask_dataset = None
        if not use_normal_data:
            train_mask_dataset = tf.data.Dataset.from_tensor_slices(
                masks[train_indx].astype('float32')).shuffle(buffer_size, seed=42).batch(batch_size)

        num_train = len(train_indx)
        num_val = len(val_indx)

        return train_data_dataset, val_data_dataset, train_mask_dataset, val_mask_dataset, num_train, num_val, seed

    def all_not_none(self, properties):
        for prop in properties:
            if prop == 'train_data' and self.train_data is None:
                return False
            if prop == 'test_data' and self.test_data is None:
                return False
            if prop == 'train_masks' and self.train_masks is None:
                return False
            if prop == 'test_masks' and self.test_masks is None:
                return False
            if prop == 'train_labels' and self.train_labels is None:
                return False
            if prop == 'test_labels' and self.test_labels is None:
                return False
            if prop == 'normal_train_data' and self.normal_train_data is None:
                return False
            if prop == 'normal_train_labels' and self.normal_train_labels is None:
                return False
            # if prop == 'test_masks_orig' and self.test_masks_orig is None:
            #    return False
            if prop == 'raw_train_data' and self.raw_train_data is None:
                return False
            if prop == 'raw_train_masks' and self.raw_train_masks is None:
                return False
            if prop == 'raw_test_masks' and self.raw_test_masks is None:
                return False
            if prop == 'raw_test_data' and self.raw_test_data is None:
                return False
            if prop == 'hyp_data' and self.hyp_data is None:
                return False
            if prop == 'hyp_masks' and self.hyp_masks is None:
                return False
            if prop == 'hyp_labels' and self.hyp_labels is None:
                return False
            if prop == 'normal_hyp_data' and self.normal_hyp_data is None:
                return False
            if prop == 'normal_hyp_labels' and self.normal_hyp_labels is None:
                return False
        return True

    def to_dict(self):
        return {
            'raw_input_shape': self.raw_input_shape,
            'input_shape': self.input_shape,
            'num_patches': self.patches_per_image(),
            'limit': self.limit,
            'std_plus': self.std_plus,
            'std_minus': self.std_minus,
            'scale_per_image': self.scale_per_image,
            'clip_per_image': self.clip_per_image,
            'patch_x': self.patch_x,
            'patch_y': self.patch_y,
            'anomaly_class': self.anomaly_class,
            'anomaly_type': self.anomaly_type,
            'data': self.data_name,
            'rfi_threshold': self.rfi_threshold,
            'ood_rfi': self.rfi,
            'lofar_subset': self.lofar_subset,
            'combine_min_std_plus': self.combine_min_std_plus,
        }
