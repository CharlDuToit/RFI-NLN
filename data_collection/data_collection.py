from utils.flagging import flag_data
from utils.metrics import get_dists
from utils.data.processor import process
from utils.data.patches import get_patches, reconstruct, reconstruct_latent_patches

import copy
import numpy as np
import os
import errno
import pickle


class DataCollection:

    def __init__(self,
                 args,
                 std_minus=0.0,
                 std_plus=0.0,
                 combine_min_std_plus=None,
                 flag_test_data=False,
                 generate_normal_data=False,
                 per_image=False,
                 clip=True,
                 log=True,
                 scale=True):

        self.generate_normal_data = generate_normal_data

        self.combine_min_std_plus = combine_min_std_plus
        self.std_minus = std_minus
        self.std_plus = std_plus
        if std_minus == 0.0 and std_plus == 0.0:
            self.clip = False
        else:
            self.clip = clip
        self.log = log
        self.scale = scale
        self.per_image = per_image

        self.rfi_threshold = args.rfi_threshold
        if self.rfi_threshold is None:
            self.flag_test_data = False
        else:
            self.flag_test_data = flag_test_data

        self.seed = args.seed
        self.raw_input_shape = args.raw_input_shape
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
        # self.test_masks_orig = None  # not generated via aoflagger

    def load_raw_data(self):
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
                    self.raw_input_shape = self.raw_train_data.shape[1:]
            else:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                        os.path.join(self.data_path, file))
        if self.data_name == 'HERA':
            if self.rfi is not None:
                rfi_models = ['rfi_stations', 'rfi_dtv', 'rfi_impulse', 'rfi_scatter']
                (_, _, test_data,
                 test_masks) = np.load('{}/HERA_04-03-2022_{}.pkl'.format(self.data_path, self.rfi), allow_pickle=True)
                rfi_models.remove(self.rfi)

                (train_data,
                 train_masks, _, _) = np.load('{}/HERA_04-03-2022_{}.pkl'.format(self.data_path, '-'.join(rfi_models)),
                                              allow_pickle=True)

            else:
                (train_data, train_masks,
                 test_data, test_masks) = np.load('{}/HERA_04-03-2022_all.pkl'.format(self.data_path),
                                                  allow_pickle=True)
            train_data[train_data == np.inf] = np.finfo(train_data.dtype).max
            test_data[test_data == np.inf] = np.finfo(test_data.dtype).max

            self.raw_train_data = train_data.astype('float32')
            self.raw_test_data = test_data.astype('float32')
            self.raw_train_masks = train_masks
            self.raw_test_masks = test_masks
            self.raw_input_shape = self.raw_train_data.shape[1:]

    def preprocess(self):
        train_data = self.raw_train_data
        test_data = self.raw_test_data
        train_masks = self.raw_train_masks
        test_masks = self.raw_test_masks

        if self.limit is not None:
            train_indx = np.random.permutation(len(train_data))[:self.limit]
            train_data = train_data[train_indx]
            train_masks = train_masks[train_indx]

        # test_masks_orig = copy.deepcopy(test_masks)
        if self.rfi_threshold is not None:
            train_masks = flag_data(train_data, self.data_name, self.rfi_threshold)
            train_masks = np.expand_dims(train_masks, axis=-1)
            if self.flag_test_data:
                test_masks = flag_data(test_data, self.data_name, self.rfi_threshold)
                test_masks = np.expand_dims(test_masks, axis=-1)

        if self.clip:
            test_data = self.get_clipped(test_data, test_masks)
            train_data = self.get_clipped(train_data, train_masks)

        if self.log:
            test_data = np.log(test_data)
            train_data = np.log(train_data)
        if self.scale:
            test_data = self.rescale(test_data)
            train_data = self.rescale(train_data)

        if self.patches:
            train_data = self.get_patches(train_data)
            train_masks = self.get_patches(train_masks)
            test_data = self.get_patches(test_data)
            test_masks = self.get_patches(test_masks)

        train_labels = self.get_labels(train_masks)
        test_labels = self.get_labels(test_masks)
        if self.generate_normal_data:
            normal_train_data, normal_train_labels = self.get_normal_data(train_data, train_masks)
            self.normal_train_data = normal_train_data
            self.normal_train_labels = normal_train_labels

        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_data = train_data
        self.train_masks = train_masks
        self.test_data = test_data
        self.test_masks = test_masks

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
        return process(data, per_image=self.per_image)

    def get_normal_data(self, data, masks):
        normal_data = data[np.invert(np.any(masks, axis=(1, 2, 3)))]
        labels = ['normal'] * len(normal_data)
        return normal_data, labels

    def get_labels(self, masks):
        labels = np.empty(len(masks), dtype='object')
        labels[np.any(masks, axis=(1, 2, 3))] = self.anomaly_class
        labels[np.invert(np.any(masks, axis=(1, 2, 3)))] = 'normal'
        return labels

    def get_clipped(self, data, masks):
        _max = np.mean(data[np.invert(masks)]) + np.abs(self.std_plus) * np.std(data[np.invert(masks)])
        _min = np.absolute(np.mean(data[np.invert(masks)]) - np.abs(self.std_minus) * np.std(data[np.invert(masks)]))
        return np.clip(data, _min, _max)

    def flag_data(self, data):
        train_masks = flag_data(data, self.data_name, self.rfi_threshold)
        return np.expand_dims(train_masks, axis=-1)

    def get_patches(self, data_or_masks):
        p_size = (1, self.patch_x, self.patch_y, 1)
        s_size = (1, self.patch_stride_x, self.patch_stride_y, 1)
        rate = (1, 1, 1, 1)
        if data_or_masks.dtype == np.dtype('bool'):
            return get_patches(data_or_masks.astype('int'), p_size, s_size, rate, 'VALID').astype('bool')
        else:
            return get_patches(data_or_masks, p_size, s_size, rate, 'VALID')

    def reconstruct(self, patches, labels=None):
        return reconstruct(patches, self.raw_input_shape, self.patch_x, self.patch_y, self.anomaly_class, labels)

    def reconstruct_latent_patches(self, patches, labels=None):
        reconstruct_latent_patches(patches, self.raw_input_shape, self.patch_x, self.patch_y, self.anomaly_class,
                                   labels)

    def load(self):
        self.load_raw_data()
        self.preprocess()

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
            if prop == 'ae_train_data' and self.normal_train_data is None:
                return False
            if prop == 'ae_train_labels' and self.normal_train_labels is None:
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
        return True

    def to_dict(self):
        return {
            'raw_input_shape': self.raw_input_shape,
            'num_patches': self.patches_per_image(),
            'num_training': self.raw_train_data.shape[0],
            'std_plus': self.std_plus,
            'std_minus': self.std_minus,
            'per_image': self.per_image,
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
