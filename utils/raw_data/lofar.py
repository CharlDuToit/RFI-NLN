import os
import errno
import pickle
import numpy as np
import tensorflow as tf

def _random_crop(image,mask,size,batch_size=512):
    output_images = np.empty((len(image), size[0], size[1], 1)).astype('float32')
    output_masks = np.empty((len(mask), size[0], size[1], 1)).astype('bool')
    strt, fnnsh = 0, batch_size
    for i in range(0,len(image),batch_size):
        stacked_image = np.stack([image[strt:fnnsh,...],
                                  mask[strt:fnnsh,...].astype('float32')],axis=0)
        cropped_image = tf.image.random_crop(stacked_image, size=[2,len(stacked_image[0]), size[0], size[1], 1])
        output_images[strt:fnnsh,...]  = cropped_image[0].numpy()
        output_masks[strt:fnnsh,...]  = cropped_image[1].numpy().astype('bool')
        strt=fnnsh
        fnnsh+=batch_size
    return output_images, output_masks

#def get_lofar_data(args):
def get_lofar_data(data_path, lofar_subset='full'):
    """"
        Walks through LOFAR dataset and returns sampled and cropped data 
        
        args (Namespace): args from utils.cmd_args 
        num_baselines (int): number of baselines to sample 
    """
    full_file = file = 'LOFAR_Full_RFI_dataset.pkl'
    if lofar_subset is None or lofar_subset == 'full':
        file = full_file
    if lofar_subset == 'L629174':
        file = 'L629174_RFI_dataset.pkl'
    if lofar_subset == 'L631961':
        file = 'L631961_RFI_dataset.pkl'

    file = os.path.join(data_path, file)
    if os.path.exists(file):
        # print(file + ' Loading')
        with open(file, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)
