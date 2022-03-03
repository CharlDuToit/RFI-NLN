import tensorflow as tf
import numpy as np
import copy
import random
from tqdm import tqdm
from model_config import BUFFER_SIZE,BATCH_SIZE
from sklearn.model_selection import train_test_split
from utils.flagging import flag_data
from utils.data import (get_lofar_data, 
                        get_hera_data,
                        get_hide_data,
                        process,
                        get_patches)

def load_hide(args):            
    """
        Load data from hide 
    """

    (train_data, train_masks, 
            test_data, test_masks) = get_hide_data(args)

    test_data = np.clip(np.fabs(test_data), 0, 500)
    test_data -= np.amin(test_data)
    test_data = process(test_data, per_image=False)

    #mi, ma = np.min(data), np.max(data)
    #output = (data - mi)/(ma -mi)
    #output = output.astype('float32')

    train_data = np.clip(np.fabs(train_data), 0, 500)
    train_data -= np.amin(train_data)
    train_data = process(train_data, per_image=False)

    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        test_indx = np.random.permutation(len(test_data))[:args.limit]

        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]
        #test_data   = test_data  [test_indx]
        #test_masks  = test_masks [test_indx]

    test_masks_orig = copy.deepcopy(test_masks)
    if args.rfi_threshold is not None:
        test_masks = flag_data(test_data,args)
        train_masks = flag_data(train_data,args)
        test_masks = np.expand_dims(test_masks,axis=-1) 
        train_masks = np.expand_dims(train_masks,axis=-1) 

    if args.patches:
        p_size = (1,args.patch_x, args.patch_y, 1)
        s_size = (1,args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1,1,1,1)

        train_data = get_patches(train_data, None, p_size,s_size,rate,'VALID')
        test_data = get_patches(test_data, None, p_size,s_size,rate,'VALID')
        train_masks = get_patches(train_masks, None, p_size,s_size,rate,'VALID').astype(np.bool)
        test_masks= get_patches(test_masks.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)
        test_masks_orig = get_patches(test_masks_orig.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1,2,3))] = args.anomaly_class
        train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1,2,3))] = args.anomaly_class
        test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))] = 'normal'

        ae_train_data  = train_data[np.invert(np.any(train_masks, axis=(1,2,3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))]

    ae_train_data = ae_train_data.astype(np.float16) 
    train_data = train_data.astype(np.float16) 
    test_data = test_data.astype(np.float16) 

    unet_train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)
    ae_train_dataset = tf.data.Dataset.from_tensor_slices(ae_train_data).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)


    return (unet_train_dataset,
            train_data, 
            train_labels, 
            train_masks, 
            ae_train_dataset,
            ae_train_data, 
            ae_train_labels, 
            test_data, 
            test_labels, 
            test_masks,
            test_masks_orig)


def load_hera(args):
    """
        Load data from hera

    """
    (train_data, test_data, 
         train_labels, test_labels, 
         train_masks, test_masks) = get_hera_data(args)


    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]
        train_labels = train_labels[train_indx]

        #test_indx = np.random.permutation(len(test_data))[:args.limit]
        #test_data = test_data[test_indx]
        #test_masks = test_masks[test_indx]

    if args.rfi is not None:
        _mask = np.array([args.rfi not in label for label in train_labels])
        train_data = train_data[_mask]
        train_masks = train_masks[_mask]
        train_labels= train_labels[_mask]

        test_data, test_labels, test_masks, _ =  np.load('/home/mmesarcik/data/HERA/HERA_{}.pkl'.format(args.rfi), 
                                                         allow_pickle=True)
        test_masks = np.expand_dims(test_masks, axis=-1)
    test_masks_orig = copy.deepcopy(test_masks)
#    if args.rfi_threshold is not None:
#        test_masks = flag_data(test_data,args)
#        train_masks = flag_data(train_data,args)
#        test_masks = np.expand_dims(test_masks,axis=-1) 
#        train_masks = np.expand_dims(train_masks,axis=-1) 

    test_data[test_data==0] = 0.001 # to make log normalisation happy
    test_data = np.nan_to_num(np.log(test_data),nan=0)

    train_data[train_data==0] = 0.001 # to make log normalisation happy
    train_data = np.nan_to_num(np.log(train_data),nan=0)

    test_data =  process(test_data, per_image=False).astype(np.float16)
    train_data = process(train_data, per_image=False).astype(np.float16)


    if args.patches:
        p_size = (1,args.patch_x, args.patch_y, 1)
        s_size = (1,args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1,1,1,1)

        train_data = get_patches(train_data, None, p_size,s_size,rate,'VALID')
        test_data = get_patches(test_data, None, p_size,s_size,rate,'VALID')
        train_masks = get_patches(train_masks, None, p_size,s_size,rate,'VALID').astype(np.bool)
        test_masks= get_patches(test_masks.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)
        test_masks_orig = get_patches(test_masks_orig.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1,2,3))] = args.anomaly_class
        train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1,2,3))] = args.anomaly_class
        test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))] = 'normal'


        ae_train_data  = train_data[np.invert(np.any(train_masks, axis=(1,2,3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))]


    ae_train_data = ae_train_data.astype(np.float16) 
    train_data = train_data.astype(np.float16) 
    test_data = test_data.astype(np.float16) 

    unet_train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    ae_train_dataset = tf.data.Dataset.from_tensor_slices(ae_train_data).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    return (unet_train_dataset,
            train_data, 
            train_labels,
            train_masks, 
            ae_train_dataset,
            ae_train_data, 
            ae_train_labels,
            test_data, 
            test_labels, 
            test_masks,
            test_masks_orig)


def load_lofar(args):            
    """
        Load data from lofar 

    """

    train_data, train_masks, test_data, test_masks = get_lofar_data( args)

    test_data[test_data==0] = 0.001 # to make log normalisation happy
    test_data = np.nan_to_num(np.log(test_data),nan=0)
    test_data = process(test_data, per_image=False)

    train_data[train_data==0] = 0.001 # to make log normalisation happy
    train_data = np.nan_to_num(np.log(train_data),nan=0)
    train_data = process(train_data, per_image=False)

    if args.limit is not None:
        train_indx = np.random.permutation(len(train_data))[:args.limit]
        test_indx = np.random.permutation(len(test_data))[:args.limit]

        train_data  = train_data [train_indx]
        train_masks = train_masks[train_indx]
        #test_data   = test_data  [test_indx]
        #test_masks  = test_masks [test_indx]

    if args.patches:
        p_size = (1,args.patch_x, args.patch_y, 1)
        s_size = (1,args.patch_stride_x, args.patch_stride_y, 1)
        rate = (1,1,1,1)

        train_data = get_patches(train_data, None, p_size,s_size,rate,'VALID')
        test_data = get_patches(test_data, None, p_size,s_size,rate,'VALID')
        train_masks = get_patches(train_masks, None, p_size,s_size,rate,'VALID').astype(np.bool)
        test_masks= get_patches(test_masks.astype('int') , None, p_size,s_size,rate,'VALID').astype(np.bool)

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1,2,3))] = args.anomaly_class
        train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1,2,3))] = args.anomaly_class
        test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))] = 'normal'
        if len(train_data) > 1000000:
            ind = np.random.choice(len(train_data), 
                                   1000000, 
                                   replace=False)
            train_data = train_data[ind]
            train_labels = train_labels[ind]
            train_masks = train_masks[ind]

        
        if len(test_data) > 1000000:
            ind= np.random.choice(len(test_data), 
                                   1000000, 
                                   replace=False)
            test_data = test_data[ind]
            test_masks = test_masks[ind]
            test_labels = test_labels[ind]


        ae_train_data  = train_data[np.invert(np.any(train_masks, axis=(1,2,3)))]
        ae_train_labels = train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))]

        #test_data  =  test_data[np.invert(np.any(test_masks, axis=(1,2,3)))]
        #test_labels = test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))]
        #test_masks = test_masks[np.invert(np.any(test_masks, axis=(1,2,3)))]

    ae_train_data = ae_train_data.astype(np.float16) 
    train_data = train_data.astype(np.float16) 
    test_data = test_data.astype(np.float16) 

    unet_train_dataset = tf.data.Dataset.from_tensor_slices(train_data).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)
    ae_train_dataset = tf.data.Dataset.from_tensor_slices(ae_train_data).shuffle(BUFFER_SIZE,seed=42).batch(BATCH_SIZE)


    return (unet_train_dataset,
            train_data, 
            train_labels, 
            train_masks, 
            ae_train_dataset,
            ae_train_data, 
            ae_train_labels, 
            test_data, 
            test_labels, 
            test_masks,
            test_masks)

