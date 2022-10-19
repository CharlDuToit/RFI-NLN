import numpy as np 
import tensorflow as tf
from data import *
from utils import args 
from architectures import *
from utils.hardcoded_args import *

def main():
    """
        Reads data and cmd arguments and trains models
    """
    #args.args.model_config = 'author'
    #args.args = resolve_model_config_args(args.args)
    #print(args.args)
    #return

    if args.args.data == 'HERA':
        data = load_hera(args.args)
    elif args.args.data == 'LOFAR':
        data = load_lofar(args.args)
    #elif args.args.data == 'HERA_PHASE':
    #    data = load_hera_phase(args.args)
    #elif args.args.data == 'HIDE':
    #    data = load_hide(args.args)

    #Add out_channels as arg?

    (unet_train_dataset, train_data, train_labels, train_masks, 
     ae_train_dataset, ae_train_data, ae_train_labels,
     test_data, test_labels, test_masks,test_masks_orig) = data

    print(" __________________________________ \n Save name {}".format(
                                               args.args.model_name))
    print(" __________________________________ \n")
    
    if args.args.model == 'UNET':
        train_unet(unet_train_dataset, train_data, train_labels, train_masks, test_data, test_labels, test_masks, test_masks_orig, args.args)

    if args.args.model == 'RNET':
        train_rnet(unet_train_dataset, train_data, train_labels, train_masks, test_data, test_labels, test_masks, test_masks_orig, args.args)

    if args.args.model == 'RFI_NET':
        train_rfi_net(unet_train_dataset, train_data, train_labels, train_masks, test_data, test_labels, test_masks, test_masks_orig, args.args)

    if args.args.model == 'CNN_RFI_SUN':
        train_cnn_rfi_sun(unet_train_dataset, train_data, train_labels, train_masks, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'DKNN':
        train_resnet(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'AE':
        train_ae(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'AE-SSIM':
        train_ae_ssim(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig, args.args)

    elif args.args.model == 'DAE':
        train_dae(ae_train_dataset, ae_train_data, ae_train_labels, test_data, test_labels, test_masks, test_masks_orig,args.args)


if __name__ == '__main__':
    #set_lofar_args(args.args)
    #set_hera_args(args.args)
    set_hera_args(args.args)
    args.args.model_config = 'common'
    args.args = resolve_model_config_args(args.args)
    main()
