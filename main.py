import numpy as np 
import tensorflow as tf
from data import *
from utils import args 
from architectures import *
from utils.hardcoded_args import *
#from utils.data import DataCollection
from data_collection import get_data_collection_from_args
from architectures import get_architecture_from_args

def main():
    """
        Reads data and cmd arguments and trains models
    """
    # Add out_channels as arg?

    new_way = True
    if new_way:
        data_collection = get_data_collection_from_args(args.args)
        arch = get_architecture_from_args(args.args)
        print("__________________________________ \n Save name {}".format(args.args.model_name))
        print("__________________________________ ")
        arch.train(data_collection)
        print("__________________________________ \n Evaluating data")
        arch.evaluate_and_save(data_collection)
    else:
        if args.args.data == 'HERA':
            #data_collection = DataCollection(args.args, std_plus=4, std_minus=1, flag_test_data=True, generate_ae_data=True)
            #data_collection.load()
            data = load_hera(args.args)
        elif args.args.data == 'LOFAR':
           # data_collection = DataCollection(args.args, std_plus=95, std_minus=3, flag_test_data=False, generate_ae_data=True)
            #data_collection.load()
            data = load_lofar(args.args)
        # elif args.args.data == 'HERA_PHASE':
        #    data = load_hera_phase(args.args)
        # elif args.args.data == 'HIDE':
        #    data = load_hide(args.args)

        (unet_train_dataset, train_data, train_labels, train_masks,
         ae_train_dataset, ae_train_data, ae_train_labels,
         test_data, test_labels, test_masks,test_masks_orig) = data

        # print(f"train_data: {np.allclose(train_data, data_collection.train_data)}")
        # print(f"train_masks: {np.allclose(train_masks, data_collection.train_masks)}")
        # print(f"train_labels: {train_labels == data_collection.train_labels}")
        # print(f"test_data: {np.allclose(test_data, data_collection.test_data)}")
        # print(f"test_masks: {np.allclose(test_masks, data_collection.test_masks)}")
        # print(f"test_labels: {test_labels == data_collection.test_labels}")
        # print(f"test_masks_orig: {np.allclose(test_masks_orig, data_collection.test_masks_orig)}")
        # print(f"ae_train_data: {np.allclose(ae_train_data, data_collection.ae_train_data)}")
        # print(f"ae_train_labels: {ae_train_labels == data_collection.ae_train_labels}")
        # return

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


def tiny_args(args_):
    args_ = set_hera_args(args_)
    args_.model_config = 'tiny'
    args_.optimal_alpha = True
    args_.optimal_neighbours = True
    args_.model = 'ASPP_UNET'
    args_ = resolve_model_config_args(args_)
    args_.epochs = 2
    args_.limit = 8 # full size images
    args_.rfi_threshold = None
    args_.seed = 'bleh'
    return args_


if __name__ == '__main__':
    args.args = tiny_args(args.args)
    main()


