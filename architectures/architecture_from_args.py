from .generic_architecture import GenericArchitecture
#from utils.hardcoded_args import resolve_model_config_args
from models import load_model
from .dknn import DKNNArchitecture
from .ae_arch import AEArchitecture
from .dae_arch import DAEArchitecture

import tensorflow as tf


def load_architecture(args, get_model=True, checkpoint='None'):
    #args = resolve_model_config_args(args)
    if get_model:
        model = load_model(args)  # Tuple for DAE
    else:
        model = None

    # Select the Architecture instance, optimizer and loss
    if args.model_class == 'UNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model_class == 'AC_UNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model_class_class == 'AE-SSIM':
        def ssim_loss(x, x_hat):
            return 1 / 2 - tf.reduce_mean(tf.image.ssim(x, x_hat, max_val=1.0)) / 2
        arch = AEArchitecture(model, args, checkpoint)
        arch.loss_func = ssim_loss
        return arch
    if args.model_class == 'DKNN':
        return DKNNArchitecture(model, args, checkpoint)
    if args.model_class == 'AE':
        return AEArchitecture(model, args, checkpoint)
    if args.model_class == 'DAE':
        return DAEArchitecture(model, args, checkpoint)
    if args.model_class == 'RNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model_class == 'CNN_RFI_SUN':
        return GenericArchitecture(model, args, checkpoint)
    if args.model_class == 'RFI_NET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model_class == 'DSC_DUAL_RESUNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model_class == 'DSC_MONO_RESUNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model_class == 'ASPP_UNET':
        return GenericArchitecture(model, args, checkpoint)
    return None
