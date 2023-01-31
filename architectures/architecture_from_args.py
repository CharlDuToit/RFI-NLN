from .generic_architecture import GenericArchitecture
#from utils.hardcoded_args import resolve_model_config_args
from models import get_model_from_args
from .dknn import DKNNArchitecture
from .ae_arch import AEArchitecture
from .dae_arch import DAEArchitecture

import tensorflow as tf


def get_architecture_from_args(args, get_model=True, checkpoint='None'):
    #args = resolve_model_config_args(args)
    if get_model:
        model = get_model_from_args(args)  # Tuple for DAE
    else:
        model = None

    # Select the Architecture instance, optimizer and loss
    if args.model == 'UNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model == 'AC_UNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model == 'AE-SSIM':
        def ssim_loss(x, x_hat):
            return 1 / 2 - tf.reduce_mean(tf.image.ssim(x, x_hat, max_val=1.0)) / 2
        arch = AEArchitecture(model, args, checkpoint)
        arch.loss_func = ssim_loss
        return arch
    if args.model == 'DKNN':
        return DKNNArchitecture(model, args, checkpoint)
    if args.model == 'AE':
        return AEArchitecture(model, args, checkpoint)
    if args.model == 'DAE':
        return DAEArchitecture(model, args, checkpoint)
    if args.model == 'RNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model == 'CNN_RFI_SUN':
        return GenericArchitecture(model, args, checkpoint)
    if args.model == 'RFI_NET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model == 'DSC_DUAL_RESUNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model == 'DSC_MONO_RESUNET':
        return GenericArchitecture(model, args, checkpoint)
    if args.model == 'ASPP_UNET':
        return GenericArchitecture(model, args, checkpoint)
    return None
