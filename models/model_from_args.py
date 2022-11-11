from .ac_unet import AC_UNET
from .ae_disc import (Encoder, Decoder, Autoencoder, Discriminator)
from .cnn_rfi_sun import CNN_RFI_SUN
from .rfi_net import RFI_NET
from .rnet import RNET
from .unet import UNET
from .dsc_dual_resunet import DSC_DUAL_RESUNET
from .dsc_mono_resunet import DSC_MONO_RESUNET
from .aspp_unet import ASPP_UNET
from .dknn import DKNN

from utils.hardcoded_args import resolve_model_config_args


def get_model_from_args(args):
    #args = resolve_model_config_args(args)
    if args.model == 'UNET':
        return UNET(args)
    if args.model == 'AC_UNET':
        return AC_UNET(args)
    if args.model == 'AE-SSIM':
        return Autoencoder(args)
    if args.model == 'DKNN':
        return DKNN(args)
    if args.model == 'AE':
        return Autoencoder(args)
    if args.model == 'DAE':
        ae = Autoencoder(args)
        discriminator = Discriminator(args)
        return ae, discriminator
    if args.model == 'RNET':
        return RNET(args)
    if args.model == 'CNN_RFI_SUN':
        return CNN_RFI_SUN(args)
    if args.model == 'RFI_NET':
        return RFI_NET(args)
    if args.model == 'DSC_DUAL_RESUNET':
        return DSC_DUAL_RESUNET(args)
    if args.model == 'DSC_MONO_RESUNET':
        return DSC_MONO_RESUNET(args)
    if args.model == 'DSC_MONO_RESUNET':
        return DSC_MONO_RESUNET(args)
    if args.model == 'ASPP_UNET':
        return ASPP_UNET(args)
    return None
