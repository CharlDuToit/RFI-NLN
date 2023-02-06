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


def load_model(model_class, **kwargs):
    if model_class == 'UNET':
        return UNET(**kwargs)
    if model_class == 'AC_UNET':
        return AC_UNET(**kwargs)
    if model_class  == 'AE-SSIM':
        return Autoencoder(**kwargs)
    if model_class  == 'DKNN':
        return DKNN(**kwargs)
    if model_class == 'AE':
        return Autoencoder(**kwargs)
    if model_class == 'DAE':
        ae = Autoencoder(**kwargs)
        discriminator = Discriminator(**kwargs)
        return ae, discriminator
    if model_class == 'RNET':
        return RNET(**kwargs)
    if model_class  == 'CNN_RFI_SUN':
        return CNN_RFI_SUN(**kwargs)
    if model_class  == 'RFI_NET':
        return RFI_NET(**kwargs)
    if model_class  == 'DSC_DUAL_RESUNET':
        return DSC_DUAL_RESUNET(**kwargs)
    if model_class  == 'DSC_MONO_RESUNET':
        return DSC_MONO_RESUNET(**kwargs)
    if model_class  == 'DSC_MONO_RESUNET':
        return DSC_MONO_RESUNET(**kwargs)
    if model_class  == 'ASPP_UNET':
        return ASPP_UNET(**kwargs)
    return None
