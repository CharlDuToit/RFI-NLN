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
from .rnet5 import RNET5
from .bounding_boxes import BoundingBox, BoundingBox_v3


def load_model(model_class, rfi_set, **kwargs):

    if rfi_set == 'separate' and model_class not in ('AE-SSIM', 'DAE', 'AE', 'DKNN'):
        return load_model(model_class, 'low', **kwargs), load_model(model_class, 'high', **kwargs)

    if model_class == 'BB':
        # return BoundingBox(**kwargs)
        return BoundingBox_v3(**kwargs)
    if model_class == 'UNET':
        return UNET(**kwargs)
    if model_class == 'AC_UNET':
        return AC_UNET(**kwargs)
    if model_class == 'AE-SSIM':
        return Autoencoder(**kwargs)
    if model_class == 'DKNN':
        return DKNN(**kwargs)
    if model_class == 'AE':
        return Autoencoder(**kwargs)
    if model_class == 'DAE':
        ae = Autoencoder(**kwargs)
        discriminator = Discriminator(**kwargs)
        return ae, discriminator
    if model_class == 'RNET':
        return RNET(**kwargs)
    if model_class == 'RNET5':
        return RNET5(**kwargs)
    if model_class == 'CNN_RFI_SUN':
        return CNN_RFI_SUN(**kwargs)
    if model_class == 'RFI_NET':
        return RFI_NET(**kwargs)
    if model_class == 'DSC_DUAL_RESUNET':
        return DSC_DUAL_RESUNET(**kwargs)
    if model_class == 'DSC_MONO_RESUNET':
        return DSC_MONO_RESUNET(**kwargs)
    if model_class == 'DSC_MONO_RESUNET':
        return DSC_MONO_RESUNET(**kwargs)
    if model_class == 'ASPP_UNET':
        return ASPP_UNET(**kwargs)
    return None
