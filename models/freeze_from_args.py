from .ac_unet import freeze_AC_UNET
from .rfi_net import freeze_RFI_NET
from .rnet import freeze_RNET
from .unet import freeze_UNET
from .dsc_dual_resunet import freeze_DSC_DUAL_RESUNET
from .dsc_mono_resunet import freeze_DSC_MONO_RESUNET
from .aspp_unet import freeze_ASPP_UNET
from .rnet5 import freeze_RNET5


def freeze_top_layers(model, model_class, **kwargs):
    """Hardcoded layers to freeze ASSUMING dropout layers exist"""
    if model_class == 'UNET':
        return freeze_UNET(model)
    if model_class == 'AC_UNET':
        return freeze_AC_UNET(model)
    if model_class == 'RNET':
        return freeze_RNET(model)
    if model_class == 'RNET5':
        return freeze_RNET5(model)
    if model_class == 'RFI_NET':
        return freeze_RFI_NET(model)
    if model_class == 'DSC_DUAL_RESUNET':
        return freeze_DSC_DUAL_RESUNET(model)
    if model_class == 'DSC_MONO_RESUNET':
        return freeze_DSC_MONO_RESUNET(model)
    if model_class == 'ASPP_UNET':
        return freeze_ASPP_UNET(model)
    return None

# kwargs = dict(input_shape=(64,64,1), dilation_rate=3, height=4, filters=16, dropout = 0.1, kernel_regularizer='l2', level_blocks=1, final_activation='sigmoid', activation='relu', bn_first=False)
