from models import *
from utils import args
#from utils.args import args
from utils.profiling import *
from utils.hardcoded_args import *


def model_from_args():
    args.args = resolve_model_config_args(args.args)
    if args.args.model == 'UNET':
        return UNET(args.args)
    if args.args.model == 'AC_UNET':
        return AC_UNET(args.args)
    if args.args.model == 'DKNN':
        pass
    if args.args.model == 'AE':
        pass
    if args.args.model == 'RNET':
        return RNET(args.args)
    if args.args.model == 'CNN_RFI_SUN':
        return CNN_RFI_SUN(args.args)
    if args.args.model == 'RFI_NET':
        return RFI_NET(args.args)
    if args.args.model == 'AE-SSIM':
        pass
    if args.args.model == 'DSC_DUAL_RESUNET':
        return DSC_DUAL_RESUNET(args.args)
    if args.args.model == 'DSC_MONO_RESUNET':
        return DSC_MONO_RESUNET(args.args)
    return None


def save_all_summaries():
    args.args = set_hera_args(args.args)
    folder = 'model_summaries/all'
    #for mc in ['common', 'author', 'custom', 'same']:
    for mc in ['common']:
        args.args.model_config = mc
        #for mdl in ['UNET', 'AC_UNET', 'RNET', 'RFI_NET', 'CNN_RFI_SUN', 'DSC_DUAL_RESUNET', 'DSC_MONO_RESUNET']:
        for mdl in ['DSC_MONO_RESUNET']:
            args.args.model = mdl
            model = get_model_from_args()
            save_summary_to_folder(model, folder, args.args)


if __name__ == '__main__':
    save_all_summaries()
