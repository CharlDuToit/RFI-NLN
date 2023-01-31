from models import *
from utils import args
#from utils.args import args
from utils.profiling import *
from utils.hardcoded_args import *

def save_all_summaries():
    args.args = set_hera_args(args.args)
    folder = 'model_summaries/all'
    #for mc in ['common', 'author', 'custom', 'same']:
    #for mc in ['common', 'full']:
    for mc in ['common']:
        args.args.model_config = mc
        mc_folder = folder + '/' + mc
        # , 'CNN_RFI_SUN'
        for mdl in ['UNET', 'AC_UNET', 'RNET', 'RFI_NET', 'DSC_DUAL_RESUNET', 'DSC_MONO_RESUNET', 'ASPP_UNET']:
        #for mdl in ['UNET']:
            args.args.model = mdl
            #args.args = resolve_model_config_args(args.args)
            model = get_model_from_args(args.args)
            save_summary_to_folder(model, mc_folder, args.args)


if __name__ == '__main__':
    save_all_summaries()
