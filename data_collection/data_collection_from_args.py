from utils.hardcoded_args import resolve_model_config_args
from .data_collection import DataCollection


def get_data_collection_from_args(args):
    args = resolve_model_config_args(args)

    if args.model in ['DAE', 'DKNN', 'AE', 'AE_SSIM']:
        gen_ae_data = True
    else:
        gen_ae_data = False

    if args.data == 'HERA':
        data_cllctn = DataCollection(args,
                                     std_plus=4,
                                     std_minus=1,
                                     flag_test_data=True,
                                     generate_normal_data=gen_ae_data,
                                     combine_min_std_plus=None)
        data_cllctn.load()
        return data_cllctn
    if args.model == 'LOFAR':
        data_cllctn = DataCollection(args,
                                     std_plus=95,
                                     std_minus=3,
                                     flag_test_data=False,
                                     generate_normal_data=gen_ae_data,
                                     combine_min_std_plus=5)
        data_cllctn.load()
        return data_cllctn
    return None
