#from utils.hardcoded_args import resolve_model_config_args
from .data_collection import DataCollection


def get_data_collection_from_args(args, generate_normal_data=False):
    #args = resolve_model_config_args(args)

    if args.model in ['DAE', 'DKNN', 'AE', 'AE_SSIM']:
        generate_normal_data = True
    else:
        generate_normal_data = generate_normal_data

    if args.optimal_neighbours:
        args.neighbours = [20]

    if args.data in ['HERA', 'HERA_PHASE']:
        if args.optimal_alpha:
            args.alphas = [0.1]
        data_cllctn = DataCollection(args,
                                     std_plus=4,
                                     std_minus=1,
                                     flag_test_data=True,
                                     clip_per_image=True,
                                     scale_per_image=True,
                                     generate_normal_data=generate_normal_data,
                                     combine_min_std_plus=None)
        #data_cllctn.load()
        return data_cllctn
    if args.data == 'LOFAR':
        if args.optimal_alpha:
            args.alphas = [0.66]
        data_cllctn = DataCollection(args,
                                     std_plus=95,
                                     std_minus=3,
                                     flag_test_data=False,
                                     clip_per_image=True,
                                     generate_normal_data=generate_normal_data,
                                     combine_min_std_plus=5,
                                     scale_per_image=True)
        #data_cllctn.load()
        return data_cllctn

    if args.data in ['ASTRON_0']:
        if args.optimal_alpha:
            args.alphas = [0.66] # not tested yet
        data_cllctn = DataCollection(args,
                                     std_plus=95,  # 95
                                     std_minus=3,  # 3
                                     flag_test_data=False,
                                     generate_normal_data=generate_normal_data,
                                     combine_min_std_plus=5,
                                     clip_per_image=True,
                                     hyp_split=0.0,
                                     scale_per_image=True)
        #data_cllctn.load()
        return data_cllctn

    if args.data in ['ant_fft_000_094_t4032_f4096',
                     'ant_fft_000_094_t4096_f4096',
                     'ant_fft_000_094_t8128_f8192',
                     'ant_fft_000_094_t12160_f16384']:
        if args.optimal_alpha:
            args.alphas = [0.66] # not tested yet
        data_cllctn = DataCollection(args,
                                     std_plus=-0.012, #0.148 we start to clip actual data in mid
                                     std_minus=0.016, # min val is 0.157 stds away from mean
                                     #std_plus=20,
                                     #std_minus=1,
                                     flag_test_data=False,
                                     generate_normal_data=generate_normal_data,
                                     combine_min_std_plus=5,
                                     clip_per_image=True,
                                     hyp_split=0.0,
                                     scale_per_image=True)
        #data_cllctn.load()
        return data_cllctn
    return None
