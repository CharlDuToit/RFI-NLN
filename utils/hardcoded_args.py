def resolve_model_config_args(args):
    """
    Adjusts args according to args.model_config
    Parameters
    ----------
    args

    Returns
    -------
    args
    """
    if args.model_config == 'args':
        return args
    elif args.model_config == 'common':
        # models have common hyperparameters, while still trying to stay as close as possible to the author's
        # original implementation
        size = 32
        args.filters = 64
        args.height = 3
        args.level_blocks = 1
        args.patches = True
        args.patch_x = size
        args.patch_y = size
        args.patch_stride_x = size
        args.patch_stride_y = size
        args.crop = True
        args.crop_x = size
        args.crop_y = size
        args.input_shape = (args.patch_x, args.patch_y, args.input_shape[-1])
        args.dropout = 0.05

        if args.model == 'UNET':
            args.filters = 64
        if args.model == 'AC_UNET':
            pass
        if args.model == 'DKNN':
            pass
        if args.model == 'AE':
            pass
        if args.model == 'RNET':
            pass
        if args.model == 'CNN_RFI_SUN':
            pass
        if args.model == 'RFI_NET':
            pass
        if args.model == 'AE-SSIM':
            pass

    elif args.model_config == 'same':
        # all models have more or less the same number of FLOPS/parameters
        # the args required to achieve this must still be obtained experimentally
        # n_filters would be the easiest to adjust
        size = 32
        args.filters = 16
        args.height = 3
        args.level_blocks = 1
        args.patches = True
        args.patch_x = size
        args.patch_y = size
        args.patch_stride_x = size
        args.patch_stride_y = size
        args.input_shape = (args.patch_x, args.patch_y, args.input_shape[-1])

        if args.model == 'UNET':
            pass
        if args.model == 'AC_UNET':
            pass
        if args.model == 'DKNN':
            pass
        if args.model == 'AE':
            pass
        if args.model == 'RNET':
            pass
        if args.model == 'CNN_RFI_SUN':
            pass
        if args.model == 'RFI_NET':
            pass
        if args.model == 'AE-SSIM':
            pass

    elif args.model_config == 'author':
        # all models are the same as the authors implementation, specifically the input_size, n_filters and height
        size = 32
        args.filters = 16
        args.height = 3
        args.level_blocks = 1
        args.patches = True
        args.patch_x = size
        args.patch_y = size
        args.patch_stride_x = size
        args.patch_stride_y = size
        args.input_shape = (args.patch_x, args.patch_y, args.input_shape[-1])

        if args.model == 'UNET':
            args.dropout = 0.5
            args.filters = 64
            args.patch_x = 276
            args.patch_y = 600
            args.patch_stride_x = 276
            args.patch_stride_y = 600
            args.input_shape = (args.patch_x, args.patch_y, args.input_shape[-1])
        if args.model == 'AC_UNET':
            args.dilation_rate = 7
            args.filters = 64
            size = 100
            args.patch_x = size
            args.patch_y = size
            args.patch_stride_x = size
            args.patch_stride_y = size
            args.input_shape = (args.patch_x, args.patch_y, args.input_shape[-1])
        if args.model == 'DKNN':
            pass
        if args.model == 'AE':
            pass
        if args.model == 'RNET':
            pass
        if args.model == 'CNN_RFI_SUN':
            pass
        if args.model == 'RFI_NET':
            pass
        if args.model == 'AE-SSIM':
            pass

    elif args.model_config == 'custom':
        # custom hyper-parameters per model
        # use this for testing and debugging

        size = 32
        args.filters = 16
        args.height = 3
        args.level_blocks = 1
        args.patches = True
        args.patch_x = size
        args.patch_y = size
        args.patch_stride_x = size
        args.patch_stride_y = size
        args.input_shape = (args.patch_x, args.patch_y, args.input_shape[-1])


        if args.model == 'UNET':
            pass
        if args.model == 'AC_UNET':
            pass
        if args.model == 'DKNN':
            pass
        if args.model == 'AE':
            pass
        if args.model == 'RNET':
            pass
        if args.model == 'CNN_RFI_SUN':
            pass
        if args.model == 'RFI_NET':
            pass
        if args.model == 'AE-SSIM':
            pass

    return args


def set_hera_args(args):
    """Exactly the same as the arguments provided by run_hera.sh unless stated otherwise"""
    from pathlib import Path
    data_path = Path('/home')
    data_path = data_path / 'ee487519' / 'DatasetsAndConfig'/ 'Given' / '43_Mesarcik_2022'
    args.data_path = str(data_path) #Charl
    args.epochs = 100 #Charl - script: 100
    args.limit = None #Charl - script: None
    args.model = 'UNET'
    # 'UNET','AE', 'DAE', 'DKNN','RNET', 'RFI_NET', 'AE-SSIM'
    args.data = 'HERA'
    args.anomaly_class = 'rfi'
    args.anomaly_type = 'MISO'
    args.latent_dim = 8
    args.alphas = [1.0]
    args.percentage_anomaly = 0
    args.neighbors = [20]
    args.radius = [0.1, 0.5, 1, 2 ,5, 10]
    args.algorithm = 'knn'
    args.seed = 'aseed'
    args.debug = '0'
    args.rotate = False
    args.crop = False
    args.crop_x = 32
    args.crop_y = 32
    args.patches = True
    args.patch_x = 32
    args.patch_y = 32
    args.patch_stride_x = 32
    args.patch_stride_y = 32
    args.rfi = None
    args.rfi_threshold = '10'
    args.clip = None
    #args.args.model_name = f'Charl-first-{args.args.model}-seed-{args.args.seed}' # script: provided by coolname
    args.input_shape = (32,32,1)
    return args


def set_lofar_args(args):
    """Exactly the same as the arguments provided by run_lofar.sh unless stated otherwise"""
    from pathlib import Path
    data_path = Path('/home')
    data_path = data_path / 'ee487519' / 'DatasetsAndConfig'/ 'Given' / '43_Mesarcik_2022'
    args.data_path = str(data_path) #Charl
    args.epochs = 1 #Charl - script: 100
    args.limit = None #Charl - script: None
    args.model = 'RNET'
    # 'UNET','AE', 'DAE', 'DKNN','RNET', 'RFI_NET', 'AE-SSIM'
    args.data = 'LOFAR'
    args.lofar_subset = 'L629174'
    args.anomaly_class = 'rfi'
    args.anomaly_type = 'MISO'
    args.latent_dim = 32
    args.alphas = [1.0]
    args.percentage_anomaly = 0
    args.neighbors = [20]
    args.radius = [0.1, 0.5, 1, 2 ,5, 10]
    args.algorithm = 'knn'
    args.seed = 'aseed'
    args.debug = '0'
    args.rotate = False
    args.crop = False
    args.crop_x = 32
    args.crop_y = 32
    args.patches = True
    args.patch_x = 32
    args.patch_y = 32
    args.patch_stride_x = 32
    args.patch_stride_y = 32
    args.rfi = None
    args.rfi_threshold = None
    args.clip = None
    #args.args.model_name = f'Charl-first-{args.args.model}-seed-{args.args.seed}' # script: provided by coolname
    args.input_shape = (32,32,1)
    return args

