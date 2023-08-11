import argparse
import os
#from utils.data import sizes
from coolname import generate_slug as new_name
#from .hardcoded_args import resolve_model_config_args
import numpy as np
from .results import load_csv

"""
    Pretty self explanatory, gets arguments for training and adds them to config
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Train generative anomaly detection models')

parser.add_argument('-model_class', metavar='-m', type=str, default='AE',
                    choices={'BB_NET', 'AC_UNET', 'UNET', 'AE', 'DAE', 'DKNN', 'RNET', 'CNN_RFI_SUN', 'RFI_NET', 'AE-SSIM', 'DSC_DUAL_RESUNET', 'DSC_MONO_RESUNET', 'ASPP_UNET', 'RNET5'},
                    help='Which model to train and evaluate')
parser.add_argument('-task', metavar='-t', type=str, default='train',
                    choices={'train', 'eval', 'eval_test', 'infer', 'transfer_train'},
                    help='Task to perform')
parser.add_argument('-freeze_top_layers', metavar='-ftl', type=str2bool, default=False,
                    help='Freeze top 2 layers?')
parser.add_argument('-cmd_args', metavar='-cmdargs', type=str, nargs='+', default=['data_path', 'data_name'],
                    help='Cmd args to overwrite csv args when loading trained model')
parser.add_argument('-rfi_set', metavar='-rs', type=str, default='combined',
                    choices={'low', 'high', 'combined', 'separate'},
                    help='Train only on low or high rfi images')
parser.add_argument('-rfi_split_ratio', metavar='-rs', type=float, default=0.01,
                    help='RFI ratio to split data into low and high sets')
parser.add_argument('-model_name', metavar='-mn', type=str, default='None',
                    help='Name of already existing model with checkpoint from which to do inference')
parser.add_argument('-parent_model_name', metavar='-pmn', type=str, default='None',
                    help='Name of already existing model with checkpoint from which to continue training')
parser.add_argument('-limit', metavar='-l', type=str, default='None',
                    help='Limit on the number of samples in training data ')
parser.add_argument('-anomaly_class', metavar='-ac', type=str, default='rfi',
                    help='The labels of the anomalous class')
parser.add_argument('-anomaly_type', metavar='-at', type=str, default='MISO',
                    choices={'MISO', 'SIMO'}, help='The anomaly scheme whether it is MISO or SIMO')
parser.add_argument('-percentage_anomaly', metavar='-p', type=float, default=0,
                    help='The percentage of anomalies in the training set')
parser.add_argument('-epochs', metavar='-e', type=int, default=150,
                    help='The number of epochs for training')
parser.add_argument('-latent_dim', metavar='-ld', type=int, default=2,
                    help='The latent dimension size of the AE based models')
parser.add_argument('-alphas', metavar='-alph', type=float, nargs='+', default=[1.0],
                    help='The maximum number of neighbours for latent reconstruction')
parser.add_argument('-neighbours', metavar='-n', type=int, nargs='+', default=[2, 4, 5, 6, 7, 8, 9],
                    help='The maximum number of neighbours for latent reconstruction')
parser.add_argument('-radius', metavar='-r', type=float, nargs='+', default=[0.1, 0.5, 1, 2, 5, 10],
                    help='The radius of the unit circle for finding neighbours in frNN')
parser.add_argument('-algorithm', metavar='-nn', type=str, choices={"frnn", "knn"},
                    default='knn', help='The algorithm for calculating neighbours')
parser.add_argument('-data_name', metavar='-d', type=str, default='HERA',
                    #choices={'HERA', 'HERA_PHASE', 'LOFAR', 'MNIST', 'CIFAR10', 'FASHION_MNIST', 'ASTRON_0'},
                    help='The dataset for training and testing the model on')
parser.add_argument('-data_path', metavar='-mp', type=str, default='./data',
                    help='Path to MVTecAD training data')
parser.add_argument('-seed', metavar='-s', type=str,
                    help='The random seed used for naming output files')
parser.add_argument('-debug', metavar='-de', type=str, default='0',
                    choices={'0', '1', '2', '3'}, help='TF debug level')
parser.add_argument('-log', metavar='-log', type=str2bool, default=True,
                    help='Take log of data?')
parser.add_argument('-rescale', metavar='-rescl', type=str2bool, default=True,
                    help='Rescale data?')
parser.add_argument('-bn_first', metavar='-bn_first', type=str2bool, default=False,
                    help='First layer BN?')
parser.add_argument('-rotate', metavar='-rot', type=str2bool, default=False,
                    help='Train on rotated augmentations?')
# CROP IS NEVER ACTUALLY USED
parser.add_argument('-crop', metavar='-cr', type=str2bool, default=False,
                    help='Train on crops?')
parser.add_argument('-crop_x', metavar='-cx', type=int,
                    help='x-dimension of crop')
parser.add_argument('-crop_y', metavar='-cy', type=int,
                    help='y-dimension of crop')
# CROP IS NEVER ACTUALLY USED
parser.add_argument('-train_with_test', metavar='-twt', type=str2bool, default=False,
                    help='Train on test set? limit will determine how many')
parser.add_argument('-patches', metavar='-ptch', type=str2bool, default=False,
                    help='Train on patches?')
parser.add_argument('-patch_x', metavar='-px', type=int, default=-1,
                    help='x-dimension of patchsize ')
parser.add_argument('-patch_y', metavar='-py', type=int,
                    help='y-dimension of patchsize ')
parser.add_argument('-patch_stride_x', metavar='-psx', type=int,
                    help='x-dimension of strides of patches')
parser.add_argument('-patch_stride_y', metavar='-psy', type=int,
                    help='y-dimension of strides of patches')
parser.add_argument('-flag_test_data', metavar='-ftd', type=str2bool, default=False,
                    help='Flag test data if rfi_threshold is given?')
parser.add_argument('-rfi', metavar='-rfi', type=str, default=None,
                    help='HERA RFI label to exclude from training')
parser.add_argument('-rfi_threshold', metavar='-rfi_threshold', type=str, default=None,
                    help='AOFlagger base threshold')
parser.add_argument('-lofar_subset', metavar='-lofar_subset', type=str, default='full',
                    choices={'L629174', 'L631961', 'full'}, help='LOFAR subset to use for training')
parser.add_argument('-scale_per_image', metavar='-spi', type=str2bool, default=True,
                    help='Normalize per image or for entire dataset')
parser.add_argument('-clip_per_image', metavar='-cpi', type=str2bool, default=True,
                    help='Clip per image or for entire dataset')
parser.add_argument('-clipper', metavar='-clip', type=str, default='None',
                    choices={'None', 'std', 'dyn_std', 'known', 'perc'},
                    help='Clip strategy to use')
parser.add_argument('-std_max', metavar='-sma', type=float, default=4,
                    help='Number of stds from mean to max clip')
parser.add_argument('-std_min', metavar='-smi', type=float, default=-1,
                    help='Number of stds from mean to min clip')
parser.add_argument('-perc_max', metavar='-pma', type=float, default=95,
                    help='Maximum percentile to clip')
parser.add_argument('-perc_min', metavar='-pmi', type=float, default=-5,
                    help='Minimum percentile to clip')
parser.add_argument('-filters', metavar='-nf', type=int, default=16,
                    help='base number of filters')
parser.add_argument('-height', metavar='-h', type=int, default=3,
                    help='Height (number of levels) of UNET type models')
parser.add_argument('-level_blocks', metavar='-nlb', type=int, default=1,
                    help='Number of blocks per level. A block can have one or more layers')
parser.add_argument('-model_config', metavar='-mcf', type=str, default='args',
                    choices={'args', 'common', 'same', 'custom', 'author', 'tiny', 'full'},
                    help='Which args to overwrite with default set.')
parser.add_argument('-dropout', metavar='-drop', type=float, default=0.0,
                    help='Dropout rate between 0 and 1')
parser.add_argument('-batch_size', metavar='-bas', type=int, default=64,
                    help='Batch size')
parser.add_argument('-buffer_size', metavar='-bus', type=int, default=2**10,
                    help='Buffer size for shuffling')
parser.add_argument('-optimal_alpha', metavar='-oalpha', type=str2bool, default=True,
                    help='Replace args.alphas with list of one alpha which is optimized for args.data for AE')
parser.add_argument('-optimal_neighbours', metavar='-oneighs', type=str2bool, default=True,
                    help='Replace args.neighbours with list of one neighbours which is optimized for AE')
parser.add_argument('-use_hyp_data', metavar='-hypd', type=str2bool, default=False,
                    help='Use reduced dataset for hyper-parameter search')
parser.add_argument('-lr', metavar='-lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('-lr_lin_decay', metavar='-lrlind', type=float, default=1.0,
                    help='Ratio of initial learning rate to linearly decay to at last epoch')
parser.add_argument('-loss', metavar='-ls', type=str, default='bce',
                    help='Loss function')
parser.add_argument('-kernel_regularizer', metavar='-kr', type=str, default='None',
                    choices={'l2', 'l1', 'None'},
                    help='Kernel regularizer')
# parser.add_argument('-kernel_initializer', metavar='-ki', type=str, default='glorot_uniform',
#                     choices={'glorot_uniform', 'he_normal', 'lecun_normal'},
#                     help='Kernel regularizer')
parser.add_argument('-input_channels', metavar='-ch', type=int, default=0,
                    help='Number of channels to extract from data. 0 will extract all channels')
parser.add_argument('-dilation_rate', metavar='-dilr', type=int, default=3,
                    help='Dilation rate for AC_UNET and ASPP_UNET')
parser.add_argument('-epoch_image_interval', metavar='-eii', type=int, default=5,
                    help='Epoch interval for which to save training images')
parser.add_argument('-images_per_epoch', metavar='-ipe', type=int, default=1,
                    help='How many images to save per epoch')
parser.add_argument('-early_stop', metavar='-es', type=int, default=20,
                    help='Early stopping for validation loss')
parser.add_argument('-shuffle_seed', metavar='-ss', type=str, default='None',
                    help='Seed to split data into training and validation. Default is random')
parser.add_argument('-val_split', metavar='-vs', type=float, default=0.2,
                    help='Validation split')
parser.add_argument('-final_activation', metavar='-fa', type=str, default='sigmoid',
                    help='Final activation function',
                    choices={'relu', 'selu', 'elu', 'gelu', 'sigmoid', 'None'}),
parser.add_argument('-activation', metavar='-a', type=str, default='relu',
                    help='Activation function for hidden layers',
                    choices={'relu', 'selu', 'elu', 'gelu', 'None'})
parser.add_argument('-output_path', metavar='-op', type=str, default='./outputs',
                    help='Path where models are saved')
parser.add_argument('-save_dataset', metavar='-sd', type=str2bool, default=False,
                    help='Save entire inferred dataset')
parser.add_argument('-shuffle_patches', metavar='-sp', type=str2bool, default=False,
                    help='Shuffle the patches? ')
parser.add_argument('-calc_train_val_auc', metavar='-tvauc', type=str2bool, default=True,
                    help='Caluclate train and val auc metrics?')
parser.add_argument('-n_splits', metavar='-nsplit', type=int, default=5,
                    help='Number of subgroups to divide batches for less memory')

args = parser.parse_args()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.debug

#args.dilation_rate = 1

"""
input_shape and raw_input_shape is no longer hardcoded.
a DataCollection is initialized with args, using args.patch_x as well as other args members
raw_input_shape is available from a DataCollection instance after it has called dc.load_raw_data()
args.input_channels limits the number of channels extracted during dc.load_raw_data()
args.input_channels == 0 will extract all available channels from the data
input_shape is available from a DataCollection instance after it has called dc.preprocess()
in main.py args is populated with these shapes from a DataCollection instance
"""

# if args.data == 'MNIST' or args.data == 'FASHION_MNIST':
#     args.raw_input_shape = (28, 28, 1)
#
# elif args.data == 'CIFAR10':
#     args.raw_input_shape = (32, 32, 3)
#
# elif args.data == 'MVTEC':
#     if (('grid' in args.anomaly_class) or
#             ('screw' in args.anomaly_class) or
#             ('zipper' in args.anomaly_class)):
#         args.raw_input_shape = (1024, 1024, 1)
#     else:
#         args.raw_input_shape = (1024, 1024, 3)
#
# elif args.data == 'HERA':
#     args.raw_input_shape = (512, 512, 1)
#
# elif args.data == 'HIDE':
#     args.raw_input_shape = (256, 256, 1)
#
# elif args.data == 'LOFAR':
#     args.raw_input_shape = (512, 512, 1)
#
# # elif args.data == 'HERA_PHASE':
# # args.input_shape = (512,512,2)
#
# args.input_shape = args.raw_input_shape
#
# if args.patches:
#     args.input_shape = (args.patch_x, args.patch_y, args.input_shape[-1])
# else:
#     args.patch_x = args.input_shape[0]
#     args.patch_y = args.input_shape[1]
#
# Crop is never actually used
# if args.crop:
#     args.input_shape = (args.crop_x, args.crop_y, args.input_shape[-1])

args.input_shape = None
args.raw_input_shape = None

if args.model_class not in ['AC_UNET', 'ASPP_UNET']:
    args.dilation_rate = 1

if args.model_name != 'None' and args.parent_model_name != 'None':
    raise ValueError('Only model_name or parent_model_name may be given, not both')

if args.task == 'transfer_train':
    if args.parent_model_name == 'None':
        raise ValueError('Parent model name required for transfer training')
if args.task in ['infer', 'eval']:
    if args.model_name == 'None':
        raise ValueError('Model name required for inference or evaluation')

if args.model_name == 'None':
    args.model_name = new_name()

if args.parent_model_name == 'None':
    args.parent_model_name = None

if args.shuffle_seed == 'None':
    args.shuffle_seed = None
else:
    args.shuffle_seed = int(args.shuffle_seed)

if args.limit == 'None':
    args.limit = None
else:
    args.limit = int(args.limit)

if args.rfi_threshold == 'None':
    args.rfi_threshold = None

if args.kernel_regularizer == 'None':
    args.kernel_regularizer = None

if args.clipper == 'None':
    args.clipper = None

if ((args.data_name == 'MNIST') or
        (args.data_name == 'CIFAR10') or
        (args.data_name == 'FASHION_MNIST')):
    args.anomaly_class = int(args.anomaly_class)


def df_to_kwargs(df_all, task='eval', parent_model_name=None, model_name=None, cmd_args=('data_path', 'data_name'), **kwargs):
    """Loads every csv. Can filter it afterwards"""
    if task == 'train':
        return None

    if df_all is None:
        print('loaded df is None')
        return None

    if (parent_model_name is not None and parent_model_name != 'None') and task == 'transfer_train':
        df = df_all.query(f'model_name=="{parent_model_name}"')
    elif (model_name is not None and model_name != 'None') and task in ('eval', 'infer', 'eval_test'):
        df = df_all.query(f'model_name=="{model_name}"')
    else:
        print('Tried to query df')
        return None

    list_kwargs = df.to_dict('records')
    if len(list_kwargs) == 0:
        print('Queried df has 0 rows')
        return None

    if len(list_kwargs) == 1:
        new_kwargs = list_kwargs[0]
    else:
        for i in range(len(list_kwargs)):
            if not np.isnan(list_kwargs[i]['bn_first']): break
        new_kwargs = list_kwargs[i]

    for arg in cmd_args:
        if arg in kwargs.keys():
            new_kwargs[arg] = kwargs[arg]

    new_kwargs['task'] = task
    new_kwargs['cmd_args'] = cmd_args
    if task == 'transfer_train':
        new_kwargs['parent_model_name'] = parent_model_name
        new_kwargs['model_name'] = model_name

    print('Loaded trained model kwargs from csv, overwrote cmd args: ', cmd_args)
    return new_kwargs


def validate_main_kwargs_dict(kwargs: dict):
    if kwargs['task'] != 'train':
        try:
            df = load_csv(**kwargs)
            kwargs_ = df_to_kwargs(df, **kwargs)
            if kwargs_ is None:
                print('Failed to load trained mddel kwargs, using all cmd args')
            else:
                kwargs = kwargs_
        except Exception as e:
            pass

    if kwargs['clipper'] in ('None', None, 'known'):
        kwargs['std_min'] = None
        kwargs['std_max'] = None
        kwargs['perc_min'] = None
        kwargs['perc_max'] = None
    elif kwargs['clipper'] == 'std':
        kwargs['perc_min'] = None
        kwargs['perc_max'] = None
    elif kwargs['clipper'] == 'perc':
        kwargs['std_min'] = None
        kwargs['std_max'] = None

    if kwargs['model_class'] in ('BB_NET',):
        kwargs['loss'] = 'bb'

    # if kwargs['task'] in ('eval', 'infer'):
    #     kwargs['early_stop'] = None
    #     #kwargs['lr'] = None
    #     #kwargs['loss'] = None
    #     #kwargs['epochs'] = None

    if kwargs['model_class'] not in ('AE', 'DAE', 'DKNN'):
        kwargs['latent_dim'] = None
        kwargs['alphas'] = None
        kwargs['neighbours'] = None
        kwargs['radius'] = None
        kwargs['algorithm'] = None
        kwargs['optimal_neighbours'] = None
        kwargs['optimal_alpha'] = None

    return kwargs

