import argparse
import os
#from utils.data import sizes
from coolname import generate_slug as new_name
from .hardcoded_args import resolve_model_config_args

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

parser.add_argument('-model', metavar='-m', type=str, default='AE',
                    choices={'AC_UNET', 'UNET', 'AE', 'DAE', 'DKNN', 'RNET', 'CNN_RFI_SUN', 'RFI_NET', 'AE-SSIM', 'DSC_DUAL_RESUNET', 'DSC_MONO_RESUNET', 'ASPP_UNET'},
                    help='Which model to train and evaluate')
parser.add_argument('-limit', metavar='-l', type=str, default='None',
                    help='Limit on the number of samples in training data ')
parser.add_argument('-anomaly_class', metavar='-a', type=str, default=2,
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
parser.add_argument('-data', metavar='-d', type=str, default='HERA',
                    choices={'HERA', 'HERA_PHASE', 'LOFAR', 'MNIST', 'CIFAR10', 'FASHION_MNIST'},
                    help='The dataset for training and testing the model on')
parser.add_argument('-data_path', metavar='-mp', type=str, default='./data',
                    help='Path to MVTecAD training data')
parser.add_argument('-seed', metavar='-s', type=str,
                    help='The random seed used for naming output files')
parser.add_argument('-debug', metavar='-de', type=str, default='0',
                    choices={'0', '1', '2', '3'}, help='TF debug level')
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
parser.add_argument('-rfi', metavar='-rfi', type=str, default=None,
                    help='HERA RFI label to exclude from training')
parser.add_argument('-rfi_threshold', metavar='-rfi_threshold', type=str, default=None,
                    help='AOFlagger base threshold')
parser.add_argument('-lofar_subset', metavar='-lofar_subset', type=str, default='full',
                    choices={'L629174', 'L631961', 'full'}, help='LOFAR subset to use for training')
parser.add_argument('-clip', metavar='-clip', type=float, default=None,
                    help='AOFlagger base threshold')
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
parser.add_argument('-batch_size', metavar='-bas', type=int, default=1024,
                    help='Batch size')
parser.add_argument('-buffer_size', metavar='-bus', type=int, default=2**13,
                    help='Buffer size')
parser.add_argument('-optimal_alpha', metavar='-oalpha', type=str2bool, default=True,
                    help='Replace args.alphas with list of one alpha which is optimized for args.data for AE')
parser.add_argument('-optimal_neighbours', metavar='-oneighs', type=str2bool, default=True,
                    help='Replace args.neighbours with list of one neighbours which is optimized for AE')
parser.add_argument('-use_hyp_data', metavar='-hypd', type=str2bool, default=False,
                    help='Use reduced dataset for hyper-parameter search')
parser.add_argument('-lr', metavar='-lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('-loss', metavar='-ls', type=str, default='bce',
                    choices={'bce', 'mse', 'dice'},
                    help='Loss function')
parser.add_argument('-kernel_regularizer', metavar='-kr', type=str, default='None',
                    choices={'l2', 'l1', 'None'},
                    help='Kernel regularizer')
parser.add_argument('-input_channels', metavar='-ch', type=int, default=0,
                    help='Number of channels to extract from data. 0 will extract all channels')

args = parser.parse_args()
args.model_name = new_name()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = args.debug
args.dilation_rate = 1

"""
input_shape and raw_input_shape is no longer hardcoded.
resolve_model_config_args will change the patch size if args.model_config != 'args'
a DataCollection is initialized with args, using args.patch_x as well as other args members
raw_input_shape is available from a DataCollection instance after it has called dc.load_raw_data()
args.input_channels limits the number of channels extracted during dc.load_raw_data()
args.input_channels == 0 will extract all available channels from the data
input_shape is available from a DataCollection instance after it has called dc.preprocess()
in main.py args is populated with these shapes from a DataCollection instance
"""
args.input_shape = None
args.raw_input_shape = None
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

if args.limit == 'None':
    args.limit = None
else:
    args.limit = int(args.limit)

if args.rfi_threshold == 'None':
    args.rfi_threshold = None

if args.kernel_regularizer == 'None':
    args.kernel_regularizer = None

if ((args.data == 'MNIST') or
        (args.data == 'CIFAR10') or
        (args.data == 'FASHION_MNIST')):
    args.anomaly_class = int(args.anomaly_class)

args = resolve_model_config_args(args)
