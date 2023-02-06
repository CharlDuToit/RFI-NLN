# INIT FILE
import sys
sys.path.insert(1,'../')

from .ae import main as train_ae 
from .ae_ssim import main as train_ae_ssim
from .dae import main as train_dae

from .resnet import main as train_resnet
from .cnn_rfi_sun import main as train_cnn_rfi_sun

from .architecture_from_args import load_architecture
from .helper import end_routine
