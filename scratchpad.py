#import models
from models import *
from utils import args
from utils.profiling import *
from utils.hardcoded_args import *
from data import *
from utils.data.patches import get_patches, reconstruct
from utils.plotting import save_flops_metric
from utils.metrics import load_csv, extract_results, ResultsCollection

#from models import (Autoencoder,
                   #Discriminator_x)

#from architectures.generic_architecture import end_routine

class Args:
    input_shape = (32, 32, 1)

def dummy_model():
    from tensorflow.keras import layers
    input_data = tf.keras.Input((100, 100, 1), name='data')
    x = input_data
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters=50,
                      kernel_size=3,
                      kernel_initializer='he_normal',
                      strides=1,
                      padding='same')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = tf.keras.Model(inputs=[input_data], outputs=[outputs])
    return model

def auto_test():
    args.set_hera_args()
    tf.compat.v1.disable_eager_execution()
    ae = Autoencoder(args.args, input_tensor=tf.compat.v1.placeholder('float32', shape=(1, 16, 16, 3))
                )
    print('ae')

def set_dummy_args():
    args.args.model = 'dummy_model'
    args.args.anomaly_class = 'dummy_anomaly_class'
    args.args.model_name = 'dummy_name'

def freeze_and_flops():

    model = dummy_model()
    flops = get_flops(model=model)
    print(flops)

def save_dsc_dual_resunet():
    args.args.input_shape = (512,240,1)
    args.args.filters = 64
    args.args.height = 4
    model = DSC_DUAL_RESUNET(args.args)
    with open( './DSC_DUAL_RESUNET_model_summary', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write(f'flops: {get_flops(model)/1e9}')
    print(get_flops(model))
    print(num_trainable_params(model))
    print(num_non_trainable_params(model))

def save_summ():

    set_hera_args(args.args)
    args.args.model_config = 'common'
    args.args = resolve_model_config_args(args.args)
    args.args.level_blocks = 2
    print(args.args)
    #set_dummy_args()
    #model = UNET_Mesarcik(args.args, n_filters=16)
    #model = AC_UNET(args.args, n_filters=64)
    #model = RFI_NET(args.args)
    #model = RFI_NET_gen(args.args)
    model = UNET(args.args)
    #model = dummy_model()
    #save_summary(model, args.args)

    with open( './UNET_class_common_model_summary', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    print(get_flops(model))
    print(num_trainable_params(model))
    print(num_non_trainable_params(model))


def main():
    model = UNET3(Args, dropout=0.1, height=3, dilation_rate=3, layers_per_level=1)

    with open('test_s32_h3_l1.txt','w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    #model = UNET(Args, dropout=0.1)
    #with open('test.txt','w') as f:
    #    model.summary(print_fn=lambda x: f.write(x + '\n'))

def load_model():
    args.set_hera_args()
    args.args.model_name = 'wild-fluorescent-gazelle-of-shopping'

    if args.args.data == 'HERA':
        data = load_hera(args.args)
    elif args.args.data == 'LOFAR':
        data = load_lofar(args.args)
    (unet_train_dataset, train_data, train_labels, train_masks,
     ae_train_dataset, ae_train_data, ae_train_labels,
     test_data, test_labels, test_masks,test_masks_orig) = data

    print(test_data.shape)

    unet = UNET(args.args)
    save_summary(unet,args.args)
    #ae = Autoencoder(args.args)
    #discriminator = Discriminator_x(args.args)

    unet(np.random.random((1,32,32,1)))
    #unet(unet_train_dataset.take(1))
    #ae(test_data)
    #discriminator(test_data)

    path = '/home/ee487519/PycharmProjects/RFI-NLN/outputs/UNET/rfi/wild-fluorescent-gazelle-of-shopping/training_checkpoints/'
    #path = '/home/ee487519/PycharmProjects/RFI-NLN/outputs/DAE_disc/rfi/Charl-first-DAE-seed-aseed/training_checkpoints/'
    unet.load_weights(path + 'checkpoint_full_model_unet')
    #discriminator.load_weights(path + 'checkpoint_full_model_disc')
    end_routine(train_data, test_data, test_labels, test_masks, test_masks_orig, [unet], 'UNET',
                args.args)

def save_all_summaries():
    args.args = set_hera_args(args.args)
    args.args.model_config = 'common'
    args.args = resolve_model_config_args(args.args)
    unet = UNET(args.args)
    save_summary_to_folder(unet, 'model_summaries', args.args)


def test_patches_arb():
    new_shape = (128, 512, 1)
    data = np.random.random((1,128,512,1))
    patch_x, patch_stride_x = 32, 32
    patch_y, patch_stride_y = 8, 8
    p_size = (1, patch_x, patch_y, 1)
    s_size = (1, patch_stride_x, patch_stride_y, 1)
    rate = (1, 1, 1, 1)

    patches = get_patches_arbitrary(data, None, p_size, s_size, rate, 'VALID')
    #patches = np.random.random(((256, 32, 8, 1)))

    recon = reconstruct(patches, new_shape, patch_x, patch_y)

    print(recon.shape == data.shape)
    print(np.allclose(recon, data))

def test_patches():
    new_shape = (4, 4, 1)
    data = np.random.random((2,4,4,1))
    data[0, 0:2, 0:2, 0] = 0
    data[0, 0:2, 2:4, 0] = 1
    data[0, 2:4, 0:2, 0] = 2
    data[0, 2:4, 2:4, 0] = 3
    data[1, 0:2, 0:2, 0] = 4
    data[1, 0:2, 2:4, 0] = 5
    data[1, 2:4, 0:2, 0] = 6
    data[1, 2:4, 2:4, 0] = 7
    patch_x, patch_stride_x = 2, 2
    patch_y, patch_stride_y = 2, 2
    p_size = (1, patch_x, patch_y, 1)
    s_size = (1, patch_stride_x, patch_stride_y, 1)
    rate = (1, 1, 1, 1)

    patches = get_patches(data, p_size, s_size, rate, 'VALID')
    #patches = np.random.random(((256, 32, 8, 1)))

    #class Args:
    #    patch_x=2
    #    patch_y=2
    #    data = 'HERA'
    #recon = reconstruct(patches, Args)

    #print(recon.shape == data.shape)
    #print(np.allclose(recon, data))

def test_flop_f1_plot():
    group_filter_dict = {'model': 'UNET'}
    rc = ResultsCollection('LOFAR', group_filter_dict={}, dir_path='./outputs/LOFAR-hyp-search')
    rc.load_groups()
    rc.save_flops_f1()
    #for label, test_f1 in zip(rc.labels, rc.test_f1_means):
        #print(label)
        #print(label, test_f1)

def test_training_curve_plot():
    rc = ResultsCollection('HERA', group_filter_dict={}, dir_path='./outputs/2022-11-02-hyp-search')
    rc.save_training_metrics('AC_UNET', 'bce', )



if __name__ == '__main__':
  #  load_model()
    #save_all_summaries()
    test_flop_f1_plot()
    #save_summ()
    #save_dsc_dual_resunet()
    #freeze_and_flops()
    #auto_test()
