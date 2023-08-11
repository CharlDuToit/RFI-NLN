#!/bin/bash
echo "Logging for run_lofar.sh at time: $(date)." >> lofar_twt_scratch.log

task=train
model_name=None
parent_model_name=None
limit=28
epochs=400

train_with_test=True
use_hyp_data=True
calc_train_val_auc=True
rfi_set=combined
rfi_split_ratio=0.01
patch=64
filters=16
height=4
dropout=0.1
batch_size=64
data_name=LOFAR
lofar_subset=full
activation=relu
final_activation=sigmoid
loss=dice
lr=1e-4
lr_lin_decay=1.0
rescale=True
log=True
bn_first=False
scale_per_image=False
clip_per_image=False
clipper=perc
std_max=3.86 # LOFAR
std_min=-0.06 # LOFAR
perc_max=99.8 # HERA and LOFAR
perc_min=0.2
# perc_min=0.2 # HERA
kernel_regularizer=l2
dilation_rate=3
early_stop=1000
shuffle_seed=None
val_split=0.2
shuffle_patches=True
epoch_image_interval=50
images_per_epoch=10
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=32
flag_test_data=False
anomaly_class=rfi
threshold=None

data_path=./data
# data_path=/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/

#for model_class in UNET AC_UNET ASPP_UNET RNET RNET5 RFI_NET DSC_MONO_RESUNET DSC_DUAL_RESUNET
#for model_class in UNET AC_UNET ASPP_UNET RNET RNET5 RFI_NET
for model_class in AC_UNET ASPP_UNET DSC_MONO_RESUNET DSC_DUAL_RESUNET
do
		for repeat in 1 2 3 4 
		do
		python3 -u main.py \
		      -model_class $model_class\
		      -task $task\
			-train_with_test $train_with_test\
			-rescale $rescale\
			-bn_first $bn_first\
		      -rfi_set $rfi_set\
		      -rfi_split_ratio $rfi_split_ratio\
		      -model_name $model_name\
		      -parent_model_name $parent_model_name\
		      -anomaly_class $anomaly_class\
				  -anomaly_type $atype\
				  -limit $limit\
          -percentage_anomaly $percentage\
				  -epochs $epochs \
				  -latent_dim $ld \
				  -alphas 0.5\
				  -neighbours 2 4 \
				  -radius 0.1 0.5 \
				  -algorithm knn\
				  -data_name $data_name\
				  -data_path $data_path\
				  -seed $d$seed\
				  -debug 0\
				  -log $log\
				  -rotate False\
				  -crop False\
				  -crop_x 0\
				  -crop_y 0\
				  -patches True\
				  -patch_x $patch\
				  -patch_y $patch\
				  -calc_train_val_auc $calc_train_val_auc\
				  -patch_stride_x $patch\
				  -patch_stride_y $patch\
				  -flag_test_data $flag_test_data\
				  -rfi None\
				  -rfi_threshold $threshold\
				  -lofar_subset $lofar_subset\
				  -scale_per_image $scale_per_image\
				  -clip_per_image $clip_per_image\
				  -clipper $clipper\
				  -std_max $std_max \
				  -std_min $std_min\
				  -perc_max $perc_max\
				  -perc_min $perc_min\
				  -filters $filters\
				  -height $height\
				  -level_blocks 1\
				  -model_config args\
				  -dropout $dropout\
				  -batch_size $batch_size\
				  -buffer_size 1024\
				  -optimal_alpha True\
				  -optimal_neighbours True\
				  -use_hyp_data $use_hyp_data \
				  -lr $lr \
                  -lr_lin_decay $lr_lin_decay \
				  -loss $loss \
				  -kernel_regularizer $kernel_regularizer\
				  -input_channels 1\
				  -dilation_rate $dilation_rate\
				  -epoch_image_interval $epoch_image_interval \
				  -images_per_epoch $images_per_epoch \
				  -early_stop $early_stop\
				  -shuffle_seed $shuffle_seed\
				  -val_split $val_split\
				  -activation $activation\
				  -final_activation $final_activation\
				  -output_path ./outputs\
				  -save_dataset False\
				  -shuffle_patches $shuffle_patches > >(tee -a lofar_twt_scratch.log) 2>&1
		done 
done

echo "Completed run_lofar.sh at time: $(date)." >> lofar_twt_scratch.log	
