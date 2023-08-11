#!/bin/bash
echo "Logging for run_lofar_eval.sh at time: $(date)." >> lofar_eval.log

task='eval'
# model_name=None
parent_model_name=None

limit=None
use_hyp_data=True
calc_train_val_auc=True
epochs=100
rfi_set=combined
rfi_split_ratio=0.01
patch=64
filters=16
height=4
dropout=0.0
batch_size=64
data_name=LOFAR
lofar_subset=full
activation=relu
final_activation=sigmoid
loss=dice
lr=1e-4
lr_lin_decay=1.0
scale_per_image=False
clip_per_image=False
clipper=perc
std_max=3.86 # LOFAR
std_min=-0.06 # LOFAR
perc_max=99.8 # HERA and LOFAR
perc_min=0.2
# perc_min=0.2 # HERA
kernel_regularizer=None
dilation_rate=3
early_stop=200
# shuffle_seed=None
val_split=0.2
shuffle_patches=True
epoch_image_interval=0
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
#data_path=/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/

OLDIFS=$IFS;
IFS=',';
# model class, model name, shuffle seed
for i in ASPP_UNET,'energetic-russet-dragon-of-fragrance',2826171739 ASPP_UNET,'camouflaged-nimble-mongrel-of-trust',43261422 ASPP_UNET,'urban-magenta-bandicoot-of-popularity',367681113 ASPP_UNET,'nocturnal-interesting-ammonite-of-wonder',2821888956;
do
    set -- $i;
		python3 -u main.py \
		      -model_class $1\
		      -task $task\
		      -rfi_set $rfi_set\
		      -rfi_split_ratio $rfi_split_ratio\
		      -model_name $2\
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
				  -log True\
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
				  -shuffle_seed $3 \
				  -val_split $val_split\
				  -activation $activation\
				  -final_activation $final_activation\
				  -output_path ./outputs\
				  -save_dataset False\
				  -shuffle_patches $shuffle_patches > >(tee -a lofar.log) 2>&1
done
IFS=$OLDIFS

echo "Completed run_lofar_eval.sh at time: $(date)." >> lofar_eval.log