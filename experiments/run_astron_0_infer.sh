#!/bin/bash
echo "Logging for run_astron_0_infer.sh at time: $(date)." >> astron_0.log

seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

# -data_path /home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/ \
# -data_path ./data \

size=64

python3 -u main.py -model UNET\
          -model_name rational-polar-groundhog-of-judgment\
          -task infer\
				  -data ASTRON_0\
				  -data_path /home/ee487519/DatasetsAndConfig/Given/ASTRON/ \
				  -anomaly_class rfi\
				  -anomaly_type MISO\
				  -use_hyp_data False\
				  -output_path ./outputs/LOFAR_nonhyp\
				  -val_split 0.0\
				  -images_per_epoch 10\
				  -batch_size 64\
				  -patches True\
				  -patch_x $size\
				  -patch_y $size\
				  -patch_stride_x $size\
				  -patch_stride_y $size\
				  -height 4\
				  -filters 64\
				  -loss bce\
				  -final_activation sigmoid\
				  -seed $d$seed > >(tee -a astron_0.log) 2>&1

echo "Completed run_astron_0_infer.sh at time: $(date)." >> astron_0.log
