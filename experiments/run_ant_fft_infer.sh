#!/bin/bash
echo "Logging for run_ant_fft_infer.sh at time: $(date)." >> run_ant_fft_infer.log

seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')

# -data_path /home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/ \
# -data_path ./data \
# RNET  illustrious-poodle-of-stimulating-joviality, filters=16
# UNET  hospitable-intrepid-coucal-of-novelty, filters=64
# DSC_DUAL_RESUNET eccentric-cute-grasshopper-of-refinement, filters=16
size=64

python3 -u main.py -model RNET\
          -model_name illustrious-poodle-of-stimulating-joviality\
          -task infer\
				  -data ant_fft_000_094_t4096_f4096\
				  -data_path /home/ee487519/PycharmProjects/correlator/\
				  -anomaly_class rfi\
				  -anomaly_type MISO\
				  -use_hyp_data False\
				  -output_path ./outputs/infer_test\
				  -val_split 0.0\
				  -images_per_epoch 10\
				  -batch_size 64\
				  -patches True\
				  -patch_x $size\
				  -patch_y $size\
				  -patch_stride_x $size\
				  -patch_stride_y $size\
				  -height 4\
				  -filters 16\
				  -loss bce\
				  -final_activation sigmoid\
				  -save_dataset True\
				  -seed $d$seed > >(tee -a run_ant_fft_infer.log) 2>&1

echo "Completed run_ant_fft_infer.sh at time: $(date)." >> run_ant_fft_infer.log
