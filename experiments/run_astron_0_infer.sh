#!/bin/bash
echo "Logging for run_lofar.sh at time: $(date)." >> lofar.log

limit=None
epochs=150
percentage=0.0
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=32
patch=32
threshold=None
data=LOFAR
use_hyp_data=False
anomaly_class=rfi
model_config=common
loss=bce
lr=1e-4
#data_path=./data
data_path=/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/

# -data_path /home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/ \
# -data_path ./data \

#for model in UNET DAE RNET RFI_NET
for model in UNET
do
		for repeat in 1
		do
		python3 -u main.py -model $model\
				  -limit $limit\
				  -data $data\
				  -data_path $data_path \
				  -anomaly_class $anomaly_class\
				  -anomaly_type $atype\
				  -epochs $epochs \
				  -latent_dim $ld \
				  -rfi_threshold $threshold\
				  -lr $lr \
				  -loss $loss \
				  -model_config $model_config \
				  -use_hyp_data $use_hyp_data \
				  -neighbours 20\
				  -algorithm knn\
				  -kernel_regularizer None\
				  -dropout 0.0\
				  -input_channels 1\
				  -seed $d$seed > >(tee -a lofar.log) 2>&1
		done 
done

echo "Completed run_lofar.sh at time: $(date)." >> lofar.log	
