#!/bin/bash
echo "Logging for run_hyp_search.sh at time: $(date)." >> hyp_search.log

limit=None
epochs=150
seed=$(openssl rand -hex 3)
d=$(date +'%m-%d-%Y-%I-%M_')
atype=MISO
ld=8
threshold=None
data=HERA
use_hyp_data=True
anomaly_class=rfi
#data_path=./data
data_path=/home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/

# -data_path /home/ee487519/DatasetsAndConfig/Given/43_Mesarcik_2022/ \
# -data_path ./data \

for model in UNET
#for model in UNET AC_UNET RNET RFI_NET DSC_DUAL_RESUNET DSC_MONO_RESUNET ASPP_UNET
do
	for repeat in 1
	do
	    #for mc in full
	    #for mc in full common
	    for mc in common
	    do
	        #for lr in 1e-4 3e-5 1e-5
	        for lr in 1e-3
	        do

	                #for loss in bce dice
              for loss in bce
              do

                  for dropout in 0.0
                  do


                    for kr in None
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
                          -dropout $dropout\
                          -kernel_regularizer $kr\
                          -lr $lr \
                          -loss $loss \
                          -model_config $mc \
                          -use_hyp_data $use_hyp_data \
                          -seed $d$seed > >(tee -a hyp_search.log) 2>&1
                    done


                  done
              done
	        done
	    done
	done
done

echo "Completed run_hyp_search.sh at time: $(date)." >> hyp_search.log										  
