#!/bin/bash
echo "Logging for run_hera_charl_transfer.sh at time: $(date)." >> hera_charl_transfer.log

task='transfer_train'
model_name=None
#parent_model_name=None
train_with_test=False
epochs=800
limit=14
freeze_top_layers=True

use_hyp_data=False
calc_train_val_auc=True
rfi_set=combined
rfi_split_ratio=0.01
patch=64
filters=16
height=4
dropout=0.1
batch_size=64
data_name=HERA_CHARL
lofar_subset=full
activation=relu
final_activation=sigmoid
loss=dice
lr=1e-4
lr_lin_decay=1.0
rescale=True
log=True
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
# model class, parent model name
# for i in AC_UNET,jolly-nondescript-porcupine-of-endeavor AC_UNET,little-kiwi-of-enjoyable-protection AC_UNET,original-ginger-butterfly-of-tranquility AC_UNET,arboreal-spicy-chicken-from-mars AC_UNET,cute-chestnut-scallop-of-bloom AC_UNET,spicy-hopping-chicken-of-merriment AC_UNET,amusing-raccoon-of-unusual-wonder AC_UNET,slim-glistening-bullfinch-of-aurora ASPP_UNET,abstract-hilarious-gerbil-of-temperance ASPP_UNET,jumping-practical-mastiff-of-proficiency ASPP_UNET,cherubic-platinum-scorpion-of-penetration ASPP_UNET,prudent-natural-pig-of-karma ASPP_UNET,armored-lilac-ant-of-finesse ASPP_UNET,enthusiastic-smiling-jaguar-of-current ASPP_UNET,smart-rousing-unicorn-of-tempest ASPP_UNET,busy-sincere-donkey-from-lemuria DSC_DUAL_RESUNET,calm-brainy-turaco-of-youth DSC_DUAL_RESUNET,hospitable-athletic-seriema-of-bliss DSC_DUAL_RESUNET,industrious-mysterious-capybara-of-courtesy DSC_DUAL_RESUNET,vigilant-attractive-myna-of-tolerance DSC_DUAL_RESUNET,authentic-orthodox-jaybird-of-education DSC_DUAL_RESUNET,meek-oxpecker-of-phenomenal-sympathy DSC_DUAL_RESUNET,dramatic-aloof-heron-of-happiness DSC_DUAL_RESUNET,mysterious-versatile-buffalo-of-expertise DSC_MONO_RESUNET,perfect-wild-cuttlefish-of-modernism DSC_MONO_RESUNET,ancient-utopian-harrier-of-sympathy DSC_MONO_RESUNET,calm-asparagus-tarsier-of-democracy DSC_MONO_RESUNET,holistic-cautious-tuna-of-realization DSC_MONO_RESUNET,prompt-space-bumblebee-of-gallantry DSC_MONO_RESUNET,smart-translucent-tapir-of-finesse DSC_MONO_RESUNET,poised-blazing-hamster-from-valhalla DSC_MONO_RESUNET,dexterous-snobbish-ibex-of-cleaning RFI_NET,silky-esoteric-gecko-of-security RFI_NET,electric-tangerine-seriema-of-pride RFI_NET,devious-spotted-pigeon-of-tenacity RFI_NET,slim-accurate-malkoha-of-experience RFI_NET,prompt-crocodile-of-astonishing-greatness RFI_NET,glaring-gainful-moose-of-completion RFI_NET,lumpy-tan-gorilla-of-karma RFI_NET,spectral-quartz-camel-of-honor RNET,enlightened-adept-goldfish-of-kindness RNET,icy-garrulous-junglefowl-of-opportunity RNET,arcane-sloppy-woodpecker-of-completion RNET,mauve-mussel-of-authentic-stamina RNET,illustrious-eminent-angora-of-agility RNET,quick-quixotic-catfish-of-felicity RNET,bulky-ninja-chameleon-of-justice RNET,fanatic-seriema-of-impressive-discussion RNET5,gentle-wealthy-fox-of-virtuosity RNET5,loutish-delectable-asp-of-revolution RNET5,righteous-kudu-of-remarkable-felicity RNET5,flat-attractive-emu-of-judgment RNET5,poised-resilient-tuna-of-innovation RNET5,stoic-blazing-hound-of-enthusiasm RNET5,proud-malamute-of-abstract-intensity RNET5,sparkling-prompt-worm-of-success UNET,furry-groovy-cuckoo-of-debate UNET,tiny-resilient-wolf-of-joviality UNET,conscious-smoky-husky-of-destiny UNET,silky-sidewinder-of-ultimate-improvement UNET,huge-burrowing-stallion-of-fascination UNET,vagabond-fractal-warthog-from-camelot UNET,inquisitive-lori-of-authentic-perfection UNET,translucent-unselfish-ocelot-of-force;
for i in AC_UNET,flashy-loutish-bustard-of-amplitude AC_UNET,vagabond-brawny-gazelle-of-love;
# for i in AC_UNET,flashy-loutish-bustard-of-amplitude AC_UNET,vagabond-brawny-gazelle-of-love AC_UNET,poised-chubby-bison-of-love AC_UNET,vagabond-mysterious-toucan-of-modernism AC_UNET,keen-magnetic-swan-of-security AC_UNET,delightful-cobalt-ferret-of-sunshine AC_UNET,nondescript-acoustic-panther-of-art AC_UNET,stylish-copper-marmot-of-symmetry ASPP_UNET,discerning-cute-goose-of-acumen ASPP_UNET,greedy-sturgeon-of-inspiring-patience ASPP_UNET,optimistic-quick-mongoose-of-promise ASPP_UNET,cherubic-adept-caracal-of-election ASPP_UNET,simple-hilarious-hound-from-eldorado ASPP_UNET,overjoyed-ivory-serval-of-psychology ASPP_UNET,elegant-important-doberman-of-temperance ASPP_UNET,caped-speedy-crow-of-peace DSC_DUAL_RESUNET,fine-impala-of-fascinating-bliss DSC_DUAL_RESUNET,vehement-mysterious-coyote-of-novelty DSC_DUAL_RESUNET,tireless-limpet-of-pragmatic-progress DSC_DUAL_RESUNET,refined-authentic-pony-of-variation DSC_DUAL_RESUNET,amazing-daffodil-seal-of-wind DSC_DUAL_RESUNET,sweet-tan-rabbit-of-music DSC_DUAL_RESUNET,nostalgic-gray-alpaca-of-exercise DSC_DUAL_RESUNET,fortunate-xanthic-serval-of-tact DSC_MONO_RESUNET,bouncy-arboreal-marmoset-of-tempest DSC_MONO_RESUNET,sociable-neon-badger-of-will DSC_MONO_RESUNET,practical-poetic-labrador-of-potency DSC_MONO_RESUNET,juicy-ostrich-of-flawless-intensity DSC_MONO_RESUNET,flat-boisterous-bison-of-wizardry DSC_MONO_RESUNET,pristine-soft-mongoose-of-might DSC_MONO_RESUNET,imperial-sepia-rook-of-force DSC_MONO_RESUNET,rational-manatee-of-ancient-completion RFI_NET,electric-dancing-mole-of-debate RFI_NET,tall-nano-seagull-of-awe RFI_NET,brainy-shaggy-crocodile-from-saturn RFI_NET,sturdy-utopian-uakari-of-merriment RFI_NET,hilarious-garrulous-snake-of-excitement RFI_NET,gigantic-bizarre-kagu-of-respect RFI_NET,brawny-affable-tarsier-of-discourse RFI_NET,godlike-heavenly-wallaby-of-lightning RNET,meticulous-sassy-albatross-of-science RNET,calculating-acrid-prawn-of-symmetry RNET,spry-chubby-magpie-of-lightning RNET,futuristic-married-kudu-of-fame RNET,excellent-loud-mongoose-of-tempest RNET,sepia-goldfish-of-ultimate-correction RNET,amusing-versatile-cow-of-assurance RNET,upbeat-feathered-tarantula-of-enterprise RNET5,nifty-bouncy-petrel-of-influence RNET5,spry-malachite-flounder-of-prowess RNET5,mellow-rose-markhor-of-prosperity RNET5,slick-papaya-seriema-of-strength RNET5,electric-delectable-coucal-of-fame RNET5,hungry-determined-barracuda-of-aptitude RNET5,burrowing-tricky-swan-of-wealth RNET5,puzzling-potoo-of-utter-saturation UNET,ambrosial-spectral-snake-of-improvement UNET,hasty-brilliant-oriole-of-reward UNET,real-merry-trout-of-kindness UNET,determined-pink-quokka-from-neptune UNET,calculating-tourmaline-seriema-from-avalon UNET,witty-congenial-deer-of-criticism UNET,woodoo-impressive-caribou-of-pleasure UNET,meek-banana-wildcat-of-felicity;
do
    set -- $i;
		python3 -u main.py \
		      -model_class $1\
		      -task $task\
			-rescale $rescale\
			    -freeze_top_layers $freeze_top_layers\
		      -rfi_set $rfi_set\
			-train_with_test $train_with_test\
		      -rfi_split_ratio $rfi_split_ratio\
		      -model_name $model_name\
		      -parent_model_name $2\
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
				  -shuffle_seed None \
				  -val_split $val_split\
				  -activation $activation\
				  -final_activation $final_activation\
				  -output_path ./outputs\
				  -save_dataset False\
				  -shuffle_patches $shuffle_patches > >(tee -a hera_charl_transfer.log) 2>&1
done
IFS=$OLDIFS

echo "Completed run_hera_charl_transfer.sh at time: $(date)." >> hera_charl_transfer.log
