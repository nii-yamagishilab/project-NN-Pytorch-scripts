#!/bin/bash
########################
# Script for training
# Usage:
# 
# bash 01_train.sh <seed> <config> <model_dir> <flag_base>
# <seed>: random seed to initialize the weight
# <config>: name of the data configuration file, 
#           e.g. config_train_toyset_vocoded
# <model_dir>: path to the directory of the model
# <flag_base>: True of False, whether this is baseline
#              method without data augmentation
########################

# the random seed
SEED=$1
# the name of the training config file 
CONFIG=$2
# path to the directory of the model
model_dir=$3
# flag
flag=$4

if [ "$#" -ne 4 ]; then
    echo -e "Invalid input arguments. Please check the doc of script."
    exit 1;
fi


# name of the log file
log_train_name=log_train
log_err_name=log_train_err
RED='\033[0;32m'
NC='\033[0m'

####
# step1. copy files & enter the directory
if [ -d ${model_dir} ];
then
    echo "cd ${model_dir}"
    cp main.py ${model_dir}
    cp ${model_dir}/../../model.py ${model_dir}
    cp ${model_dir}/../../data_augment.py ${model_dir}
    cp ${CONFIG}.py ${model_dir}
    cd ${model_dir}
else
    echo "Cannot find ${model_dir}"
    exit 1;
fi

####
# step 3. training
if [ "${flag}" == "True" ];
then
    # for baseline method without data augmentation
    com="python main.py --model-forward-with-file-name 
    		--num-workers 8 --epochs 20 --no-best-epochs 10 
		--batch-size 8 --lr 0.000001 --not-save-each-epoch 
		--sampler block_shuffle_by_length 
		--lr-decay-factor 0.1 --lr-scheduler-type 1 
		--lr-steplr-size 5 --seed ${SEED}
		--module-config ${CONFIG} >${log_train_name} 2>${log_err_name}"
else
    # for baseline method with data augmentation and contrastive loss
    com="python main.py --model-forward-with-file-name 
                --num-workers 8 --epochs 50 --no-best-epochs 10 
		--batch-size 1 --lr 0.000005 --not-save-each-epoch 
		--sampler block_shuffle_by_length 
		--lr-decay-factor 0.1 --lr-scheduler-type 1 
		--lr-steplr-size 10 --seed ${SEED} 
		--module-config ${CONFIG} >${log_train_name} 2>${log_err_name}"
fi

echo -e "${RED}Training starts${NC}"
echo -e "${RED}Please monitor the log trainig: $PWD/${log_train_name}${NC}\n"
echo ${com}
eval ${com}
echo -e "Training process finished"
echo -e "Training log has been written to $PWD/${log_train_name}"
echo -e "Detailed error log has been written to $PWD/${log_err_name}"

