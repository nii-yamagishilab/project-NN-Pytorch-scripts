#!/bin/bash
########################
# Script for training
# Usage:
# 
# bash 01_train.sh <seed> <config> <model_dir> <al_numsample> 
#        <al_numcycle> <al_epoch_per_cycle> <al_lr> <seedmodel> 
# <seed>: random seed to initialize the weight
# <config>: name of the data configuration file, e.g. config_AL_train_toyset
# <model_dir>: path to the directory of the model
# <al_numsample>: number of samples selected in each active-learning cycle
# <al_numcycle>: maximum number of active-learning cycles to run
# <al_epoch_per_cycle>: number of training epochs in each cycle
# <al_lr>: learning rate
# <seedmodel>: path to the model weights trained on the seed data set.
#     If seedmodel is "None", the model will be trained on
#     seed set for a few epochs before active learning started.
#     Check --active-learning-pre-train-epoch-num below
########################

# the random seed
SEED=$1
# the name of the training config file 
CONFIG=$2
# path to the directory of the model
model_dir=$3

# number of samples selected in each AL cycle
AL_NUMSAMPLE=$4 
# maximum number of AL cycles
AL_NUMCYCLE=$5
# number of training epoch in each AL cycle
AL_EPOCH_P_CYCLE=$6
# learning trate
AL_LR=$7
# path to the seed model
PRETRAINED=$8

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
    cp ${CONFIG}.py ${model_dir}
    cd ${model_dir}
else
    echo "Cannot find ${model_dir}"
    exit 1;
fi

####
# step 3. training
if [ "${PRETRAINED}" == "None" ];
then
    com="python main.py --model-forward-with-file-name
	    --num-workers 4 --epochs ${AL_EPOCH_P_CYCLE} --no-best-epochs 100
	    --batch-size 16 --lr ${AL_LR} 
	    --seed ${SEED} --module-config ${CONFIG}
	    --active-learning-cycle-num ${AL_NUMCYCLE}
	    --active-learning-new-sample-per-cycle ${AL_NUMSAMPLE}
	    --active-learning-pre-train-epoch-num 30 >${log_train_name} 
	    2>${log_err_name}"
else
    com="python main.py --model-forward-with-file-name
	    --num-workers 4 --epochs ${AL_EPOCH_P_CYCLE} --no-best-epochs 100
	    --batch-size 16 --lr ${AL_LR} 
	    --seed ${SEED} --module-config ${CONFIG}
	    --active-learning-cycle-num ${AL_NUMCYCLE}
	    --active-learning-new-sample-per-cycle ${AL_NUMSAMPLE}
	    --trained-model ${PRETRAINED} >${log_train_name} 2>${log_err_name}"
fi

echo -e "${RED}Training starts${NC}"
echo -e "${RED}Please monitor the log trainig: $PWD/${log_train_name}${NC}\n"
echo ${com}
eval ${com}
echo -e "Training process finished"
echo -e "Trainig log has been written to $PWD/${log_train_name}"

