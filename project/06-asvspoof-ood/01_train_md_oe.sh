#!/bin/bash
########################
# Script for evaluation
# Usage:
#   1. please check that config.py has been properly configured
#   2. $: bash 00_train.sh SEED CONFIG_NAME > /dev/null 2>&1 &
#   3. please check log_train and log_err to monitor the training 
#      process
########################

# the random seed
SEED=$1
# the name of the training config file for out-lier exposure
CONFIG=$2
# the name of the 2nd step training config for OOD
CONFIG2=$3

log_train_name=log_train_oe
log_err_name=log_train_oe_err

echo -e "\nTraining (with outlier-exposure) starts"
echo -e "Please monitor the log trainig: $PWD/${log_train_name}.txt\n"

SAVEMODEL=trained_network_oe
com="python main.py --module-model model_oe 
	    --save-trained-name ${SAVEMODEL}
	    --way-to-merge-datasets separate
	    --model-forward-with-file-name --num-workers 3 --epochs 100 
       	    --no-best-epochs 50 --batch-size 64 
       	    --sampler block_shuffle_by_length --lr-decay-factor 0.5 
       	    --lr-scheduler-type 1 --lr 0.0003 
       	    --not-save-each-epoch --seed ${SEED} 
       	    --module-config ${CONFIG}  >${log_train_name}.txt 2>${log_err_name}.txt "

if [ ! -e ${SAVEMODEL}.pt ];
then
    echo ${com}
    eval ${com}
    echo -e "Trainig log has been written to $PWD/${log_train_name}.txt"
else
    echo -e "Skip out-lier training; using existing ${SAVEMODEL}.pt"
fi


log_train_name=log_train_oe_md
log_err_name=log_train_oe_md_err

echo -e "\nTraining for m-distance calculation starts"
echo -e "Please monitor the log trainig: $PWD/${log_train_name}.txt\n"

# This step will load the trained_network from step 1
#  
# only collect statistics
SAVEMODEL=trained_network_oe_md
com="python main.py --module-config ${CONFIG2} --module-model model_ood 
	    --model-forward-with-file-name --epochs 1 
	    --save-trained-name ${SAVEMODEL} >${log_train_name}.txt 2>${log_err_name}.txt "
if [ ! -e ${SAVEMODEL}.pt ];

then
    echo ${com}
    eval ${com}
    echo -e "Trainig log has been written to $PWD/${log_train_name}.txt"
else
    echo -e "Skip m-distance training; using existing ${SAVEMODEL}.pt"
fi

