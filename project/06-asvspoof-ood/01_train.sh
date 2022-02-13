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
# the name of the training config file 
CONFIG=$2

log_train_name=log_train
log_err_name=log_train_err

echo -e "\nTraining starts"
echo -e "Please monitor the log trainig: $PWD/${log_train_name}.txt\n"

com="python main.py --model-forward-with-file-name --num-workers 3 --epochs 100 
       --no-best-epochs 50 --batch-size 64 
       --sampler block_shuffle_by_length --lr-decay-factor 0.5 
       --lr-scheduler-type 1 --lr 0.0003 
       --not-save-each-epoch --seed ${SEED} 
       --module-config ${CONFIG}  >${log_train_name}.txt 2>${log_err_name}.txt "
echo ${com}
eval ${com}
echo -e "\nTraining process finished"
echo -e "Trainig log has been written to $PWD/${log_train_name}.txt"





