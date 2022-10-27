#!/bin/bash
########################
# Script for training
# Usage:
#   1. please check that config.py has been properly configured
#   2. $: bash 00_train.sh SEED CONFIG_NAME > /dev/null 2>&1 &
#   4. please check log_train and log_err to monitor the training 
#      process
########################

# the random seed
SEED=$1

# the name of the training config file 
CONFIG=$2

# folder name
model_dir=$3

log_train_name=log_train
log_err_name=log_train_err

####
# step1. copy files & enter the folder
if [ -d ${model_dir} ];
then
    cp ./main.py ${model_dir}
    cp ./${CONFIG}.py ${model_dir}
    cp ${model_dir}/../../model.py ${model_dir}
    cd ${model_dir}
else
    echo "Cannot find ${model_dir}"
    exit 1;
fi

####
# step2. decide whether this model requires SSL fine-tune
FINETUNE=`echo ${model_dir} | grep ft | wc -l`

if [ ${FINETUNE} -gt 0 ];
then
    echo "Training process will fine-tune SSL"
    # command to train model with SSL fine-tuned
    #  the learning rate and batch size are different
    com="python main.py --model-forward-with-file-name 
	     --num-workers 8 --epochs 100
	     --no-best-epochs 10 --batch-size 8 
	     --sampler block_shuffle_by_length --lr-decay-factor 0.5 
	     --lr-scheduler-type 1 --lr 0.000001 
	     --not-save-each-epoch --seed ${SEED}
	     --module-config ${CONFIG}
	     >${log_train_name} 2>${log_err_name}"
else
    echo "Training process use fixed SSL, no fine-tuning"
    # command to train model without fine-tuning SSL
    com="python main.py --model-forward-with-file-name 
	    --num-workers 3 --epochs 100 
	    --no-best-epochs 10 --batch-size 64 
	    --sampler block_shuffle_by_length --lr-decay-factor 0.5 
	    --lr-scheduler-type 1 --lr 0.0003 
	    --not-save-each-epoch --seed ${SEED} 
	    --module-config ${CONFIG}  
	    >${log_train_name} 2>${log_err_name}"
fi

####
# step 3. training
echo -e "Training starts"
echo -e "Please monitor the log trainig: $PWD/${log_train_name}\n"
echo ${com}
eval ${com}
echo -e "Training process finished"
echo -e "Trainig log has been written to $PWD/${log_train_name}"

