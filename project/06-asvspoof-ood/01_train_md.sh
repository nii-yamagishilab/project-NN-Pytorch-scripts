#!/bin/bash

# the random seed
SEED=$1
# the name of the training config file 
CONFIG=$2

log_train_name=log_train
log_err_name=log_train_err

echo -e "\nTraining starts"
echo -e "Please monitor the log trainig: $PWD/${log_train_name}.txt\n"

# only collect statistics
com="python main.py --module-config ${CONFIG} --module-model model_ood 
	    --model-forward-with-file-name --epochs 1 
	    --save-trained-name trained_network_md   >${log_train_name}.txt 2>${log_err_name}.txt "
echo ${com}
eval ${com}

echo -e "\nTraining process finished"
echo -e "Trainig log has been written to $PWD/${log_train_name}.txt"




