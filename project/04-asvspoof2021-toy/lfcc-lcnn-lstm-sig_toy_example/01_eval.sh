#!/bin/bash
########################
# Script for evaluation
# Usage:
#   1. please check that config.py has been properly configured
#   2. please specify the trained model
#      here, we use a pre-trained model from ASVspoof2019
#   2. $: bash 01_eval.sh
########################

log_name=log_eval
trained_model=__pretrained/trained_network.pt 

echo -e "Run evaluation"
source $PWD/../../../env.sh
python main.py --inference --model-forward-with-file-name \
       --trained-model ${trained_model}> ${log_name}.txt 2>${log_name}_err.txt
cat ${log_name}.txt | grep "Output," > ${log_name}_score.txt
echo -e "Process log has been written to $PWD/${log_name}.txt"
echo -e "Score has been written to $PWD/${log_name}_score.txt"
