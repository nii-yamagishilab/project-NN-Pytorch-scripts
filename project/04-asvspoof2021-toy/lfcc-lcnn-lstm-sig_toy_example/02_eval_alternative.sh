#!/bin/bash
########################
# Script for quick evaluation
# 01_eval.sh requires configuration of config.py,
# which further requires preparation of test.lst
# and protocol.txt.
#
# It is convenient when doing numerious experiments
# on the same data set but troublesome when evaluating
# different evaluationg sets.
#
# This script assumes one model has been trained, 
# and it just needs the directory to the waveforms of
# the evaluation set. It will automatically produce 
# the test.lst and config config_auto.py
#
# Usage:
# bash 02_eval_alternative.sh <wav_dir> <name_eval> <trained_model>
# <wav_dir>: abosolute path to the directory of eval set waveforms
# <name_eval>: name of the evaluation set, any arbitary string
# <trained_model>: path to the trained model file
########################

if [ "$#" -ne 3 ]; then
    echo -e "Invalid input arguments. Please check doc of the script, and use:"
    echo -e "bash 02_eval_alternative.sh <wav_dir> <name_eval> <trained_model>"
    exit
fi

# path to the directory of waveforms
eval_wav_dir=$1
# name of the evaluation set (any string)
eval_set_name=$2
# path to the trained model
trained_model=$3


echo -e "Run evaluation"

# step1. load python environment
source $PWD/../../../env.sh

# step2. prepare test.lst
ls ${eval_wav_dir} | xargs -I{} basename {} .wav > ${eval_set_name}.lst

# step3. export for config_auto.py
export TEMP_DATA_NAME=${eval_set_name}
export TEMP_DATA_DIR=${eval_wav_dir}

# step4. run evaluation
log_name=log_eval_${eval_set_name}
python main.py \
       --inference \
       --model-forward-with-file-name \
       --trained-model ${trained_model} \
       --module-config config_auto > ${log_name}.txt 2>${log_name}_err.txt

cat ${log_name}.txt | grep "Output," > ${log_name}_score.txt

echo -e "Process log has been written to $PWD/${log_name}.txt"
echo -e "Score has been written to $PWD/${log_name}_score.txt"

# step5. delete intermediate files
#  this script is created in step2
rm ${eval_set_name}.lst
#  this is created by the python code (for convenience)
rm ${eval_set_name}_utt_length.dic
