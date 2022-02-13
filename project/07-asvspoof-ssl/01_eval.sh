#!/bin/bash
########################
# Script for scoring waveforms
#
# Usage:
# bash 02_evaluate.sh <wav_dir> <name_eval> <model_dir> <trained_model> <tag>
# 
# <wav_dir>: absolute path to the directory of eval set waveforms
# <name_eval>: name of the evaluation set, any arbitary string
# <model_dir>: absolute path to the project of a specific CM model 
# <trained_model>: absolute path to the trained model file
# <tag>: any string, which will be attached to the output log file
########################

if [ "$#" -ne 5 ]; then
    echo -e "Invalid input arguments. Please check doc of the script, and use:"
    echo -e "bash 02_evaluate.sh <wav_dir> <name_eval> <model_dir> <trained_model> <tag>"
    exit 1;
fi

# path to the directory of waveforms
eval_wav_dir=$1
# name of the evaluation set (any string)
eval_set_name=$2
# path to the CM project 
model_dir=$3
# path to the trained model
trained_model=$4
# tag to the output score file
tag=$5

#### 
# copy files
echo -e "\nRun evaluation\n"


if [ -d ${model_dir} ];
then
    cp ./main.py ${model_dir}
    cp ./config_auto.py ${model_dir}
    cp ${model_dir}/../../model.py ${model_dir}
    
    cd ${model_dir}
else
    echo "Cannot find ${model_dir}"
    exit 1;
fi


### 
# step1. load python environment
# assume that this will be done in 00_demo.sh
#source $PWD/../../../../../env2.sh


### 
# step2. prepare test.lst
ls ${eval_wav_dir} | xargs -I{} basename {} .wav > ${eval_set_name}.lst

###
# step3. export variables for config_auto.py
echo -e "export TEMP_DATA_NAME=${eval_set_name}"
echo -e "export TEMP_DATA_DIR=${eval_wav_dir}"
export TEMP_DATA_NAME=${eval_set_name}
export TEMP_DATA_DIR=${eval_wav_dir}

###
# step4. run evaluation
log_name=log_eval_${tag}_${eval_set_name}
com="python main.py --inference 
	    --ignore-cached-file-infor
	    --model-forward-with-file-name 
	    --trained-model ${trained_model} 
	    --module-config config_auto > ${log_name} 2>${log_name}_err"
echo ${com}
eval ${com}

cat ${log_name} | grep "Output," | awk '{print $2" "$3" "$4" "$5}' | sed "s:,::g" > ${log_name}_score.txt

echo -e "\nProcess log has been written to $PWD/${log_name}"

TEMPVAL=`cat ${log_name}_score.txt | wc -l`
if [ ${TEMPVAL} -gt 0 ];
then
    echo -e "\nScore has been written to $PWD/${log_name}_score.txt"
else
    echo -e "\nFailed. Please check ${log_name}_err and ${log_name}"
fi

###
#  step5. delete intermediate files
#  this script is created in step2
rm ${eval_set_name}.lst
#  this is created by the python code (for convenience)
#rm ${eval_set_name}_utt_length.dic

# done
