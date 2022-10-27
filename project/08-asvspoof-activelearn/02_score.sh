#!/bin/bash
########################
# Script for scoring waveforms
#
# Usage:
# bash 01_scoring.sh <wav_dir> <name_eval> <model_dir> <trained_model> <tag>
# 
# <wav_dir>: path to the directory of eval set waveforms
#            please use full path, not relative path
# <name_eval>: name of the evaluation set, you can use any name
# <model_dir>: path to the project of a specific CM model 
#              it should contain main.py and model.py
#            please use full path, not relative path
# <trained_model>: path to the trained model file
# <tag>: any string, which will be attached to the name of the score file
########################
RED='\033[0;32m'
NC='\033[0m'

if [ "$#" -ne 5 ]; then
    echo -e "Invalid input arguments. Please check the doc and try:"
    echo -e "bash 01_scoring.sh <wav_dir> <name_eval> <model_dir> <trained_model> <tag>"
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
    cp ./config_auto.py ${model_dir}
    cp ./main.py ${model_dir}
    cp ${model_dir}/../../model.py ${model_dir}
    cd ${model_dir}
else
    echo "Cannot find ${model_dir}"
    exit 1;
fi

### 
# step1. prepare test.lst
ls ${eval_wav_dir} | xargs -I{} basename {} .wav > ${eval_set_name}.lst

###
# step2. export variables for config_auto.py
echo -e "export TEMP_DATA_NAME=${eval_set_name}"
echo -e "export TEMP_DATA_DIR=${eval_wav_dir}"
export TEMP_DATA_NAME=${eval_set_name}
export TEMP_DATA_DIR=${eval_wav_dir}

###
# step3. run scoring
log_name=log_eval_${tag}_${eval_set_name}
com="python main.py --inference 
	    --ignore-cached-file-infor
	    --model-forward-with-file-name 
	    --trained-model ${trained_model} 
	    --module-config config_auto > ${log_name} 2>${log_name}_err"
echo ${com}
eval ${com}

###
# step4. collect results
cat ${log_name} | grep "Output," | awk '{print $2" "$4" "$5}' | sed "s:,::g" > ${log_name}_score.txt

TEMPVAL=`cat ${log_name}_score.txt | wc -l`
if [ ${TEMPVAL} -gt 0 ];
then
    echo -e "\n${RED}Score has been written to $PWD/${log_name}_score.txt${NC}"
else
    echo -e "\n${RED}Failed. Please check $PWD/${log_name}_err and $PWD/${log_name}${NC}"
fi

rm ${eval_set_name}.lst

# done
