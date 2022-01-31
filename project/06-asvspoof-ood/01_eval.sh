#!/bin/bash

TRAINED_MODEL=$1
CONFIG=$2
PREFIX=$3

if [ ! -e ${TRAINED_MODEL} ];
then
    echo "Not found ${TRAINED_MODEL} for inference. Skip this step"
    exit
fi

log_file=log_output_${PREFIX}_${CONFIG}
com="python main.py --inference --model-forward-with-file-name
	    --trained-model ${TRAINED_MODEL}
	    --module-config ${CONFIG} > ${log_file} 2>log_gen_err"
echo ${com}
eval ${com}
cat ${log_file} | grep "Output," | awk '{print $2" "$3" "$4" "$5}' | sed "s:,::g" > ${log_file}_score.txt

echo -e "Score file has been generated: $PWD/${log_file}_score.txt"
