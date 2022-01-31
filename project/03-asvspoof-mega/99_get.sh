#!/bin/bash

SRC=~/WORK/WORK/WORK/pytorch-project-2020/proj-asvspoof/proj-experiment-01

for feattype in lfcc spec2 lfb
do
    for nettype in fixed attenion lstmsum
    do
	for output in am oc p2s sig
	do
	    if [[ ${feattype} == 'lfb' && ${output} == 'am' ]];
	    then
		SRC_TMP=${SRC}/lfb2-lcnn-${nettype}-${output}
	    else
		SRC_TMP=${SRC}/${feattype}-lcnn-${nettype}-${output}
	    fi
	    
	    if [ ${nettype} == 'attenion' ];
	    then
		TAR_TMP=${feattype}-lcnn-attention-${output}
	    else
		TAR_TMP=${feattype}-lcnn-${nettype}-${output}
	    fi
	    
	    echo ${SRC_TMP} ${TAR_TMP}
	    for run in 01 02 03 04 05 06
	    do
		mkdir -p ${TAR_TMP}/${run}
		rm ${TAR_TMP}/${run}/*
		cat ${SRC_TMP}/${run}/00_run.sh | head -n 1 >  ${TAR_TMP}/${run}/00_train.sh
		cat ${SRC_TMP}/${run}/00_run.sh | head -n 2 | tail -n 1 >  ${TAR_TMP}/${run}/01_eval.sh
		cp ${SRC_TMP}/${run}/model.py ${TAR_TMP}/${run}
		
		mkdir ${TAR_TMP}/${run}/__pretrained
		cp ${SRC_TMP}/${run}/trained_network.pt ${TAR_TMP}/${run}/__pretrained
		cp ${SRC_TMP}/${run}/log_output_testset ${TAR_TMP}/${run}/__pretrained
	    done

	done
    done
done

tar -czvf pretrained.tar */*/__pretrained
rm -r */*/__pretrained


exit


SRC=~/WORK/WORK/WORK/pytorch-project-2020/proj-asvspoof/proj-experiment-01/99_export_github/03-asvspoof-mega
for dir in `ls -d *-lcnn-*`
do
    for run in 01 02 03 04 05 06
    do
	cp ${SRC}/${dir}/${run}/00_train.sh ${dir}/${run}
    done
done
