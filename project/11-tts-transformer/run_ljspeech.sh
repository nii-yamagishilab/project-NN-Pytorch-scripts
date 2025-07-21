#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00

# name of the environment
NAME_ENV=torchnn

srcprj=$PWD
source ~/.bashrc
conda activate ${NAME_ENV}

echo "Submitted job ID: $JOB_ID"

cd ${srcprj}
command="python main.py configs/sys_ljspeech.yaml"

echo ${command}
eval ${command}
