#!/bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=24:00:00

# name of the environment
NAME_ENV=torchnn

srcprj=$PWD
source ~/.bashrc
conda activate ${NAME_ENV}

echo "Submitted job ID: $JOB_ID"

cd ${srcprj}
command="torchrun --standalone --nproc_per_node=2 main.py configs/sys.yaml"

echo ${command}
eval ${command}
