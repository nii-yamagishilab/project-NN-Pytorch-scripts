#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=3:00:00

# for slurm experiments
expdir=$PWD

source ~/.bashrc
conda activate whisperx

# we need cuda and cudnn separately
module load cuda/12.8.0
module load cudnn/9.8.0 

cd ${expdir}
python whisperx_align.py $1 $2 $3
