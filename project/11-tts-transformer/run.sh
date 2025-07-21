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


#cd /gs/bs/tgh-25IAC/ud03523/WORK/project-wedefense/wedefense/egs/detection/asvspoof5/v13_ssl_aasist
#config=conf/XLSR_AASIST_v1.yaml
#exp_dir=exp/`echo ${config} | xargs -I{} basename {}`_${JOB_ID}
#rm -r ${exp_dir}

cd ${srcprj}
command="python main.py configs/sys.yaml"

echo ${command}
eval ${command}
