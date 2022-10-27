#!/bin/bash
########################
# Script for demonstration
# 
# Usage: bash 00_demo.sh PATH CONFIG RAND_SEED
# where 
#   PATH can be model-AL-NegE
#        or other model-* folders
#   CONFIG can be config_AL_train_toyset
#       if you prepare other config, you can use them as well
#   RAND_SEED can be 01, 02 or any other numbers
# 
# This script will 
#  1. install pytorch env using conda 
#  2. untar data set (toy_dataset)
#  3. run training and scoring
#
# This script will use the data set in DATA/
#
# If GPU memory is less than 16GB, please reduce
#   --batch-size in 01_train.sh
# 
########################
RED='\033[0;32m'
NC='\033[0m'

PRJDIR=$1
CONFIG=$2
RAND_SEED=$3
###
# Configurations
###

# we will use this toy data set for demonstration
configs_name=${CONFIG}
PRJDIR=${PRJDIR}/${CONFIG}

# we will use scripts below to train and score 
train_script=01_train.sh
score_script=02_score.sh

# we will use the following training configurations for active learning
# number of samples selected in each AL cycle
AL_NUMSAMPLE=16
# maximum number of AL cycles
AL_NUMCYCLE=5
# number of training epoch in each AL cycle
AL_EPOCH_P_CYCLE=2
# learning trate
AL_LR=0.000001
# path to the seed model (will be downloaded in 01_download.sh)
PRETRAINED=$PWD/seed_model/w2v-ft-gf_trained_network.pt


###
# Configurations fixed
##
main_script=main.py

condafile=$PWD/../../env-fs-install.sh
envfile=$PWD/../../env-fs.sh
trained_model=trained_network


#####
# check
SUBDIR=${RAND_SEED}
if [ ! -d ${PRJDIR}/${SUBDIR} ];
then
    mkdir -p ${PRJDIR}/${SUBDIR}
fi

#####
# step 1
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step1. Preparation: load environment and get toy data${NC}"
# create conda environment
bash ${condafile} || exit 1;

# load env.sh. It must be loaded inside a sub-folder $PWD/../../../env.sh
# otherwise, please follow env.sh and manually load the conda environment and 
# add PYTHONPATH
cd DATA 
source ${envfile} || exit 1;
cd ../

#####
# step 2 download 
bash 01_download.sh

#####
# step 3 training
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step3. training on toy set${NC}"

com="bash ${train_script} ${RAND_SEED}
	  ${configs_name} ${PRJDIR}/${SUBDIR} 
	  ${AL_NUMSAMPLE} ${AL_NUMCYCLE} ${AL_EPOCH_P_CYCLE} 
	  ${AL_LR} ${PRETRAINED}"
echo ${com}
eval ${com}

#####
# step 4 inference using trained model
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step4. score the toy data set${NC}"

# we will use the trained model from each cycle
for cycle in `ls ${PRJDIR}/${SUBDIR}/${trained_model}*.pt | grep cycle`
do
    cycle=`echo ${cycle} | xargs -I{} basename {}`
    if [ -e $PWD/${PRJDIR}/${SUBDIR}/${cycle} ];
    then
	echo -e "${RED}Scoring using ${cycle}${NC}"
	com="bash ${score_script} $PWD/DATA/toy_example/eval toy_eval_set 
    	  	  $PWD/${PRJDIR}/${SUBDIR} $PWD/${PRJDIR}/${SUBDIR}/${cycle}
	  	  trained_${cycle}"
	echo ${com}
	eval ${com}
    fi
done

