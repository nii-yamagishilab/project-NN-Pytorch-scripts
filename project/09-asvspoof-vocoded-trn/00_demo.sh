#!/bin/bash
########################
# Demo script for training using vocoded data
# 
# Usage: bash 00_demo.sh PATH CONFIG RAND_SEED
# where 
#   PATH can be model-ID-7
#        or other model-* folders
#   CONFIG can be config_train_toyset_paired
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

if [ "$#" -ne 3 ]; then
    echo -e "Invalid input arguments. Please check the doc of script."
    exit 1;
fi

###
# Configurations
###

# we will use this toy data set for demonstration
configs_name=${CONFIG}
PRJDIR=${PRJDIR}/${CONFIG}

# we will use scripts below to train and score 
train_script=01_train.sh
score_script=02_score.sh

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
echo -e "${RED}Step3. run training process on toy set${NC}"

if [[ "$1" == "model-ID-2" ]];
then
    # baseline method use a different configuration
    flag=True
else
    flag=False
fi
com="bash ${train_script} ${RAND_SEED} ${configs_name} ${PRJDIR}/${SUBDIR} ${flag}"
echo ${com}
eval ${com}

#####
# step 4 inference using trained model
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step4. score the toy data set${NC}"

trainedmodel=${trained_model}.pt
if [ -e $PWD/${PRJDIR}/${SUBDIR}/${trainedmodel} ];
then
    com="bash ${score_script} $PWD/DATA/toy_example_vocoded/eval toy_eval_set 
         $PWD/${PRJDIR}/${SUBDIR} $PWD/${PRJDIR}/${SUBDIR}/${trainedmodel}
	 trained"
    echo ${com}
    eval ${com}
fi
