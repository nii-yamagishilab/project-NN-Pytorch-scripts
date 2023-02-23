#!/bin/bash
########################
# Demo script for scoring using pretrained models
# 
# Usage: bash 00_demo_pretrained.sh MODEL_DIR TESTSET_DIR TESTSET_NAME
# where 
#   MODEL_DIR can be $PWD/model-ID-7/trained-for-paper/01
#        or other model-* folders
#        It must be a path to the specific model with specific training 
#        configuration folder and a specific random seed
# 
#   TESTSET_DIR is the path to the directory of the test set waveforms
#        for example $PWD/DATA/toy_example_vocoded/eval 
#
#   TESTSET_NAME is the name of the test set
#        it can be anything text string 
#
# This script will 
#  1. install pytorch env using conda 
#  2. download the toy data set for demonstration (if necessary)
#  3. run scoring
#
# This script will use the data set in DATA/
# 
########################
RED='\033[0;32m'
NC='\033[0m'

PRJDIR=$1


if [ "$#" -ne 3 ]; then
    TESTSET_DIR=$PWD/DATA/toy_example_vocoded/eval
    TESTSET_NAME=toy_eval_set
    echo -e "Use toy test set for demonstration."
else
    TESTSET_DIR=$2
    TESTSET_NAME=$3
fi

###
# Configurations
###

# we will use scripts below to train and score 
score_script=02_score.sh

###
# Configurations fixed
##
main_script=main.py
condafile=$PWD/../../env-fs-install.sh
envfile=$PWD/../../env-fs.sh
trained_model=trained_network

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
#bash 01_download_pretrained.sh

#####
# step 3 inference using trained model
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step3. score the toy data set using models trained by Xin${NC}"


trainedmodel=${trained_model}.pt
if [ -e ${PRJDIR}/${trainedmodel} ];
then
    com="bash ${score_script} 
    	      ${TESTSET_DIR}
	      ${TESTSET_NAME}
	      $PRJDIR
	      $PRJDIR/${trainedmodel}
	      trained"
    echo ${com}
    eval ${com}
fi

