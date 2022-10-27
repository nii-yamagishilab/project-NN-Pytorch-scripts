#!/bin/bash
########################
# Script for demontration
# 
# Usage: bash 00_demo.sh PATH CONFIG RAND_SEED
# where 
#   PATH can be model-LFCC-LLGF
#        or other model-* folders
#   CONFIG can be config_train_toyset
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

#####
# configurations (no need to change)
link_set=https://zenodo.org/record/6456704/files/project-04-toy_example.tar.gz
set_name=project-04-toy_example.tar.gz

link_pretrained=https://zenodo.org/record/6456692/files/project-07-asvspoof-ssl.tar
model_name=project-07-asvspoof-ssl.tar

score_script=02_score.sh
train_script=01_train.sh
main_script=main.py
condafile=$PWD/../../env-fs-install.sh
envfile=$PWD/../../env-fs.sh
pretrained_model=__pretrained/trained_network.pt
trained_model=./trained_network.pt

# we will use this toy data set for demonstration
configs_name=${CONFIG}
PRJDIR=${PRJDIR}/${CONFIG}

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
echo -e "${RED}Step1. load conda environment${NC}"
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
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step2. download and untar the data set, pre-trained model${NC}"
cd DATA
if [ ! -e ${set_name} ];
then
    echo -e "\nDownloading toy data set"
    wget -q --show-progress ${link_set}
    tar -xzf ${set_name}
fi

if [ ! -d asvspoof2019_LA ];
then
    echo -e "\nCopy place holder for ASVspoof 2019 LA"
    cp -rP ../../03-asvspoof-mega/DATA/asvspoof2019_LA/ ./
fi
cd ..

if [ ! -e ${model_name} ];
then
    echo -e "\nDowloading pre-trained model"
    wget -q --show-progress ${link_pretrained}
    tar -xf ${model_name}
fi

echo -e "\nDowloading pre-trained SSL models"
bash 01_download_ssl.sh


#####
# step 3 training
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step3. training using ${CONFIG} ${NC}"

com="bash ${train_script} ${RAND_SEED} ${CONFIG} ${PRJDIR}/${SUBDIR}"
echo ${com}
eval ${com}

#####
# step 4 inference 
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step4. scoring using the toy set${NC}"
com="bash ${score_script} $PWD/DATA/toy_example/eval toy_eval_set 
	  $PWD/${PRJDIR}/${SUBDIR} $PWD/${PRJDIR}/${SUBDIR}/${trained_model} 
	  trained"
echo ${com}
eval ${com}

#####
# step 5 inference
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step5. scoring using the toy set and models trained by author${NC}"
com="bash ${score_script} $PWD/DATA/toy_example/eval toy_eval_set
          $PWD/${PRJDIR}/../config_train_asvspoof2019/01
          $PWD/${PRJDIR}/../config_train_asvspoof2019/01/__pretrained/${trained_model}
          pretrained"
echo ${com}
eval ${com}
