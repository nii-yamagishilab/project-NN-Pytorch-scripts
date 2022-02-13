#!/bin/bash
########################
# Script for confidence estimation
# 
# Usage: bash 00_demo.sh PATH
# where PATH can be model-W2V-XLSR-ft-GF/config_train_asvspoof2019/
# or other model-*/config_train_asvspoof2019 folders
#
# This script will 
#  1. install pytorch env using conda 
#  2. untar data set (toy_dataset)
#  3. run evaluation and training process
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

#####
# configurations (no need to change)
link_set=https://www.dropbox.com/sh/bua2vks8clnl2ha/AAABr4g7NPZH7GhgGSltaztVa/toy_example.tar.gz
set_name=toy_example.tar.gz
link_pretrained=https://www.dropbox.com/sh/bua2vks8clnl2ha/AADo_Djnm3qzRiS2-TJX7k0oa/project-07-asvspoof-ssl.tar
model_name=project-07-asvspoof-ssl.tar

eval_script=01_eval.sh
train_script=01_train.sh
main_script=main.py
condafile=$PWD/../../env-fs-install.sh
envfile=$PWD/../../env-fs.sh
pretrained_model=__pretrained/trained_network.pt
trained_model=./trained_network.pt

# random seeds
seeds=(1 10 100)
dirnames=(01 02 03)

# try the first random index
seed_idx=0

#####
# check
SUBDIR=${dirnames[${seed_idx}]}
if [ ! -d ${PRJDIR}/${SUBDIR} ];
then
    if [ ! -d ${PRJDIR} ];
    then
	echo "Not found: ${PRJDIR}"
    else
	echo "Fail to parse input argument"
    fi
    echo "Please use one of the following as argument"
    ls -d model-*/config_train_asvspoof2019
    exit 1;
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
# step 3 run evaluation process on toy set
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step3. scoring toy data set (using pre-trained model)${NC}"
com="bash ${eval_script} $PWD/DATA/toy_example/eval toy_eval_set 
	  $PWD/${PRJDIR}/${SUBDIR} $PWD/${PRJDIR}/${SUBDIR}/${pretrained_model} pretrained"
echo ${com}
eval ${com}

if [ -e $PWD/${PRJDIR}/${SUBDIR}/log_eval_pretrained_toy_eval_set_score.txt ];
then
    echo -e "\nCompute EERs"
    com="python 02_evaluate.py 
        ${PRJDIR}/${SUBDIR}/log_eval_pretrained_toy_eval_set_score.txt
    	DATA/toy_example/protocol.txt"
    echo ${com}
    eval ${com}
fi


#####
# step 4 training
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step4. run training process${NC}"

if [ ! -e DATA/asvspoof2019_LA/train_dev/LA_D_1000265.wav ];
then
    echo -e "${RED}ASVspoof2019 LA data is not found${NC}"
    echo -e "${RED}We will use the toy training set for demonstration only${NC}"
    echo -e "${RED}If you want to train a good model, please use ASVspoof2019 LA data${NC}"
    com="bash ${train_script} ${seeds[${seed_idx}]} config_train_toyset ${PRJDIR}/${SUBDIR}"
else
    TRAINCONFIG=`echo ${PRJDIR} | xargs -I{} basename {}`   
    com="bash ${train_script} ${seeds[${seed_idx}]} ${TRAINCONFIG} ${PRJDIR}/${SUBDIR}"
fi
echo ${com}
eval ${com}

#####
# step 5 inference using newly trained model
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step5. run evaluation process (using newly trained model)${NC}"
com="bash ${eval_script} $PWD/DATA/toy_example/eval toy_eval_set 
	  $PWD/${PRJDIR}/${SUBDIR} $PWD/${PRJDIR}/${SUBDIR}/${trained_model} trained"
echo ${com}
eval ${com}

if [ -e $PWD/${PRJDIR}/${SUBDIR}/log_eval_trained_toy_eval_set_score.txt ];
then
    echo -e "\nCompute EERs"
    com="python 02_evaluate.py 
        ${PRJDIR}/${SUBDIR}/log_eval_trained_toy_eval_set_score.txt
    	DATA/toy_example/protocol.txt"
    echo ${com}
    eval ${com}
fi
