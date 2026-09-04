#!/bin/bash
########################
# Script for confidence estimation
# 
# Usage: bash 00_demo.sh PATH
# where PATH can be AM-softmax-conf/config_train_asvspoof2019/
#
# This script will 
#  1. install pytorch env using conda 
#  2. untar data set
#  3. run evaluation and training process
#
# This script will use the data set in DATA/
#
# If GPU memory is less than 16GB, please reduce
#   --batch-size in 01_train.sh
# 
# bc train data will not be downloaded due to license issue
# 
########################
RED='\033[0;32m'
NC='\033[0m'

PRJDIR=$1

#####
# configurations (no need to change)
link_set=https://zenodo.org/record/6456704/files/project-06-asvspoof-ood.tar
set_name=project-06-asvspoof-ood.tar
temp_name=project-06-dataset_ood
link_pretrained=https://zenodo.org/record/6456692/files/project-06-asvspoof-ood.tar
model_name=project-06-asvspoof-ood.tar

# scripts
eval_script=01_eval_md_oe.sh
# outlier-exposure training followed by m-distance wrapper
train_script=01_train_md_oe.sh

# starting base model
pretrained_model=Softmax-energy/config_train_asvspoof2019/01/__pretrained/trained_network.pt

# name of the output model (after outlier-exposure and m-distance)
trained_model=./trained_network_oe_md.pt

eval_configs="config_test_asvspoof2019 config_test_vcc"
main_script=main.py
envfile=$PWD/../../env.sh

# random seeds
seeds=(1 10 100)
dirnames=(01 02 03)
# try the first random index
seed_idx=0
#####

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
    ls -d Softmax-*/config_train*
    ls -d AM-softmax-*/config_train*
    exit 1;
fi

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Load environment${NC}"

cd DATA 
source ${envfile}
cd ../


echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Copy pretrained model from ${pretrained_model}${NC}"
cp ${pretrained_model} ${PRJDIR}/${SUBDIR}

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step4. run training process${NC}"
# training configuration for exposure bias
TRAINCONFIG=`echo ${PRJDIR} | xargs -I{} basename {}`
# training configuration for the m-distance
TRAINCONFIG2=${TRAINCONFIG}_ood

cp ${main_script} ${PRJDIR}/${SUBDIR}
cp ${train_script} ${PRJDIR}/${SUBDIR}
cp ${TRAINCONFIG}.py ${PRJDIR}/${SUBDIR}
cp ${TRAINCONFIG2}.py ${PRJDIR}/${SUBDIR}
cd ${PRJDIR}/${SUBDIR}
pwd

# training
bash ${train_script} ${seeds[${seed_idx}]} ${TRAINCONFIG} ${TRAINCONFIG2}
cd - > /dev/null


echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step5. run evaluation process (using newly trained model)${NC}"
# run scripts
for EVALCONFIG in `echo ${eval_configs}`
do
    echo -e "\n${RED}Scoring ${EVALCONFIG}${NC}"
    cp ${main_script} ${PRJDIR}/${SUBDIR}
    cp ${EVALCONFIG}.py ${PRJDIR}/${SUBDIR}
    cp ${eval_script} ${PRJDIR}/${SUBDIR}
    cd ${PRJDIR}/${SUBDIR}
    pwd 

    # evaluation using pre-trained model
    bash ${eval_script} ${trained_model} ${EVALCONFIG} "trained"
    cd - > /dev/null
done

