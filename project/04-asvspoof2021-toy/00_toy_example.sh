#!/bin/bash
########################
# Script for toy_example
# This script will 
#  1. install pytorch env using conda 
#  2. untar toy data set
#  3. run evaluation and training process
#
# This script will use the toy data set in
# DATA/toy_example. 
#
# If GPU memory is less than 16GB, please reduce
#   --batch-size in 00_train.sh
########################
RED='\033[0;32m'
NC='\033[0m'

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step1. install conda environment${NC}"
# create conda environment
bash 01_conda.sh

# untar the toy-example data
echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step2. untar toy data set${NC}"
cd DATA
if [ ! -e "toy_example.tar.gz" ];
then
    echo "Downloading toy data set"
    wget -q --show-progress https://zenodo.org/record/6456704/files/project-04-toy_example.tar.gz
    mv project-04-toy_example.tar.gz toy_example.tar.gz
fi
tar -xzf toy_example.tar.gz
cd ..

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step3. run evaluation process (using pre-trained model)${NC}"
# run scripts
cd lfcc-lcnn-lstm-sig_toy_example
# evaluation using pre-trained model
bash 01_eval.sh

echo -e "\n${RED}=======================================================${NC}"
echo -e "${RED}Step4. run training process (using pre-trained model)${NC}"
#training, with pre-trained model as initialization
bash 00_train.sh

