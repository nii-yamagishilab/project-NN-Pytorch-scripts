#!/bin/bash
########################
# Script to download files
# 
# Usage: bash download.sh
# 
# This will download a toy data set
# and the SSL model from Fairseq
########################
RED='\033[0;32m'
NC='\033[0m'

##### 
# Download toy data set
link_set=https://zenodo.org/record/7315515/files/project-09-toy_example_vocoded.tar.gz
set_name=project-09-toy_example_vocoded.tar.gz

echo -e "\n${RED}=======================================================${NC}"

cd DATA

if [ ! -e ${set_name} ];
then
    echo -e "${RED}Download and untar the toy data set${NC}"
    wget -q --show-progress ${link_set}
else
    echo -e "Use downloaded ${set_name}"
fi

if [ -e ${set_name} ];
then
    if [ ! -d toy_example ];
    then
	tar -xzf ${set_name}
    fi
else
    echo -e "\nCannot download ${set_name}"
fi

cd ..

#####
# Download continually trained SSL model
# pre-trained will be downloaded using 01_download_pretrainedcm.sh

link_set=https://zenodo.org/record/8336949/files/wav2vec_ft2_vox_vocoded.pt
set_name=wav2vec_ft2_vox_vocoded.pt

cd SSL_pretrained

echo -e "\n${RED}=======================================================${NC}"
if [ ! -e ${set_name} ];
then
    echo -e "${RED}Download and untar the toy data set${NC}"
    wget -q --show-progress ${link_set}
else
    echo -e "Use downloaded ${set_name}"
fi

if [ ! -e ${set_name} ];
then
    echo -e "\nCannot download ${set_name}"
fi

cd ..
