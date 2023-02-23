#!/bin/bash
########################
# Script to download files
# 
# Usage: bash download_pretrained.sh
# 
# This will download pretrained models
########################
RED='\033[0;32m'
NC='\033[0m'

##### 
# Download trained models in the paper
link_set=https://zenodo.org/record/7668535/files/project09-cm-vocoded-trained.tar
set_name=project09-cm-vocoded-trained.tar

echo -e "\n${RED}=======================================================${NC}"

if [ ! -e ${set_name} ];
then
    echo -e "${RED}Download and untar the trained models${NC}"
    wget -q --show-progress ${link_set}
fi

if [ -e ${set_name} ];
then
    if [ ! -d model-ID-2/trained-for-paper ];
    then
	tar -xvf ${set_name}
    fi
else
    echo -e "\nCannot download ${set_name}"
fi
