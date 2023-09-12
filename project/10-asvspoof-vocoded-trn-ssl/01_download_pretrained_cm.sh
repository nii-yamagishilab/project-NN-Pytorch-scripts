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


MODELNAME=$1
# if input $PWD/model-ID-P3/trained-for-paper-/01
# get the name model-ID-P3
MODELNAME=`echo ${MODELNAME} | awk -F '/' '{print $(NF-2)}'`
##### 

# Download trained models in the paper
link_set=https://zenodo.org/record/8337778/files/project10-cm-ssl-${MODELNAME}.tar
set_name=project10-cm-ssl-${MODELNAME}.tar


echo -e "\n${RED}=======================================================${NC}"

if [ ! -e ${set_name} ];
then
    echo -e "${RED}Download and untar the trained models${NC}"
    wget -q --show-progress ${link_set}
fi

if [ -e ${set_name} ];
then
    if [ ! -d ${MODELNAME}/trained-for-paper ];
    then
	tar -xvf ${set_name}
    fi
else
    echo -e "\nCannot download ${set_name}"
fi
