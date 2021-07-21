#!/bin/bash

MODELLINK=https://www.dropbox.com/sh/bua2vks8clnl2ha/AAB0vGtbIG8yrNJqE06e6Uqsa/project-02-asvspoof.tar.gz
MODELNAME=project-02-asvspoof

# download pre-trained model and toy test set data from dropbox
if [ ! -e "./lfcc-lcnn-a-softmax/__pretrained/trained_network.pt" ];then
    echo -e "${RED}Downloading pre-trained model${NC}"
    wget -q --show-progress ${MODELLINK}

    if [ -e "./${MODELNAME}.tar.gz" ];then	
	tar -xzf ${MODELNAME}.tar.gz
	rm ${MODELNAME}.tar.gz
    else
	echo "Cannot download ${MODELLINK}. Please contact the author"
    	exit
    fi
fi
