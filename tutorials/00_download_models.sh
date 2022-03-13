#!/bin/bash
# Download pre-trained models that are larger than 10MB
# 
# 
SRCDIR=data_models
if command -v wget &> /dev/null
then
    TOOL="wget -q --show-progress"
elif command -v curl &> /dev/null
then
    TOOL="curl -L -O -J"
else
    echo "Could not find a tool to download files"
    echo "Please manully modify the script and use a tool available"
fi


MODELLINK=https://zenodo.org/record/6349637/files/project-05-nn-vocoders-blow.tar
MODELNAME=project-05-nn-vocoders-blow

DIRNAME=pre_trained_blow

cd ${SRCDIR}/${DIRNAME}
${TOOL} ${MODELLINK}
if [ -e ${MODELNAME}.tar ];
then
    tar -xvf ${MODELNAME}.tar
    if [ ! -e "./__pre_trained/trained_network.pt" ];
    then
        echo "Cannot download ${MODELLINK}. Please contact the author"
    fi
else
    echo "Cannot download ${MODELLINK}. Please contact the author"
fi
cd -



MODELLINK=https://zenodo.org/record/6349637/files/project-05-nn-vocoders-waveglow.tar
MODELNAME=project-05-nn-vocoders-waveglow

DIRNAME=pre_trained_waveglow

cd ${SRCDIR}/${DIRNAME}
${TOOL} ${MODELLINK}
if [ -e ${MODELNAME}.tar ];
then
    tar -xvf ${MODELNAME}.tar
    if [ ! -e "./__pre_trained/trained_network.pt" ];
    then
        echo "Cannot download ${MODELLINK}. Please contact the author"
    fi
else
    echo "Cannot download ${MODELLINK}. Please contact the author"
fi
cd -


MODELLINK=https://zenodo.org/record/6349637/files/project-05-nn-vocoders-wavenet.tar
MODELNAME=project-05-nn-vocoders-wavenet

DIRNAME=pre_trained_wavenet

cd ${SRCDIR}/${DIRNAME}
${TOOL} ${MODELLINK}
if [ -e ${MODELNAME}.tar ];
then
    tar -xvf ${MODELNAME}.tar
    if [ ! -e "./__pre_trained/trained_network.pt" ];
    then
        echo "Cannot download ${MODELLINK}. Please contact the author"
    fi
else
    echo "Cannot download ${MODELLINK}. Please contact the author"
fi
cd -


MODELLINK=https://zenodo.org/record/6349637/files/project-05-nn-vocoders-iLPCNet.tar
MODELNAME=project-05-nn-vocoders-iLPCNet

DIRNAME=pre_trained_ilpcnet

cd ${SRCDIR}/${DIRNAME}
${TOOL} ${MODELLINK}
if [ -e ${MODELNAME}.tar ];
then
    tar -xvf ${MODELNAME}.tar
    if [ ! -e "./__pre-trained/trained_network.pt" ];
    then
        echo "Cannot download ${MODELLINK}. Please contact the author"
    fi
else
    echo "Cannot download ${MODELLINK}. Please contact the author"
fi
cd -
