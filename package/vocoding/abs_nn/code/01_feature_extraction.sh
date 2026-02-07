#!/usr/bin/bash
# Usage: 
#  1. Please specify the input waveform directory
#     Please specify a path to save the file list
#     Please specify a directory to save acoustic features
#
#  2. bash 01_feature_extraction.sh

PRJDIR=$PWD

# load environment
eval "$(conda shell.bash hook)"
conda activate vocoding


# ========== configurations =========
# input waveform directory
# here, we use a toy subset from asvspoof19_bona for demonstration
# you may also try $PRJDIR/../data/voxceleb2/dev
INPUTWAVDIR=$PRJDIR/../../data/asvspoof19_bona/trn

# path to save a file list
SCPLIST=$INPUTWAVDIR/../trn.lst

# output feature directory
OUTFEATDIR=$PRJDIR/data/data-feat

# AUDIO FILE EXTENSION .flac or .wav
# If you use .flac, install libsndfile pysoudfile as written in
#  project-NN-Pytorch-scripts/env.yml
WAVEXT=.wav
# ===================================

mkdir -p ${OUTFEATDIR}
if [[ ! -f ${SCPLIST} ]];
then
    echo "Create file list"
    find ${INPUTWAVDIR} -type f -name "*${WAVEXT}" -exec basename {} ${WAVEXT} \; > ${SCPLIST}
    echo "File list saved to ${SCPLIST}"
else
    echo "Use file list ${SCPLIST}"
fi


echo "Processing ${INPUTWAVDIR}"
# Run code
com="python feature_extraction.py ${SCPLIST} ${INPUTWAVDIR} ${OUTFEATDIR} ${WAVEXT}"
echo ${com}
eval ${com} 
echo "Acoustic features saved to ${OUTFEATDIR}"


# Done
