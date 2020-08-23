#!/bin/sh
# This script downsample the waveform to 16kHz and normalize the waveform amplitude
# Usage:
#  1. Modify SOURCE_DIR, TARGET_DIR
#  2. Specify the path of SOX and SV56 in sub_down_sample.sh and sub_sv56.sh
#     SOX can be downloaded from https://sourceforge.net/projects/sox/
#     SV56 can be downloaded from https://github.com/openitu/STL
#  3. sh 00_batch.sh
# Requirement:
#  sox, sv56


# Directory of input waveforms
SOURCE_DIR=$1

# Directory to store the processed waveforms
TARGET_DIR=$2

# Sampling rate to be used
SAMP=$3

# Extension
EXT=$4

#PARALLEL=${PARALLELPATH}
##########
TMP=${TARGET_DIR}_TMP
mkdir -p ${TMP}
mkdir -p ${TARGET_DIR}

# step0. create file list
find ${SOURCE_DIR}/ -type f -o -type l -name "*.${EXT}"  > ${TARGET_DIR}/file.lst
echo `wc -l ${TARGET_DIR}/file.lst`

# step1. down-sampling
cat ${TARGET_DIR}/file.lst | ${PARALLELPATH} sh sub_down_sample.sh {} ${TMP}/{/.}.wav ${SAMP}

# step2.
cat ${TARGET_DIR}/file.lst | ${PARALLELPATH} bash sub_sv56.sh ${TMP}/{/.}.wav ${TARGET_DIR}/{/.}.wav

rm ${TARGET_DIR}/file.lst
ls ${TMP} | ${PARALLELPATH} rm ${TMP}/{}
rm -r ${TMP}
