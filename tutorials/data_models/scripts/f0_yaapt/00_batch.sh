#!/bin/sh
# ---- batch script to extract F0
# Usage:
#   1. config INPUT_WAV_DIR and OUTPUT_F0_DIR
#   2. run sh 00_batch.sh
# No dependency required

# Directory of input waveform
INPUT_WAV_DIR=$1 #/home/smg/wang/WORK/WORK/WORK/voice-privacy-challenge-2019/TESTDATA/LibriSpeech/wav/wav16k_norm
# Directory to store output F0
OUTPUT_F0_DIR=$2 #/home/smg/wang/WORK/WORK/WORK/voice-privacy-challenge-2019/TESTDATA/LibriSpeech/f0/f0_yaapt

# SAMPLE list
#LIST=/home/smg/wang/WORK/WORK/WORK/voice-privacy-challenge-2019/TESTDATA/LibriSpeech/scripts/get_sample/sample.lst
LIST=${OUTPUT_F0_DIR}/file.lst

mkdir -p ${OUTPUT_F0_DIR}
ls ${INPUT_WAV_DIR} | grep wav > ${LIST}
cat ${LIST} | ${PARALLELPATH} python3 get_f0.py ${INPUT_WAV_DIR}/{/.}.wav ${OUTPUT_F0_DIR}/{/.}.f0
rm ${LIST}
