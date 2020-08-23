#!/bin/sh
# ---- script to extract Mel-spectrogram from waveforms
# Usage:
# 1. change WAVDIR and OUTDIR
# 2. sh 00_batch.sh
# Requirement:
# python, numpy

# directory of input waveform (16kHz)
WAVDIR=$1

# output directory to store extracted Mel-spectrogram
OUTDIR=$2


### 
mkdir -p ${OUTDIR}

find ${WAVDIR} -type f -name "*.wav" > ${OUTDIR}/file.lst

cat ${OUTDIR}/file.lst | ${PARALLELPATH} python3 sub_get_mel.py ${WAVDIR}/{/.}.wav ${OUTDIR}/{/.}.mfbsp

rm ${OUTDIR}/file.lst
