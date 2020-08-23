#!/bin/bash

source env.sh

# step1 waveform processing
STEP1FLAG=1
# step2 Mel-spectrogram
STEP2FLAG=1
# step3 F0
STEP3FLAG=1

# input raw wavevorm
PRJDIR=$PWD/../

WAVDIR=${PRJDIR}/wav_16k
NORMED_WAVDIR=${PRJDIR}/wav_16k_norm
SAMP=16000
WAV_FORMAT=wav

# mel
WAVDIR_FOR_MEL=${WAVDIR}
MEL_OUTPUTDIR=${PRJDIR}/melspec
# F0
WAVDIR_FOR_F0=${WAVDIR}
F0_OUTPUTDIR=${PRJDIR}/f0

# 
if [ ${STEP1FLAG} -gt 0 ]; then
    cd wav
    sh 00_batch.sh ${WAVDIR} ${NORMED_WAVDIR} ${SAMP} ${WAV_FORMAT}
    cd ../
fi


if [ ${STEP2FLAG} -gt 0 ]; then
    cd mel
    sh 00_batch.sh ${WAVDIR_FOR_MEL} ${MEL_OUTPUTDIR}
    cd ../
fi

if [ ${STEP3FLAG} -gt 0 ]; then
    cd f0/f0_yaapt
    sh 00_batch.sh ${WAVDIR_FOR_F0} ${F0_OUTPUTDIR}
    cd ../
fi
