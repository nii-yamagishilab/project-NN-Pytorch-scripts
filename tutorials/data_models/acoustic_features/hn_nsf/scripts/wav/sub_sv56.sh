#!/bin/bash
# --- use sv56 to normalize waveform amplitude
# Before use, please specify SOX and SV56
# Usage: sh sub_sv56.sh input_wav output_wav

# level of normalization
LEV=26

# path to sox and sv56demo
SOX=${SOXPATH}
SV56=${SV56PATH}

# input file path
file_name=$1

# output file path
OUTPUT=$2

if [ -e ${file_name} ];
then
    # basename
    basename=`basename ${file_name} .wav`
    # input file name
    INPUT=${file_name}

    # raw data name
    RAWORIG=${OUTPUT}.raw
    # normed raw data name
    RAWNORM=${OUTPUT}.raw.norm
    # 16bit wav
    BITS16=${OUTPUT}.16bit.wav
    
    if ! type "${SOX}" &> /dev/null; then
	echo "${SOX} not in path"
	exit
    fi
    
    SAMP=`${SOX} --i -r ${INPUT}`
    BITS=`${SOX} --i -b ${INPUT}`

    # make sure input is 16bits int
    if [ ${BITS} -ne 16 ];
    then
	echo "${file_name} is not 16bit"
    else
	${SOX} ${INPUT} ${RAWORIG}
    fi

    if ! type "${SV56}" &> /dev/null;
    then
	echo "${SV56} is not in path"
    else
	echo "convert and normed ${INPUT} to ${OUTPUT}"
	# norm the waveform
	${SV56} -q -sf ${SAMP} -lev -${LEV} ${RAWORIG} ${RAWNORM} > log_sv56 2>log_sv56

	# convert
	${SOX} -t raw -b 16 -e signed -c 1 -r ${SAMP} ${RAWNORM} ${OUTPUT}

	rm ${RAWNORM}
	rm ${RAWORIG}
    fi
else
    echo "not found ${file_name}"
fi

