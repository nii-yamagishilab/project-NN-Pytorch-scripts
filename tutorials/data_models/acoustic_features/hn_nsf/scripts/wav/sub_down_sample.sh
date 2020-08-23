#!/bin/sh
# ------- Use sox to downsample
# Before use, please specify SOXPATH
# Usage: sub_down_sample.sh input_wav output_wav target_sampling_rate
#
#SOXPATH=sox

# Following suggetions in http://sox.sourceforge.net/Docs/FAQ
# For info http://sox.sourceforge.net/sox.html
# 1. no need to add dither explicitly, because SoX automatically adds TPDF dither when the output bit-depth is less than 24
# 2. dither -s cannot be used for 24kHz: Noise-shaping (only for certain sample rates) can be selected with -s 
# 3. rate -I: Phase setting: if resampling to < 40k, use intermediate phase (-I)
# 4. using default HQ, rather than VHQ because bit = 16bit

${SOXPATH} $1 -b 16 $2 rate -I $3
