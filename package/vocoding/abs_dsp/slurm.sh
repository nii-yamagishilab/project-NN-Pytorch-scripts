#!/bin/sh
#$ -cwd    
#$ -l cpu_4=1  
#$ -l h_rt=12:00:00

# Example demonstration
# 1. download the toy_example data follow ../README.md
# 2. install dependency: librosa, soundfile, pyworld, parallel
# 3. change the conda environment to yours
# 4. bash slurm.sh
# 
# To use your own data
# 1. change INPUTDIR, INPUTLIST and OUTDIR, EXT
# 3. bash slurm.sh
#    The output will be in OUTPUTDIR

prjdir=$PWD

# activate environment
source ~/.bashrc
conda activate vocoding
cd ${prjdir}

####
# INPUT folder
INPUTDIR=$PWD/../data/asvspoof19_bona/trn
# List of filename
#   file1
#   file2
#   ...
INPUTLIST=$PWD/../data/asvspoof19_bona/trn.lst

# Output folder
OUTDIR=$PWD/output_vocoded
# Format
EXT=.wav
####

if [[ ! -f ${INPUTLIST} ]];
then
    echo "Create file list"
    find ${INPUTDIR} -type f -name "*${EXT}" -exec basename {} ${EXT} \; > ${INPUTLIST}
fi

# griffin lim
outtemp=${OUTDIR}_gl
mkdir ${outtemp}
cat ${INPUTLIST} | parallel python abs_dsp.py ${INPUTDIR}/{.}${EXT} ${outtemp}/{.}${EXT} griffinlim

# world
outtemp=${OUTDIR}_wo
mkdir ${outtemp}
cat ${INPUTLIST} | parallel python abs_dsp.py ${INPUTDIR}/{.}${EXT} ${outtemp}/{.}${EXT} pyworld