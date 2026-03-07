#!/bin/sh
#$ -cwd    
#$ -l cpu_4=1  
#$ -l h_rt=4:00:00

# for example: 
# qsub -g ${gid} -o log_waveglow -e log_waveglow_err run.sh waveglow

tmpdir=$PWD
source ~/.bashrc
conda activate vocoding
cd ${tmpdir}

# ===== configurations =====
# a tag that will be added in front of the output file name
tag=VOC
# real audio data folder
bon_folder=$PWD/data/real
# json (with time alingment) folder
json_folder=$PWD/data/json
# vocoded data folder
voc_folder=$PWD/data/voc
# file list
filelst=$PWD/data/file.lst
# file extension
ext=.flac

# output folder to save partially replaced wav
outwav=$PWD/data_output/audio
# output folder to save raw json (without vocoded time tag)
outjson=$PWD/data_output/partial_json_raw
# output folder to save raw json (with vocoded time tag !!!!!!)
outjson_tagged=$PWD/data_output/partial_json
# ===========================


mkdir -p ${outwav}
mkdir -p ${outjson}
mkdir -p ${outjson_tagged}
com="python main.py ${bon_folder} ${voc_folder} ${json_folder} ${outwav} ${outjson} ${filelst} ${tag}_ ${ext}"
echo ${com}
eval ${com}

# convert json and add vocoder tag
ls ${outjson} | parallel python add_tag_json.py ${outjson}/{} ${outjson_tagged}/{}

echo "Output json saved to $outjson_tagged"
echo "Output audio saved to ${outwav}"