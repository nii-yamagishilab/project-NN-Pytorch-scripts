#!/bin/sh

MODELLINK=https://zenodo.org/record/6456692/files/project-05-nn-vocoders-blow.tar
MODELNAME=project-05-nn-vocoders-blow

DATANAME=vctk-blow
ENVFILE=../../../env.sh

RED='\033[0;32m'
NC='\033[0m'


# download pre-trained model and toy test set data
if [ ! -e "./__pre_trained/trained_network.pt" ];then
    echo -e "${RED}Downloading pre-trained model${NC}"
    wget -q --show-progress ${MODELLINK}

    if [ -e "./${MODELNAME}.tar" ];then	
	tar -xf ${MODELNAME}.tar
	mv ${DATANAME} ../DATA/
	rm ${MODELNAME}.tar
    else
	echo "Cannot download ${MODELLINK}. Please contact the author"
    	exit
    fi
fi

# try pre-trained model
if [ -e "../DATA/${DATANAME}" ];then
    echo -e "${RED}Try pre-trained model${NC}"
    source ${ENVFILE}
    python main.py --inference --model-forward-with-file-name \
	   --trained-model __pre_trained/trained_network.pt \
	   --output-dir __pre_trained/output_converted
    echo -e "${RED}Please check generated waveforms from pre-trained model in ./__pre_trained/output_converted ${NC}"
    echo -e "${RED}These can be compared with the official samples on https://blowconversions.github.io/ ${NC}"
    echo -e "---- \n"
else
    echo "Cannot find ../DATA/${DATANAME}. Please contact the author"
fi


# train the model
if [ -e "../DATA/${DATANAME}/vctk_wav" ];then
    echo -e "${RED}Train a new model${NC}"
    echo -e "${RED}Training will take several hours. Please don't quit this job. ${NC}"
    echo -e "${RED}Please check log_train and log_err for monitoring the training process.${NC}"
    source ${ENVFILE}
    python main.py --num-workers 3 --no-best 100 --epochs 400 --batch-size 32 --lr 0.0001 \
	   --cudnn-deterministic-toggle --cudnn-benchmark-toggle \
	   --model-forward-with-file-name \
	   --lr-decay-factor 0.2 --lr-scheduler-type 1 > log_train 2>log_err
    
else
    echo -e "${RED}To train a new model${NC},"
    echo -e "${RED}Please download VCTK and run the re-naming script:${NC}"
    echo "https://github.com/joansj/blow#preprocessing"
    
    echo "Then, please convert waveform in .pt format to .wav"
    echo "and save them to ../DATA/${DATANAME}/vctk_wav"
fi
