#!/bin/sh

DATALINK=https://zenodo.org/record/6456704/files/project-01-train-data-set.tar
PACKNAME=project-01-train-data-set
FILENAME=cmu-arctic-data-set

MODELLINK=https://zenodo.org/record/6456692/files/project-05-nn-vocoders-iLPCNet.tar
MODELNAME=project-05-nn-vocoders-iLPCNet

ENVFILE=../../../env.sh

RED='\033[0;32m'
NC='\033[0m'

# download CMU-arctic data
if [ ! -e "../DATA/${FILENAME}" ];then
    echo -e "${RED}Downloading data${NC}"
    wget -q --show-progress ${DATALINK}

    if [ -e "./${PACKNAME}.tar" ];then	
	tar -xf ${PACKNAME}.tar
	cd ${FILENAME}
	sh 00_get_wav.sh
	cd ../
	mv ${FILENAME} ../DATA/${FILENAME}
	rm ${PACKNAME}.tar
    else
	echo "Cannot download ${DATALINK}. Please contact the author"
    	exit
    fi
fi


# download pre-trained model
if [ ! -e "./__pre-trained/trained_network.pt" ];then
    echo -e "${RED}Downloading pre-trained model${NC}"
    wget -q --show-progress ${MODELLINK}

    if [ -e "./${MODELNAME}.tar" ];then	
	tar -xf ${MODELNAME}.tar
	rm ${MODELNAME}.tar
    else
	echo "Cannot download ${MODELLINK}. Please contact the author"
    	exit
    fi
fi

# try pre-trained model
if [ -e "../DATA/${FILENAME}" ];then
    echo -e "${RED}Try pre-trained model${NC}"
    source ${ENVFILE}
    python main.py --inference --trained-model __pre-trained/trained_network.pt --output-dir __pre-trained/output
    echo -e "${RED}Please check generated waveforms from pre-trained model in ./__pre-trained/output"
    echo -e "----"
else
    echo "Cannot find ../DATA/${FILENAME}. Please contact the author"
fi


# train the model
if [ -e "../DATA/${FILENAME}" ];then
    echo -e "${RED}Train a new model${NC}"
    echo -e "${RED}Training will take several hours. Please don't quit this job. ${NC}"
    echo -e "${RED}Please check log_train and log_err for monitoring the training process.${NC}"
    source ${ENVFILE}

    STAGEFLAG=stage1
    python main.py --num-workers 5 --no-best 100 --epochs 100 --batch-size 256 \
	   --lr 0.0001 --model-forward-with-target --temp-flag ${STAGEFLAG} \
	   --not-save-each-epoch --save-trained-name trained_network_s1 \
	   --cudnn-deterministic-toggle --cudnn-benchmark-toggle > log_train 2>log_err

    STAGEFLAG=stage2
    python main.py --num-workers 5 --no-best 100 --epochs 400 --batch-size 256 \
	   --lr 0.0001 --model-forward-with-target --temp-flag ${STAGEFLAG} \
	   --not-save-each-epoch --trained-model trained_network_s1.pt\
	   --cudnn-deterministic-toggle --cudnn-benchmark-toggle > log_train_2 2>log_err_2

else
    echo "Cannot find ../DATA/${FILENAME}. Please contact the author"
fi


# generate using trained model
if [ -e "../DATA/${FILENAME}" ];then
    echo -e "${RED}Model is trained${NC}"
    echo -e "${RED}Generate waveform${NC}"
    source ${ENVFILE}
    python main.py --inference --trained-model trained_network.pt --output-dir output
else
    echo "Cannot find ../DATA/${FILENAME}. Please contact the author"
fi
