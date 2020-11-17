#!/bin/sh
ENVFILE=../../../env.sh
RED='\033[0;32m'
NC='\033[0m'

source ../../../env.sh

echo -e "\n${RED}This project is supposed to be used on NII internal server${NC}"

echo -e "\n${RED}Try pre-trained model (running for ~20 minutes) ${NC}"

LOGFILE=log_output_testset_pretrained
python main.py --inference --model-forward-with-file-name --trained-model __pretrained/trained_network.pt > ${LOGFILE} 2>&1

echo -e "\n${RED}This is the result using pre-trained model on your machine ${NC}"
python ../00_evaluate.py ${LOGFILE}

echo -e "\n${RED}This is the result produced by Xin Wang ${NC}"
LOGFILE=__pretrained/log_output_testset
python ../00_evaluate.py ${LOGFILE}

echo -e "\n${RED}Train model from scratch (running for a few hours) ${NC}"
echo -e "${RED}You can open another terminal and check log_train ${NC}"
python main.py --model-forward-with-file-name --num-workers 3 --epochs 100 \
       --no-best-epochs 50 --batch-size 64 --lr-decay-factor 0.5 \
       --lr-scheduler-type 1 --lr 0.0003 > log_train 2>log_err

echo -e "\n${RED}Evaluating the trained model (running ...) ${NC}"
LOGFILE=log_output_testset
python main.py --inference --model-forward-with-file-name > ${LOGFILE} 2>&1

echo -e "\n${RED}This is the result produced by your trained model  ${NC}"
python ../00_evaluate.py ${LOGFILE}
