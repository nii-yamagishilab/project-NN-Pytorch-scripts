#!/bin/bash
######################################################
#__author__ = "Xin Wang"
#__email__ = "wangxin@nii.ac.jp"
#
# Demonstration script for score fusing
# Usage:
# bash 00_demo_fuse_score.sh
# 
######################################################
RED='\033[0;32m'
NC='\033[0m'

# score fuse script
EVALSCRIPT_FUSE=$PWD/03_fuse_score_evaluate.py

echo -e "${RED}For reference, this is the score fusing result from pre-trained models${NC}"
echo -e "(Scores were produced by pre-trained models on NII's server)" 
python ${EVALSCRIPT_FUSE} lfcc-lcnn-lstmsum-sig/01/__pretrained/log_output_testset lfb-lcnn-lstmsum-sig/01/__pretrained/log_output_testset spec2-lcnn-lstmsum-sig/01/__pretrained/log_output_testset  rawnet2/01/__pretrained/log_output_testset 


