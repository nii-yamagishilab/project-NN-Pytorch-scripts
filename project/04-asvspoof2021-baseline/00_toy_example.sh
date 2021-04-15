#!/bin/bash
########################
# Script for toy_example
# This script will 
#  1. install pytorch env using conda 
#  2. untar toy data set
#  3. run evaluation and training process
#
# If GPU memory is less than 16GB, please reduce
#   --batch-size in 00_train.sh
########################
# create conda environment
bash 01_conda.sh

# untar the toy-example data
cd DATA
tar -xzvf toy_example.tar.gz
cd ..

# run scripts
cd lfcc-lcnn-lstm-p2s_toy_example
# evaluation using pre-trained model
bash 01_eval.sh
#training, with pre-trained model as initialization
bash 00_train.sh

