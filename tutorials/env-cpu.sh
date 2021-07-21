#!/bin/bash
# if necessary, load conda environment
eval "$(conda shell.bash hook)"
conda activate pytorch-cpu

# Add the package root directory to python path
# We need some of the blocks in sandbox
export PYTHONPATH=$PWD/../:$PYTHONPATH