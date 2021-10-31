#!/bin/bash
# if necessary, load conda environment
eval "$(conda shell.bash hook)"
conda activate nn-scripts-pt1.6

# when running in ./projects/*/*, add this top directory
# to python path
export PYTHONPATH=$PWD/../../../:$PYTHONPATH

