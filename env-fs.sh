#!/bin/bash
# if necessary, load conda environment
eval "$(conda shell.bash hook)"

conda activate fairseq-pip2
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Cannot load fairseq-pip2"
    exit 1
fi

# when running in ./projects/*/*, add this top directory
# to python path
export PYTHONPATH=$PWD/../../../:$PYTHONPATH

