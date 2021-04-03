#!/bin/bash
# if necessary, load conda environment
eval "$(conda shell.bash hook)"
conda activate jupyter

jupyter lab --no-browser --port=9001 --ip=127.0.0.1
