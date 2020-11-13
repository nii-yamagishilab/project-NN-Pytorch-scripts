#!/bin/bash
# startup jupyter for ssh tunnel
eval "$(conda shell.bash hook)"
conda activate jupyter

jupyter lab --no-browser --port=9001 --ip=127.0.0.1
