#!/bin/bash
# Install dependency for s3prl

# Name of the conda environment
ENVNAME=s3prl-pip2

eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}

retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.8 pip --yes
    conda activate ${ENVNAME}

    # install trorch
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
    
    # git clone s3prl
    git clone https://github.com/s3prl/s3prl.git
    
    cd s3prl
    #  checkout this specific commit. Latest commit does not work
    git checkout 90d11f2faa6cc46f6d3c852604a9a80e09d18ab1
    pip install -e .

    # install scipy
    conda install -c anaconda scipy=1.7.1 --yes
    # install pandas
    pip install pandas==1.4.3

else
    echo "Conda environment ${ENVNAME} has been installed"
fi
