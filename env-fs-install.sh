#!/bin/bash
# Install dependency for fairseq

# Name of the conda environment
ENVNAME=fairseq-pip2

eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.8 pip --yes
    conda activate ${ENVNAME}
    
    # install specified pytorch version, 
    # fairseq setup.py does not do that
    pip install torch==1.9.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
    
    # install scipy
    pip install scipy==1.7.1 numpy==1.21.2
    
    # git clone fairseq
    #  fairseq 0.10.2 on pip does not work
    git clone https://github.com/pytorch/fairseq
    cd fairseq

    #  checkout this specific commit. Latest commit does not work
    git checkout 862efab86f649c04ea31545ce28d13c59560113d
    pip install --editable ./

    # install pandas
    pip install pandas==1.4.3
else
    echo "Conda environment ${ENVNAME} has been installed"
fi
