#!/bin/bash

# We need scripts and data IOs in this repo
git clone --depth 1 https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts.git
# install conda environment
conda env create -f env.yaml

# load conda environment
eval "$(conda shell.bash hook)"
conda activate vocoding

if [[ "$CONDA_DEFAULT_ENV" == "vocoding" ]]; then
  echo "Correct env activated"
  pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
  pip install gdown
  conda develop $PWD/project-NN-Pytorch-scripts
else
  echo "Wrong or no env activated"
  exit 1
fi


# Download the model and data packages
cd abs_nn
mkdir tmp; cd tmp
gdown 19QQRQtACWPmOJlk4gBQgl1lq5_SX0u8P
tar -xzf NNvocoders.tar.gz

# get the toy data
mv data ../../
# get the pre-trained models
mv pretrained* ../
