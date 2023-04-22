# Welcome 

This is a set of Python / Pytorch scripts and tools for various speech-processing projects. 

It is maintained by [Xin Wang](http://tonywangx.github.io/) since 2021.  

XW is a Pytorch newbie. Please feel free to give suggestions and feedback.

## Notes


* The repo is relatively large. Please use `--depth 1` option for fast cloning.

```sh
git clone --depth 1 https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts.git
```

* Latest updates:
   1. Neural vocoders pretrained on VoxCeleb2 dev and other datasets are available in tutorial notebook **chapter_a3.ipynb** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing)
   2. Code, databases, and resources for the paper below were added. Please check [project/09-asvspoof-vocoded-trn/](project/09-asvspoof-vocoded-trn/) for more details.
      > Xin Wang, and Junichi Yamagishi. Spoofed training data for speech spoofing countermeasure can be efficiently created using neural vocoders. Proc. ICASSP 2023, accepted. https://arxiv.org/abs/2210.10570
   3. Code for the paper for the paper below were added. Please check [project/08-asvspoof-activelearn](project/08-asvspoof-activelearn) for more details.
      > Xin Wang, and Junichi Yamagishi. Investigating Active-Learning-Based Training Data Selection for Speech Spoofing Countermeasure. In Proc. SLT, accepted. 2023.
   4. Pointer to tutorials on neural vocoders were moved to [./tutorials/b1_neural_vocoder](./tutorials/b1_neural_vocoder/README.md).
   
   5. All pre-trained models were moved to [Zenodo](https://doi.org/10.5281/zenodo.6349636).

## Contents

This repository contains a few projects and tutorials.

### Project


Folder | Project
------------ | -------------
[project/01-nsf](project/01-nsf) | Neural source-filter waveform models
[project/05-nn-vocoders ](project/05-nn-vocoders ) | Other neural waveform models including WaveNet, WaveGlow, and iLPCNet.
[project/03-asvspoof-mega](project/03-asvspoof-mega) | Speech spoofing countermeasures  : a comparison of some popular countermeasures
[project/06-asvspoof-ood](project/06-asvspoof-ood) | Speech spoofing countermeasures  with confidence estimation
[project/07-asvspoof-ssl](project/07-asvspoof-ssl) | Speech spoofing countermeasures with pre-trained self-supervised-learning (SSL) speech feature extractor
[project/08-asvspoof-activelearn](project/08-asvspoof-activelearn) | Speech spoofing countermeasures in an active learning framework
[project/09-asvspoof-vocoded-trn](project/09-asvspoof-vocoded-trn) | Speech spoofing countermeasures using vocoded speech as spoofed data


See [project/README.md](project) for an overview.

### Tutorials

 Folder | Status | Contents 
 --- | :-- | :-- 
 [b1_neural_vocoder](tutorials/b1_neural_vocoder) | readable and executable | tutorials on selected neural vocoders
 [b2_anti_spoofing](tutorials/b2_anti_spoofing) | partially finished | tutorials on [speech audio anti-spoofing](https://www.asvspoof.org/) 
 [b3_voice_privacy](tutorials/b3_voiceprivacy_ch) | readable and executable | tutorials on [voice privacy challenge](https://www.voiceprivacychallenge.org/) baselines

See [tutorials/README.md](tutorials) for an overview.

## Python environment

Projects above use either one of the two environments:

For most of the projects, install [env.yml](./env.yml) is sufficient 
```sh
# create environment
conda env create -f env.yml

# load environment (whose name is pytorch-1.7)
conda activate pytorch-1.7
```

For projects using SSL models, use [./env-fs-install.sh](./env-fs-install.sh) to install the dependency.
```sh
# make sure other conda envs are not loaded
bash env-fs-install.sh

# load
conda activate fairseq-pip2
```

## How to use

Most of the projects include a simple demonstration script. Take `project/01-nsf/cyc-noise-nsf` as an example:

```sh
# cd into one project
cd project/01-nsf/cyc-noise-nsf-4

# add PYTHONPATH and activate conda environment
source ../../../env.sh 

# run the script
bash 00_demo.sh
```

The printed messages will show what is happening. 

Detailed instruction is in README of each project.

## Folder structure

Name | Function
------------ | -------------
./core_scripts | scripts (Numpy or Pytorch code) to manage the training process, data io, etc.
./core_modules | finalized pytorch modules 
./sandbox | new functions and modules to be test
./project | project directories, and each folder correspond to one model for one dataset
./project/\*/\*/main.py | script to load data and run training and inference
./project/\*/\*/model.py | model definition based on Pytorch APIs
./project/\*/\*/config.py | configurations for training/val/test set data
./project/\*/\*/\*.sh | scripts to wrap the python codes

See more instructions on the design and conventions of this repository [misc/DESIGN.md](misc/DESIGN.md)



## Resources & links

* [NSF model homepage](https://nii-yamagishilab.github.io/samples-nsf/)

* [Presentation slides related to the above projects](http://tonywangx.github.io/slide.html)


---
By [Xin Wang](http://tonywangx.github.io/)