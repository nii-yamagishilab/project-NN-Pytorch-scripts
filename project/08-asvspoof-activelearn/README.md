# Active Learning for Spoofing Countermeasure


<img src="https://pbs.twimg.com/media/FPFN_3AaIAM5ANl?format=png&name=medium" alt="drawing" width="400"/>

This is the project using active learning to train a speech spoofing countermeasure. 

Arxiv link: [https://arxiv.org/abs/2203.14553](https://arxiv.org/abs/2203.14553)

```
Xin Wang, and Junichi Yamagishi. Investigating Active-Learning-Based Training Data Selection for Speech Spoofing Countermeasure. In Proc. SLT, accepted. 2023.


@inproceedings{Wang2023,
author = {Wang, Xin and Yamagishi, Junichi},
booktitle = {Proc. SLT},
pages = {accepted},
title = {{Investigating Active-learning-based Training Data Selection for Speech Spoofing Countermeasure}},
year = {2023}
}
```

The ideas are not straightforward to implement. It is also complicated to prepare off-the-self scripts for all kinds of data sets. 

Hence, this project demonstrates the training and scoring process using **a toy data set**.

The data sets and resources used in the experiment of the paper is on [zenodo](https://zenodo.org/record/7497769/files/project-08-activelearning-data-resources.tar). Some data sets cannot be fully re-distributed by this repository, but we provide a link to the original repository. 

We also provide the audio file selected by the two best AL systems. Please check the `selected_files` in the above tar package on zenodo.


If you need to run it on your own databases, check the steps required below. Apologize if the scripts will take you much time to setup : )

## Quick start

No need to setup anything, just:
```sh
bash 00_demo.sh model-AL-NegE config_AL_train_toyset 01
```

Here
* `model-AL-NegE` is the model name. It can be other models
* `config_AL_train_toyset` is the name of the prepared toy data set configuraiton. 
* `01` is a random seed

The script will
1. Download a toy data set, an  SSL front end, and a seed CM model.
2. Build a *conda* environment if it is not available.
3. Train the CM `model-AL-NegE` on the toy set specified in `config_AL_train_toyset` with random seed `01`
4. Score the evaluation data in a toy set (which is specified in 00_demo.sh)

Notes:
1. The demonstration script uses the toy data set as both the seed training set and the pool. 
2. To run the experiment on your data, please check `step-by-step` below.
3. We don't compute EER from the CM scores. See [tutorial](../../tutorials/b2_anti_spoofing) here on how to compute EER.
4. Score file contains three fields. Please use the `CM score` column to compute EER

```sh
TRIAL_NAME CM_SCORE CONFIDENCE_SCORE
LA_E_9933162 -9.500082 -4.499880
```


## Step-by-step

### Folder structure

```sh
|- 00_demo.sh:     demonstration script
|- 01_download.sh: script to download toy data and seed models
|- 01_train.sh:    script to start model training
|- 02_score.sh:    script to score waveforms  
|
|- main.py:                   main entry of python code
|- config_AL_train_toyset.py: configuration for the toy data set
|- config_auto.py:            general config for scoring 
|                             (no need to change it)
|
|- model-AL-NegE:              Al-NegE CM model in the paper
|   |- model.py                CM definition
|   |- config_AL_train_toyset  working folder when using toy 
|      |                       dataset
|      |= 01                   running with random seed = 01
|         |
|         |= trained_network_al_cycle_xxx.pt: trained network
|         |                                   after xxx cycles
|         |= epoch_al_cycle_xxx_epoch_yyy.pt: trained network
|         |                                   with intermediate statistics
|         |                                   after xxx cycles
|         |= cache_al_data_log_xxx.txt:       cache that shows the 
|         |                                   data selected in each cycle
|         |= log_..._cycle_xxx_NNN:           raw output file for test set NNN
|         |                                   after xxx active learning cycle
|         |= log_..._cycle_xxx_NNN_err:       raw code error messages
|         |= log_..._cycle_xxx_NNN_score.txt: scores printed in CSV format for EER
|         |                                   computation
|         |= log_train:                       log of model training
|         |= log_train_err:                   code error messages during training
|         |= NNN.dic:                         cache of data length (temporary files)
|
|- model-AL-Adv:               ALAdv in the paper
|- model-AL-Pas:               ALPas in the paper
|- model-AL-PosE:              ALPosE in the paper
|- model-AL-Rem:               ALRem in the paper
|
|- seed_model:     folder to download CM pre-trained on seed 
|                  training set
|- SSL_pretrained: folder to download SSL model 
|- DATA:           folder to save the toy data set
```

Files or folders marked with = will be produced after running the demo script.


### Flow

When running `bash 00_demo.sh model-AL-NegE config_AL_train_toyset 01`,
The demonstration script `00_demo.sh` will
1. Prepare the working folder
   * Copy `main.py` to `model-AL-NegE/config_AL_train_toyset/01`
   * Copy `config_AL_train_toyset.py` to `model-AL-NegE/config_AL_train_toyset/01`
   * Copy `model-AL-NegE/model.py` to `model-AL-NegE/config_AL_train_toyset/01`
2. Call `01_train.sh` to start training in `model-AL-NegE/config_AL_train_toyset/01`
   * Python command line in `01_train.sh` is called
   * CM is trained for multiple active learning cycles (Figure 1 above)
   * The trained CM after each cycle will be saved as `model-AL-NegE/config_AL_train_toyset/01/trained_network_al_cycle_NNN.pt`
3. Call `02_score.sh` to score the toy data set
   * For each cycle `model-AL-NegE/config_AL_train_toyset/01/trained_network_al_cycle_NNN.pt`, score the test set

### How to run the script on other databases


#### Step 1
Follow `DATA/toy_example` and prepare the data for a seed training set and a pool data set.  For each set, you need

```sh
DATA/SEEDSET/
|- train_dev:    folder to save the trn. and dev. sets waveforms
|- eval:         folder to save the eval. set waveforms 
|- protocol.txt: protocol file
|                each line contains SPEAKER TRIAL ... Label
|- scp
|  |- train.lst  list of file names in trn. set 
|  |- val.lst    list of file names in dev. set 
|  |- test.lst   list of file names in eval. set
                 (just file name w/o extension)
```
Note that
* If the pool or seed set contains multiple subsets, just prepare each subset in the same manner as above. 
* Name of `*.lst` can be anything other than `train`, `val`, or `test`.
* We will tell the script which lst to use in the next step.

#### Step 2
Create a `config_AL_NNN.py` based on `config_AL_train_toyset.py`. 

#### Step 3
Run 
```sh
bash 00_demo.sh MODEL config_AL_NNN RAND_SEED
```

The trained CM from each active learning cycle will be saved to `MODEL/config_AL_NNN/RAND_SEED`. 


---
That's all