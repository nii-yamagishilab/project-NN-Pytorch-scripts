# Vocoded spoofed speech data for spoofing countermeasures


<img src="https://pbs.twimg.com/media/FgzqiamVIAE7X9W?format=png&name=4096x4096" alt="drawing" width="600"/>

This project uses vocoded spoofed data to train spoofing speech countermeasures. 

Arxiv link: [https://arxiv.org/abs/2210.10570](https://arxiv.org/abs/2210.10570)

```
ICASSP 2023 submission

Xin Wang, and Junichi Yamagishi. Spoofed training data for speech spoofing countermeasure can be efficiently created using neural vocoders. 

```

This project demonstrates the training and scoring process using **a toy vocoded data set**.

If you need to run it on your own databases, check the steps required below. 

## Resources

Link on Zenodo | Content
------------ | -------------
[project09-ASVspoof2019LA-etrim.tar](https://zenodo.org/record/7314976/files/project09-ASVspoof2019LA-etrim.tar) | Time information to derive non-speech-trimmed version of ASVspoof2019 LA data set (LA19etrim in the paper)
[project09-voc.v1.tar](https://zenodo.org/record/7314976/files/project09-voc.v1.tar) | Scripts to produce vocoded spoofed data for Voc.v1 in the paper
[project09-voc.v2.tar](https://zenodo.org/record/7314976/files/project09-voc.v2.tar) | Vocoded spoofed data for Voc.v2 in the paper
[project09-voc.v3.tar](https://zenodo.org/record/7314976/files/project09-voc.v3.tar) | Vocoded spoofed data for Voc.v3 in the paper
[project09-voc.v4.tar](https://zenodo.org/record/7314976/files/project09-voc.v4.tar) | Vocoded spoofed data for Voc.v4 in the paper

Notes:
1. Voc.v1 uses ESPNet. We don't re-distribute the vocoded data.
2. Voc.v2 - v4 only contains the vocoded spoofed data. The bona data should be downloaded from https://doi.org/10.7488/ds/2555. We don't re-distribute the bona fide data
3. `00_demo.sh` will use a toy data set to demonstrate the usage of the scripts. Please download the data above and run the scripts to train fully-fledged CMs.

## Quick start

No need to setup anything, just run
```sh
bash 00_demo.sh model-ID-7 config_train_toyset_ID_7 01
```

Here
* `model-ID-7` is the model with ID 7 in the paper
* `config_train_toyset_ID_7` is the name of the prepared toy data set configuraiton for model ID 7. 
* `01` is a random seed

The script will
1. Download a toy data set, and a pre-trained SSL front end.
2. Build a *conda* environment if it is not available.
3. Train the CM `model-ID-7` on the toy set specified in `config_train_toyset_ID_7` with random seed `01`
4. Score the evaluation data in a toy set (which is specified in 00_demo.sh)

Note that each model has a specific data configuration file, even though they all use the same toy vocoded data set. Hence, please try the following combinations.
```sh
bash 00_demo.sh model-ID-6 config_train_toyset_ID_6 01
    
bash 00_demo.sh model-ID-4 config_train_toyset_ID_4 01
    
bash 00_demo.sh model-ID-2 config_train_toyset_ID_2 01
```



Other notes:
1. The demonstration script uses the toy data set as both the seed training set and the pool. 
2. To run the experiment on your data, please check `step-by-step` below.
3. We don't compute EER from the CM scores. See [tutorial](../../tutorials/b2_anti_spoofing) here on how to compute EER.
4. Score file contains multiple fields. Please use the `CM score` column to compute EER

```sh
TRIAL_NAME CM_SCORE MODEL_CONFIDENCE DATA_CONFIDENCE
LA_E_9933162 -9.500082 0.00012 0.00013
```


## Step-by-step

### Folder structure

```sh
|- 00_demo.sh:     demonstration script
|- 01_download.sh: script to download toy data and seed models
|- 01_train.sh:    script to start model training
|- 02_score.sh:    script to score waveforms  
|
|- main.py:                     main entry of python code
|- config_train_toyset_ID_7.py: toy data set configuration for model ID 7 
|- config_train_toyset_ID_6.py: toy data set configuration for model ID 6
|- config_train_toyset_ID_4.py: toy data set configuration for model ID 4
|- config_train_toyset_ID_2.py: toy data set configuration for model ID 2
|- config_auto.py:              general config for scoring 
|                               (no need to change it)
|
|- model-ID-7:                  CM model with ID 7 in the paper Table 3
|   |- model.py                 CM definition
|   |- data_augment.py          augmentation functions when loading waveforms
|   |- config_train_toyset_ID_7 working folder when using toy dataset 
|      |= 01                    running with random seed = 01
|         |
|         |= trained_network.pt: trained network (best so far)
|         |= epoch_yyy.pt:       trained network after yyy epoch
|         |                      with intermediate statistics
|         |= log_..._NNN:        raw output file for test set NNN
|         |= log_..._NNN_err:    raw code error messages
|         |= log_..._NNN_score.txt: scores printed in CSV format for EER
|         |                         computation
|         |= log_train:          log of model training
|         |= log_train_err:      code error messages during training
|         |= NNN.dic:            cache of data length (temporary files)
|
|- model-ID-6:
|- model-ID-4:
|- model-ID-2:
|
|- SSL_pretrained: folder to download SSL model 
|- DATA:           folder to save the toy data set
```

Notes:
1. You can compare the differences between different model.py and config*.py
2. Models 1, 3, and 5 in Table 3 of the paper are not included. They can be easily created using model-ID-2, model-ID-4, and model-ID-6, respectively. Then, just replace the data set with the ASVspoof2019 LA data set.
3. Files or folders marked with = will be produced after running the demo script.

### Flow

When running `bash 00_demo.sh model-ID-7 config_train_toyset_ID_7 01`,
The demonstration script `00_demo.sh` will
1. Prepare the working folder
   * Copy `main.py` to `model-ID-7/config_train_toyset_ID_7/01`
   * Copy `config_train_toyset_ID_7.py` to `model-ID-7/config_train_toyset_ID_7/01`
   * Copy `model-ID-7/model.py` and  'model-ID-7/model.py' to `model-ID-7/config_train_toyset_ID_7/01`
2. Call `01_train.sh` to start training in `model-ID-7/config_train_toyset_ID_7/01`
   * Python command line in `01_train.sh` is called
   * The trained CM will be saved as `model-ID-7/config_train_toyset_ID_7/01/trained_network.pt`
3. Call `02_score.sh` to score the toy data set

### How to run the script on other databases


#### Step 1
Follow `DATA/toy_example_vocoded` and prepare the data for a seed training set and a pool data set.  For each set, you need

```sh
DATA/setname/
|- train_dev:    folder to save the trn. and dev. sets waveforms
|- eval:         folder to save the eval. set waveforms 
|- protocol.txt: protocol file
|                each line contains SPEAKER TRIAL ... Label
| 
|- scp                    (just file name w/o extension)
|  |- train.lst           list of file names in trn. set 
|  |- train_bonafide.lst  list of bona fide file names in trn. set 
|  |- dev.lst             list of file names in dev. set 
|  |- test.lst            list of file names in eval. set
                 
```
Note that
* Name of `*.lst` can be anything other than `train`, `val`, or `test`.
* Prepare the dataset configuration is annoying. If you have any questions, please email me.

#### Step 2
Create a `config_NNN.py` based on toy configuration `config_train_toyset_ID_7.py`. 

The comments in `config_train_toyset_ID_7` will guide you through the preparation process. 

#### Step 3
Run 
```sh
bash 00_demo.sh MODEL config_NNN RAND_SEED
```

The trained CM from each active learning cycle will be saved to `MODEL/config_NNN/RAND_SEED`. 


---
That's all