# Vocoded spoofed speech data for spoofing countermeasures


<img src="https://pbs.twimg.com/media/FgzqiamVIAE7X9W?format=png&name=4096x4096" alt="drawing" width="600"/>

This project uses vocoded spoofed data to train spoofing speech countermeasures. 

Arxiv link: [https://arxiv.org/abs/2210.10570](https://arxiv.org/abs/2210.10570)

```bibtex
Xin Wang, and Junichi Yamagishi. Spoofed training data for speech spoofing countermeasure can be efficiently created using neural vocoders. Proc. ICASSP, 2023 (to appear)
```

This project demonstrates the training and scoring process using **a toy vocoded data set**. If you need to run it on your own databases, check Step-by-step. 

Apologize for inconvenience for the troublesome setting and onerous manual work. 

## 1. Resources


### 1.1 Training set

Link on Zenodo | Content
------------ | -------------
[project09-voc.v1.tar](https://zenodo.org/record/7314976/files/project09-voc.v1.tar) | Scripts to produce vocoded spoofed data for Voc.v1 in the paper
[project09-voc.v2.tar](https://zenodo.org/record/7314976/files/project09-voc.v2.tar) | Vocoded spoofed data for Voc.v2 in the paper
[project09-voc.v3.tar](https://zenodo.org/record/7314976/files/project09-voc.v3.tar) | Vocoded spoofed data for Voc.v3 in the paper
[project09-voc.v4.tar](https://zenodo.org/record/7314976/files/project09-voc.v4.tar) | Vocoded spoofed data for Voc.v4 in the paper

Notes:
1. Voc.v1 uses ESPNet. We don't re-distribute the vocoded data.
2. Voc.v2 - v4 only contains the vocoded spoofed data. The bona data should be downloaded from https://doi.org/10.7488/ds/2555. We don't re-distribute the bona fide data.
3. `00_demo.sh` will automatically download a toy data set to demonstrate the usage of the scripts. If you want to train a fully-fledged model, please download the data above, make necessary modifications to the configuration file (see step-by-step), and run the scripts.

### 1.2 Evaluation sets

The evaluation sets used in the paper are from the following data set. 

Name in the paper and link      | Content
------------ | -------------
[LA19eval](https://datashare.ed.ac.uk/handle/10283/3336) |  evaluation set in ASVspoof2019 Logical Access (LA) partition
[LA21eval](https://zenodo.org/record/4837263) and [labels](https://github.com/asvspoof-challenge/2021) |  evaluation set of ASVspoof2021 LA partition
[DF21eval](https://zenodo.org/record/4835108) and [labels](https://github.com/asvspoof-challenge/2021) |  evaluation set of ASVspoof2021 DF partition
------------ | -------------
[LA19etrim](https://zenodo.org/record/7314976/files/project09-ASVspoof2019LA-etrim.tar) | Time information to derive non-speech-trimmed version of ASVspoof2019 LA evaluation set
LA21hid |  hidden subset in the ASVspoof2021 LA evaluation set
DF21hid |  hidden subset in the ASVspoof2021 DF evaluation set
[WaveFake](https://zenodo.org/record/5642694) | WaveFake (the whole dataset)
[InWild](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | In-the-Wild Audio Deepfake Data (the whole dataset)
[LA15eval (in paper appendix)](https://datashare.ed.ac.uk/handle/10283/853) | the evaluation set in ASVspoof2015 dataset

Notes:

1. LA21hid and DF21hid have been included in the public available ASVspoof 2021 data sets. Just download the LA21eval and DF21eval. Files belonging to the hidden sets are marked in the label file (see On meta-labels [here](https://github.com/asvspoof-challenge/2021/tree/main/eval-package)) . You may use the scripts [here](https://github.com/asvspoof-challenge/2021/tree/main/eval-package) to compute the EERs on the hidden set.
2. We only share the time information with which the LA19eval can be trimmed into LA19etrim. We don't redistribute the audio files.

### 1.3 Pretrained Vocoders

Neural vocoders used to create Voc.v2, Voc.v3, and Voc.v4 are available in this notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing).

Please follow the instruction in the notebook and run it.  This notebook will
1. download the pre-trained models on the Colab runtime,
2. and do copy-synthesis (vocoding) on one sample utterance.

You will find scripts to use the pre-trained models offline.

If you want to train the neural vocoders, please check [project/01-nsf](../01-nsf), [project/05-nn-vocoders](../05-nn-vocoders), and the [tutorials on neural vocoders](../../tutorials/b1_neural_vocoder/).


## 2 Quick start

### 2.1 Training and evaluation using a toy-data set

No need to setup anything, just run
```sh
bash 00_demo.sh model-ID-7 config_train_toyset_ID_7 01
```

Here
* `model-ID-7` is the model with ID 7 in the paper
* `config_train_toyset_ID_7` is the name of the prepared toy data set configuraiton for model ID 7. 
* `01` is a random seed

This 00_demo.sh script will
1. Download a toy data set and a pre-trained SSL front end.
2. Build a *conda* environment if it is not available.
3. Train the CM `model-ID-7` using the toy set specified in `config_train_toyset_ID_7` and a random seed value of `01`
4. Score the evaluation trials in the toy set 

Note that each model has a specific data configuration file, even though they all use the same toy vocoded data set. Hence, please try the following combinations.
```sh
bash 00_demo.sh model-ID-6 config_train_toyset_ID_6 01
    
bash 00_demo.sh model-ID-4 config_train_toyset_ID_4 01
    
bash 00_demo.sh model-ID-2 config_train_toyset_ID_2 01
```



Other notes:
1. To run the experiment on your data, please check `step-by-step` below.
2. We don't compute EER from the CM scores. See [tutorial](../../tutorials/b2_anti_spoofing) here on how to compute EER.
4. Score file contains multiple columns. Please use the 2nd column `CM score` to compute EER. The 3rd and 4th columns are the model and data confidence scores.

```sh
TRIAL_NAME CM_SCORE MODEL_CONFIDENCE DATA_CONFIDENCE
LA_E_9933162 -9.500082 0.00012 0.00013
```

### 2.2 Evaluation using a toy-data set and models trained in the paper


Just run
```sh
# Download pre-trained models
bash 01_download_pretrained.sh

# Run script
bash 00_demo_pretrained.sh MODEL_DIR TESTSET_DIR TESTSET_NAME
```
where
* **MODEL_DIR** is the path to the model directory that contains trained_network.pt
* **TESTSET_DIR** is the path to the path to the directory of the test set waveforms  
* **TESTSET_NAME** is a text string used to name the test set.

For example
```sh
bash 00_demo_pretrained.sh $PWD/model-ID-7/trained-for-paper/01 $PWD/DATA/toy_example_vocoded/eval toyset_eval
```

The script will download trained models (~5GB) and score the test set files. 
 It will take some time to download the files. 

The printed information will tell you where to find the generated score file.




## 3. Step-by-step

### 3.1 Folder structure

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
|- model-ID-7:                 CM model with ID 7 in the paper Table 3
|   |- model.py                CM definition
|   |- data_augment.py         augmentation functions when loading waveforms
|   |- config_train_toyset_ID_7 working folder when using toy dataset 
|        |                       
|        |- 01                  training and testing using random seed 01
|
|- model-ID-6: similar to above
|- model-ID-4: ...
|- model-ID-2: ...
|
|- SSL_pretrained: folder to download SSL model 
|- DATA:           folder to save the toy data set
```

Notes:
1. You can compare the differences between different model.py and config*.py
2. Models 1, 3, and 5 in Table 3 of the paper are not included. They can be easily created using model-ID-2, model-ID-4, and model-ID-6, respectively. Then, just replace the data set with the ASVspoof2019 LA data set.


### 3.2 Flow of script

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

### 3.3 How to run the script on other databases


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