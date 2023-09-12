# Can Large-scale vocoded spoofed data improve speech spoofing countermeasure with a self-supervised front end?

This project uses large-scale vocoded spoofed data to train SSL-based front end for speech spoofing countermeasures. 

```bibtex
Xin Wang, and Junichi Yamagishi. Can Large-scale vocoded spoofed data improve speech spoofing countermeasure with a self-supervised front end? Submitted
```

This project demonstrates the training and scoring process using **a toy vocoded data set**. If you need to run it on your own databases, check 3. Step-by-step below.

Apologize for inconvenience for the troublesome setting and onerous manual work. 

## 1. Vocoded VoxCeleb2

The vocoded Voxceleb2 dataset cannot be re-distributed due to restriction of the license and privacy issue.  However, pre-trained vocoders and scripts to do copy-synthesis can be found in this in this notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing).

Please follow the instruction in the notebook and run it.  This notebook will
1. download the pre-trained models on the Colab runtime,
2. and do copy-synthesis (vocoding) on one sample utterance.

You will find scripts to use the pre-trained models offline.

## 2 Quick start

### 2.1 Training and evaluation using a toy-data set

No need to setup anything, just run
```sh
bash 00_demo.sh model-ID-P3 config_train_toyset 01
```

Here
* `model-ID-P3` is the model with ID P3 in the paper
* `config_train_toyset` is the name of the prepared toy data set configuraiton for model ID 7. 
* `01` is a random seed

This 00_demo.sh script will
1. Download a toy data set and the continually trained SSL front end.
2. Build a *conda* environment if it is not available.
3. Train the CM `model-ID-P3` using the toy set specified in `config_train_toyset` and a random seed value of `01`
4. Use the trained model to score trials in the toy set (evaluation subset)



Other notes:
1. If the environment cannot be installed correctly, please check [../../env-s3prl-install.sh](../../env-s3prl-install.sh) and manually install the dependency.
2. To run the experiment on your data, please check `step-by-step` below.
3. We don't compute EER from the CM scores. See this [tutorial](../../tutorials/b2_anti_spoofing) on how to compute EER.
4. Score file contains multiple columns. Please use the 2nd column `CM score` to compute EER. The 3rd and 4th columns are the model and data confidence scores.

```sh
TRIAL_NAME CM_SCORE MODEL_CONFIDENCE DATA_CONFIDENCE
LA_E_9933162 -9.500082 0.00012 0.00013
```

### 2.2 Evaluation using a toy-data set and models trained in the paper


Just run
```sh
# Run script
bash 00_demo_pretrained.sh MODEL_DIR TESTSET_DIR TESTSET_NAME
```
where
* **MODEL_DIR** is the path to the model directory that contains trained_network.pt
* **TESTSET_DIR** is the path to the path to the directory of the test set waveforms  
* **TESTSET_NAME** is a text string used to name the test set.

For example
```sh
bash 00_demo_pretrained.sh $PWD/model-ID-P3/trained-for-paper/01 $PWD/DATA/toy_example_vocoded/eval toyset_eval
```

Notes:
1. Remember to use full path (e.g., by adding $PWD). All the pre-trained models on zenodo should be accessed using `model-ID-*/trained-for-paper/01`
1. The script will download trained models and score the test set files.  It will take some time to download them. 
2. The printed information will tell you where to find the generated score file.
3. Score files produced by the authors are available in `*/trained-for-paper/01/log_output_*.txt`.


## 3. Step-by-step

### 3.1 Folder structure

```sh
|- 00_demo.sh:     demonstration script
|- 01_download.sh: script to download toy data and SSL model
|- 01_train.sh:    script to start model training
|- 02_score.sh:    script to score waveforms  
|
|- main.py:                     main entry of python code
|- config_train_toyset.py:      configuration file for each model
|- config_auto.py:              general config for scoring 
|                               (no need to change it)
|
|- model-ID-P3:                CM model with ID P3 in the paper
|   |- model.py                CM definition
|   |- data_augment.py         augmentation functions when loading waveforms
|   |- config_train_toyset     working folder when using toy dataset 
|        |                       
|        |- 01                  training and testing using random seed 01
|
|- model-ID-*:                 Similar to model-ID-P3
|
|- SSL_pretrained: folder to download SSL model 
|- DATA:           folder to save the toy data set
```


### 3.2 Flow of script

When running `bash 00_demo.sh model-ID-P3 config_train_toyset 01`,
The demonstration script `00_demo.sh` will
1. Prepare the working folder
   * Copy `main.py` to `model-ID-P3/config_train_toyset/01`
   * Copy `config_train_toyset.py` to `model-ID-P3/config_train_toyset/01`
   * Copy `model-ID-P3/model.py` and  'model-ID-P3/data_augment.py' to `model-ID-P3/config_train_toyset/01`
2. Call `01_train.sh` to start training in `model-ID-P3/config_train_toyset_ID_P3/01`
   * Python command line in `01_train.sh` is called
   * The trained CM will be saved as `model-ID-P3/config_train_toyset/01/trained_network.pt`
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
Create a `config_NNN.py` based on toy configuration `config_train_toyset.py`. 

The comments in `config_train_toyset` will guide you through the preparation process. 

#### Step 3
Run 
```sh
bash 00_demo.sh MODEL config_NNN RAND_SEED
```

The trained CM from each active learning cycle will be saved to `MODEL/config_NNN/RAND_SEED`. 


---
That's all