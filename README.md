# project-NII-pytorch-scripts
By Xin Wang, National Institute of Informatics, 2021

I am a new pytorch user. If you have any suggestions or questions, pleas email wangxin at nii dot ac dot jp

**Table of Contents**

* [Note](#note)

* [Overview](#overview)

* [Requirement](#env)

* [How to use](#use)

* [Project design](#conv)

* [Misc](#miscs)


------
<a name="note"></a>
## 1. Note 

**To SPCC 2021 participants**

Hands-on materials are in `./tutorials`. Please follow the `./tutorials/README` and work in this folder first

```sh
cd ./tutorials
head -n 2 README.md
# Hands-on materials for Neural vocoders
```



If you haven't obtained the package from SPCC, you can clone it from this repository. The repository is relatively large. You may use `--depth 1` option to skip unnecessary files.

```sh
git clone --depth 1 https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts.git
```



<a name="overview"></a>
## 2. Overview
This repository hosts Pytorch codes for the following projects:

### Neural source-filter waveform model [./project/01-nsf](./project/01-nsf)

1. [Cyclic-noise neural source-filter waveform model (NSF)](https://nii-yamagishilab.github.io/samples-nsf/nsf-v4.html)

2. [Harmonic-plus-noise NSF with trainable sinc filter](https://nii-yamagishilab.github.io/samples-nsf/nsf-v3.html) 

3. [Harmonic-plus-noise NSF with fixed FIR filter](https://nii-yamagishilab.github.io/samples-nsf/nsf-v2.html) 

All the projects include a pre-trained model on CMU-arctic database (4 speakers) and a demo script to run, train, do inference. Please check [./project/01-nsf/README](./project/01-nsf/README).

Generated samples from pre-trained models are in `./project/01-nsf/*/__pre_trained/output`.

Tutorial on NSF models is also available in [./tutorials](./tutorials)

Note that this is the re-implementation of the projects based on [CURRENNT](https://github.com/nii-yamagishilab/project-CURRENNT-public). All the papers published so far used CURRENNT implementation. 

Many samples can be found on [NSF homepage](https://nii-yamagishilab.github.io/samples-nsf/).


### Other neural waveform models [./project/05-nn-vocoders](./project/05-nn-vocoders)

1. [WaveNet vocoder](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

2. [WaveGlow](https://ieeexplore.ieee.org/document/8683143)

3. [Blow](https://blowconversions.github.io/)

All the projects include a pre-trained model and a one-click demo script. Please check [./project/05-nn-vocoders/README](./project/05-nn-vocoders/README).

Generated samples from pre-trained models are in `./project/05-nn-vocoders/*/__pre_trained/output`.

Tutorial is also available in [./tutorials](./tutorials)


### ASVspoof project with toy example [./project/04-asvspoof2021-toy](./project/04-asvspoof2021-toy)

It takes time to download ASVspoof2019. Therefore, this project demonstrates how to train and evaluate the ASVspoof model using a toy dataset.

Please try this project before checking other ASVspoof projects.

A similar project is adopted for [ASVspoof2021 LFCC-LCNN baseline](https://github.com/asvspoof-challenge/2021), although the LFCC front-end is slightly different.

Please check [./project/04-asvspoof2021-toy/README](./project/04-asvspoof2021-toy/README).

### Speech anti-spoofing for ASVspoof 2019 LA [./project/03-asvspoof-mega](./project/03-asvspoof-mega)

Projects for [this anti-spoofing project (A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection, paper on arxiv)](https://arxiv.org/abs/2103.11326).

There were 36 systems investigated, each of which was trained and evaluated for 6 rounds with different random seeds.

![EER-mintDCF](./misc/fig_eer_table.png)

Pre-trained models, scores, training recipes are all available. Please check [./project/03-asvspoof-mega/README](./project/03-asvspoof-mega/READNE).


### (Preliminary) speech anti-spoofing [./project/02-asvspoof](./project/02-asvspoof)

1. Baseline LFCC + LCNN-binary-classifier (lfcc-lcnn-sigmoid)

2. LFCC + LCNN + angular softmax (lfcc-lcnn-a-softmax)

3. LFCC + LCNN + one-class softmax (lfcc-lcnn-ocsoftmax)

4. LFCC + ResNet18 + one-class softmax (lfcc-restnet-ocsoftmax)

This is a preliminart experiment on ASVspoof2019 LA task. I trained each system for 6 times on various GPU devices (single V100 or P100 card), each time with a different random initial seed. Figure below shows the DET curves for these systems:
![det_curve](./misc/fig_det_baselines.png)

The results vary a lot when simply changing the initial random seends, even with the same random seed, Pytorch environment, and [deterministic algorithm selected](https://pytorch.org/docs/stable/notes/randomness.html). This preliminary test motivated the study in `./project-03-asvspoof-mega`.

For LCNN, please check [this paper](https://www.isca-speech.org/archive/Interspeech_2019/abstracts/1768.html); for LFCC, please check [this paper](https://www.isca-speech.org/archive/interspeech_2015/i15_2087.html); for one-class softmax in ASVspoof, please check [this paper](https://arxiv.org/pdf/2010.13995).

<a name="env"></a>
## 3. Python environment

You may use [./env.yml](./env.yml) to create the environment: 

```sh
# create environment
conda env create -f env.yml

# load environment (whose name is nn-script-pt1.6)
conda activate nn-script-pt1.6
```

<a name="use"></a>
## 4. How to use
Take `project/01-nsf/cyc-noise-nsf` as an example:

```sh
# cd into one project
cd project/01-nsf/cyc-noise-nsf-4

# add PYTHONPATH and activate conda environment
source ../../../env.sh 

# run the script
bash 00_demo.sh
```

The printed info will show what is happening. The script may need 1 day or more to finish.

You may also put the job to the background rather than waiting for the job in front of the terminal:

```sh
# run the script in background
bash 00_demo.sh > log_batch 2>&1 &
```

The above steps will download the CMU-arctic data, run waveform generation using a pre-trained model, and train a new model. 

<a name="conv"></a>
## 5. Project design and convention

### Data format

* Waveform: 16/32-bit PCM or 32-bit float WAV that can be read by [scipy.io.wavfile.read](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html) 

* Other data: binary, float-32bit, litten endian ([numpy dtype <f4](https://numpy.org/doc/1.18/reference/generated/numpy.dtype.html)). The data can be read in python by:

```python
# for a data of shape [N, M]
f = open(filepath,'rb')
datatype = np.dtype(('<f4',(M,)))
data = np.fromfile(f,dtype=datatype)
f.close()
```

I assume data should be stored in [c_continuous format](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) (row-major). 
There are helper functions in ./core_scripts/data_io/io_tools.py to read and write binary data:

```python
# create a float32 data array
import numpy as np
data = np.asarray(np.random.randn(5, 3), dtype=np.float32)

# write to './temp.bin' and read it as data2
import core_scripts.data_io.io_tools as readwrite
readwrite.f_write_raw_mat(data, './temp.bin')
data2 = readwrite.f_read_raw_mat('./temp.bin', 3)

# result should 0
data - data2
```

More instructions can be found in the Jupyter notebook [./tutorials/c01_data_format.ipynb](./tutorials/c01_data_format.ipynb).


### Files in this repository

Name | Function
------------ | -------------
./core_scripts | scripts to manage the training process, data io, and so on
./core_modules | finished pytorch modules 
./sandbox | new functions and modules to be test
./project | project directories, and each folder correspond to one model for one dataset
./project/\*/\*/main.py | script to load data and run training and inference
./project/\*/\*/model.py | model definition based on Pytorch APIs
./project/\*/\*/config.py | configurations for training/val/test set data

The motivation is to separate the training and inference process, the model definition, and the data configuration. For example:

* To define a new model, change model.py

* To run on a new database, change config.py

### How the script works

The script starts with main.py and calls different function for model training and inference. 

```sh
During training:

     <main.py>        Entry point and controller of training process
        |           
   Argument parse     core_scripts/config_parse/arg_parse.py
   Initialization     core_scripts/startup_config.py
   Choose device     
        | 
Initialize & load     core_scripts/data_io/customize_dataset.py
training data set
        |----------|
        .     Load data set   <config.py> 
        .     configuration 
        .          |
        .     Loop over       core_scripts/data_io/customize_dataset.py
        .     data subset
        .          |       
        .          |---------|
        .          .    Load one subset   core_scripts/data_io/default_data_io.py
        .          .         |
        .          |---------|
        .          |
        .     Combine subsets 
        .     into one set
        .          |
        |----------|
        |
Initialize & load 
development data set  
        |
Initialize Model     <model.py>
Model(), Loss()
        | 
Initialize Optimizer core_scripts/op_manager/op_manager.py
        |
Load checkpoint      --trained-model option to main.py
        |
Start training       core_scripts/nn_manager/nn_manager.py f_train_wrapper()
        |             
        |----------|
        .          |
        .     Loop over training data
        .     for one epoch
        .          |
        .          |-------|    core_scripts/nn_manager/nn_manager.py f_run_one_epoch()
        .          |       |    
        .          |  Loop over 
        .          |  training data
        .          |       |
        .          |       |-------|
        .          |       .    get data_in, data_tar, data_info
        .          |       .    Call data_gen <- Model.forward(...)   <mode.py>
        .          |       .    Call Loss.compute()                   <mode.py>
        .          |       .    loss.backward()
        .          |       .    optimizer.step()
        .          |       .       |
        .          |       |-------|
        .          |       |
        .          |  Save checkpoint 
        .          |       |
        .          |  Early stop?
        .          |       | No  \
        .          |       |      \ Yes
        .          |<------|       |
        .                          |
        |--------------------------|
       Done
```

A detailed flowchat is [./misc/APPENDIX_1.md](./misc/APPENDIX_1.md). This may be useful if you want to hack on the code.


<a name="miscs"></a>
## 6 On NSF projects (./project/01-nsf)
### Differences from [CURRENNT implementation](https://github.com/nii-yamagishilab/project-CURRENNT-public)
There may be more, but here are the important ones:

* "Batch-normalization": in CURRENNT, "batch-normalization" is conducted along the length sequence, i.e., assuming each frame as one sample;

* No bias in CNN and FF: due to the 1st point, NSF in this repository uses bias=false for CNN and feedforward layers in neural filter blocks, which can be helpful to make the hidden signals around 0;

* Smaller learning rate: due to the 1st point, learning rate in this repository is decreased from 0.0003 to a smaller value. Accordingly, more training epochs are required;

* STFT framing/padding: in CURRENNT, the first frame starts from the 1st step of a signal; in this Pytorch repository (as Librosa), the first frame is centered around the 1st step of a signal, and the frame is padded with 0;

* STFT backward: in CURRENNT, STFT backward follows the steps in [this paper](https://ieeexplore.ieee.org/document/8915761/); in Pytorch repository, backward over STFT is done by the Pytorch library. 

* ...

The learning curves look similar to the CURRENNT version.
![learning_curve](./misc/fig1_curve.png)


### 24kHz
Most of my experiments are done on 16 kHz waveforms. For 24 kHz waveforms, FIR or sinc digital filters in the model may be changed for better performance:

1. **hn-nsf**: lp_v, lp_u, hp_v, and hp_u are calculated for 16 kHz configurations. For different sampling rate, you may use this online tool http://t-filter.engineerjs.com to get the filter coefficients. In this case, the stop-band for lp_v and lp_u is extended to 12k, while the pass-band for hp_v and hp_u is extended to 12k. The reason is that, no matter what is the sampling rate, the actual formats (in Hz) and spectral of sounds don't change with the sampling rate;

2. **hn-sinc-nsf and cyc-noise-nsf**: for the similar reason above, the cut-off-frequency value (0, 1) should be adjusted. I will try (hidden_feat * 0.2 + uv * 0.4 + 0.3) * 16 / 24 in model.CondModuleHnSincNSF.get_cut_f();


## Links

* [NSF model homepage](https://nii-yamagishilab.github.io/samples-nsf/)

* [Presentation slides related to this projects](http://tonywangx.github.io/slide.html)


The end