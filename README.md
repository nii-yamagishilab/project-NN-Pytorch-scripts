# project-NII-pytorch-scripts
By Xin Wang, National Institute of Informatics, 2021

I am a new pytorch user. If you have any suggestions or questions, pleas email wangxin at nii dot ac dot jp


* [Note](#note)
* [Overview](#overview)
* [Requirement](#env)
* [How to use](#use)
* [Project design](#conv)
* [Misc](#miscs)


------
<a name="note"></a>
## 1. For SPCC 2021 participants

Hands-on materials are in `./tutorial`. Please go to this folder
```sh
cd ./tutorial
```
and follow README.md there.

The root folder here holds materials for part3 of the hands-on.

<!---
## <a name="update"></a>1. Update log

* 2021-04: Add projects on ASVspoof [./project/03-asvspoof-mega](./project/03-asvspoof-mega) for paper [A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection (on arxiv)](https://arxiv.org/abs/2103.11326)
* 2020-11: Add preliminary projects on ASVspoof [./project/02-asvspoof](./project/02-asvspoof)
* 2020-08: Add tutorial materials [./tutorials](./tutorials). Most of the materials are Jupyter notebooks and can be run on your Laptop using CPU. 
--->

<a name="overview"></a>
## 2. Overview
This repository hosts Pytorch codes for the following projects:

### Neural source-filter waveform model [./project/01-nsf](./project/01-nsf)

1. [Cyclic-noise neural source-filter waveform model (NSF)](https://nii-yamagishilab.github.io/samples-nsf/nsf-v4.html)

2. [Harmonic-plus-noise NSF with trainable sinc filter](https://nii-yamagishilab.github.io/samples-nsf/nsf-v3.html) 

3. [Harmonic-plus-noise NSF with fixed FIR filter](https://nii-yamagishilab.github.io/samples-nsf/nsf-v2.html) 

All NSF projects come with pre-trained models on CMU-arctic (4 speakers) and a one-click demo script to run, train, do inference. Please check [./project/01-nsf/README](./project/03-asvspoof-mega/READNE).

Generated samples from pre-trained models are in `./project/01-nsf/*/__pre_trained/output`.

Tutorial is also available in [./tutorial/](./tutorial)

Note that this is the re-implementation of projects based on [CURRENNT](https://github.com/nii-yamagishilab/project-CURRENNT-public). All the papers published so far used CURRENNT implementation. Many samples can be found on [NSF homepage](https://nii-yamagishilab.github.io/samples-nsf/).


### Neural source-filter waveform model [./project/05-nn-vocoders](./project/05-nn-vocoders)

1. [WaveNet vocoder](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

2. [WaveGlow](https://ieeexplore.ieee.org/document/8683143)

3. [Blow](https://blowconversions.github.io/)

All projects come with pre-trained models and a one-click demo script. Please check [./project/01-nsf/README](./project/03-asvspoof-mega/READNE).

Generated samples from pre-trained models are in `./project/05-nn-vocoders/*/__pre_trained/output`.

Tutorial is also available in [./tutorial/](./tutorial)


### ASVspoof project with toy example [./project/04-asvspoof2021-toy](./project/04-asvspoof2021-toy)

It takes time to download ASVspoof2019. Therefore, this project demonstrates how to train and evaluate the ASVspoof model using a toy dataset.

Please try this project before checking other ASVspoof projects.

A similar project is adopted for ASVspoof2021 LFCC-LCNN baseline (https://www.asvspoof.org/), although the LFCC front-end is slightly different.

### Speech anti-spoofing for ASVspoof 2019 LA [./project/03-asvspoof-mega](./project/02-asvspoof-mega)

Projects for [this anti-spoofing project (A Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection, paper on arxiv)](https://arxiv.org/abs/2103.11326).

There were 36 systems investigated, each of which was trained and evaluated for 6 rounds with different random seeds.

![EER-mintDCF](./misc/fig_eer_table.png)

Pre-trained models, scores, training recipes are all available. Please check [./project/03-asvspoof-mega/README](./project/03-asvspoof-mega/READNE).

 
### (Preliminary) speech anti-spoofing [./project/02-asvspoof](./project/02-asvspoof)
1. Baseline LFCC + LCNN-binary-classifier (lfcc-lcnn-sigmoid)
2. LFCC + LCNN + angular softmax (lfcc-lcnn-a-softmax)
3. LFCC + LCNN + one-class softmax (lfcc-lcnn-ocsoftmax)
4. LFCC + ResNet18 + one-class softmax (lfcc-restnet-ocsoftmax)

On ASVspoof2019 LA, EER is around 3%, and min-tDCF (legacy-version) is around 0.06~0.08. I trained each system for 6 times on various GPU devices (single V100 or P100 card), each time with a different random initial seed. Figure below shows the DET curves for these systems:
![det_curve](./misc/fig_det_baselines.png)

As you can see how the results vary a lot when simply changing the initial random seends. Even with the same random seed, Pytorch environment, and [deterministic algorithm selected](https://pytorch.org/docs/stable/notes/randomness.html), the trained model may be different due to the CUDA and GPU. It is encouraged to run the model multiple times with different random seeds and show the variance of the evaluation results.

For LCNN, please check (Lavrentyeva 2019); for LFCC, please check (Sahidullah 2015); for one-class softmax in ASVspoof, please check (Zhang 2020).

<a name="env"></a>
## 3. Python environment

You may use [./env.yml](./env.yml) or [./env2.yml](./env2.yml) to create the environment: 

```sh
# create environment
conda env create -f env.yml

# load environment (whose name is pytorch-1.6)
conda activate pytorch-1.6
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

The printed info will tell you what is happening. The script may need 1 day or more to finish.

You may also put the job to background rather than waiting for the job while keeping the terminal open:

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
```
# for a data of shape [N, M]
>>> f = open(filepath,'rb')
>>> datatype = np.dtype(('<f4',(M,)))
>>> data = np.fromfile(f,dtype=datatype)
>>> f.close()
```
I assume data should be stored in [c_continuous format](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html) (row-major). 
There are helper functions in ./core_scripts/data_io/io_tools.py to read and write binary data:
```
# create a float32 data array
>>> import numpy as np
>>> data = np.asarray(np.random.randn(5, 3), dtype=np.float32)
# write to './temp.bin' and read it as data2
>>> import core_scripts.data_io.io_tools as readwrite
>>> readwrite.f_write_raw_mat(data, './temp.bin')
>>> data2 = readwrite.f_read_raw_mat('./temp.bin', 3)
>>> data - data2
# result should 0
```

More can be found in [./tutorials](./tutorials)


### File directory

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

The script calls different function during model training and inference. 
A detailed flowchat is [./misc/APPENDIX_1.md](./misc/APPENDIX_1.md).

This may be useful if you want to hack on the code.


<a name="miscs"></a>
## 6 On NSF projects (./project/01-nsf)
### Differences from CURRENNT NSF implementation
There may be more, but here are the important ones:

* "Batch-normalization": in CURRENNT, "batch-normalization" is conducted along the length sequence, i.e., assuming each frame as one sample. There is no equivalent implementation on this Pytorch repository;

* No bias in CNN and FF: due to the 1st point, NSF in this repository uses bias=false for CNN and feedforward layers in neural filter blocks, which can be helpful to make the hidden signals around 0;

* smaller learning rate: due to the 1st point, learning rate in this repository is decreased from 0.0003 to a smaller value. Accordingly, more training epochs;

* STFT framing/padding: in CURRENNT, the first frame starts from the 1st step of a signal; in this Pytorch repository (as Librosa), the first frame is centered around the 1st step of a signal, and the frame is padded with 0;

* (minor one) STFT backward: in CURRENNT, STFT backward follows the steps in [this paper](https://ieeexplore.ieee.org/document/8915761/); in Pytorch repository, backward over STFT is done by the Pytorch library. 

* ...

The learning curves look similar to the CURRENNT (cuda) version.
![learning_curve](./misc/fig1_curve.png)


### 24kHz
Most of my experiments are done on 16 kHz waveforms. For 24 kHz waveforms, FIR or sinc digital filters in the model may be changed for better performance:
    
    1. in hn-nsf: lp_v, lp_u, hp_v, and hp_u are calculated on for 16 kHz configurations. For different sampling rate, you may use this online tool http://t-filter.engineerjs.com to get the filter coefficients. In this case, the stop-band for lp_v and lp_u is extended to 12k, while the pass-band for hp_v and hp_u is extended to 12k. The reason is that, no matter what is the sampling rate, the actual formats (in Hz) and spectral of sounds don't change along the sampling rate;

    2. in hn-sinc-nsf and cyc-noise-nsf: for the similar reason above, the cut-off-frequency value (0, 1) should be adjusted. I will try (hidden_feat * 0.2 + uv * 0.4 + 0.3) * 16 / 24 in model.CondModuleHnSincNSF.get_cut_f();


The end