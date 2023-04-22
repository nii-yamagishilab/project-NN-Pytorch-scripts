
# Neural vocoders

This project is Pytorch re-implementation of a few neural waveform models.

* They are used in the hands-on tutorial on neural vocoders [../../tutorials/b1_neural_vocoder/README.md](../../tutorials/b1_neural_vocoder/README.md)

* Note that the tutorial **chapter_a3_pretrained_vocoders.ipynb** includes pre-trained HiFiGAN and WaveGlow on VoxCeleb2 dev and other speech datasets [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xObWejhqcdSxFAjfWI7sudwPPMoCx-vA?usp=sharing).

* The code to extract the input Mel-spectrogram and F0 are included in the above tutorial and notebooks as well. This folder assumes that the input Mel-spectrogram and F0 have been prepared in advance. 

**It is better to check the tutorials before diving into this project**.

## Quick start

Choose a model
```sh
cd wavenet
```

Setup the dependency and PYTHONPATH
```sh
source ../../../env.sh 
```

Run script
```sh
bash 00_demo.sh
```

This script will
1. download the CMU database and pre-extracted features;
2. generate samples using a pre-trained model and the pre-extracted features;
3. trains a new model on the data.

This may take a few days or more. You may try to run `00_demo.sh` in the background.
```sh
bash 00_demo.sh > log_batch 2>&1 &
```

## Folder structure


```sh
| - DATA: folder to store data 
|    | = cmu-arctic-data-set: data set for 00_demo.sh
|     
| - wavenet: WaveNet
|    |    This is implemented using blocks in sandbox/block_wavenet.py
|    |
|    | -  00_demo.sh: script to download data and pre-trained model, run synthesis
|    |          and training
|    | -  main.py: main function
|    | -  config.py: configuration of input and output data
|    | -  model.py: wrapper of model
|    | -  __pre_trained: folder to store pre-trained models by Xin
|    |        |
|    |        | - trained_network.pt: pre-trained model by Xin
|    |        | - log_train: training log of the pre-trained model
|    |        | - output_xin: sample waveform synthesized using pre-trained model by xin
|    |        | = output: sample waveform synthesized using pre-trained model
|    |
|    | =  cmu-*.dic: cached information on the data set (automatically created)
|    | =  epochNNN.pt: trained model after NNN-th epoch
|    | =  trained_network.pt: currently the best model (best on validation set)
|    | =  log_train.txt: training log
|    | =  log_err.txt: error log (it also contains the training loss for each trial)
|
| - waveglow: WaveGlow
|         This is implemented using blocks in sandbox/block_waveglow.py
|         Post-filtering / de-noising in the official WaveGlow is NOT included
|         
| - waveglow-2: similar to waveglow above but uses a legacy implementation of 
|         WaveNet layer (see option self.flag_affine_legacy_implementation in model.py)
|
|
| - blow: a flow-based waveform model for voice conversion
| 
|
| - ilpcnet: LPCNet using with a Gaussian distribution on waveform
|            The difference is that we will predict LPC coefficients from the input
|            Mel-spectrogram
|
| - HiFi-GAN: a HiFi-GAN vocoder adopted from the official implementation
|
```

Folder and file annotated by '=' will be produced after running the 00_demo.sh.
The folder structure of other models is similar to that under `wavenet`.



## Note
1. Models here are created mainly for tutorial purposes. They are not trying to reproduce the performance in paper or official implementation. Neither are the pre-trained models tuned best to the database due to limited computation resources.

2. For readers who want to improve the results, please feel free to change the
model definition and (hyper-)parameter settings.

3. Also feel free to use the model.py and other model definition modules for your 
own projects.

4. To-do list:

   * synthesis speed of WaveNet is very slow. In CUDA/C++ implementation (CURRENNT), synthesis speed was acceptable. However, I am not sure how to avoid the overhead in
       Pytorch. Note that I've used the buffer to cache intermediate data for dilated conv (see sandbox/block_nn.py Conv1dForARModel.forward);

   * ~~waveglow does not include the post-filtering / de-noising step;~~

   * ...

4. `model.py` in each project was created from a common template. It may include functions unnecessary for the model but required for the script to run. For a detailed explanation on how the script works, please check ../../README.md


Reference
---------

* WaveNet: van den Oord, A. et al. WaveNet: A generative model for raw audio.  arXiv Prepr. arXiv1609.03499 (2016) 

* WaveGlow: Prenger, R., Valle, R. & Catanzaro, B. WaveGlow: A Flow-based Generative Network for Speech Synthesis. in Proc. ICASSP 3617–3621 (2019). https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/

* Blow: J. Serra, S. Pascual, & C. Segura (2019). Blow: a single-scale hyperconditioned flow for non-parallel raw-audio voice conversion. In Advances in Neural Information Processing Systems (NeurIPS). 2019. https://blowconversions.github.io/

* iLPCNet: M.-J. Hwang, E. Song, R. Yamamoto, F. Soong, and H.-G. Kang, Improving LPCNet-based text-to-speech with linear prediction-structured mixture density network. in Proc. ICASSP, 2020, pp. 7219–7223.

* HiFi-GAN: Jungil Kong, Jaehyeon Kim, Jaekyoung Bae, HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis, Proc. NIPS. HifiGAN is based on code in from https://github.com/jik876/hifi-gan

---
That's all
   