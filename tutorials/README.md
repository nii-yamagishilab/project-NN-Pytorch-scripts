# Tutorials on Neural vocoders

Xin Wang, National Institute of Informatics, 2022

**The latest notebooks are maintained on Goolge colab, please click [here](https://colab.research.google.com/drive/1EO-ggi1U9f2zXwTiqg7AEljVx11JKta7?usp=sharing).**

**Notebooks in this directory are not maintained anymore.**


This hands-on material covers only the basics of some neural vocoders. Models and implementations are for the tutorial, therefore lacking intensive tuning and optimization. Neither am I good at that. If you have ideas on how to improve, your feedback is appreciated!

**Table of Contents:**

* [Usage](#usage)
* [Work on Part 1 and 2](#part12)
* [Work Part 3](#part3)
* [FAQ](#faq)


If you are in the root directory of the hands-on package, please go to the tutorial folder

```sh
# go to the tutorial folder
cd ./tutorials

# you should be able to see many ipynb files
ls *.ipynb
c01_data_format.ipynb 
...
```

## <a name="usage"></a>1. Usage

Readers may check the materials in this order:

* __Part1__: c*.ipynb explain 1) Pytorch conventions in this tutorial; 2) building blocks of common neural vocoders;

* __Part2__: s*.ipynb are on three types of models: NSF, WaveNet, and WaveGlow. They explain details of each module and contain analysis-synthesis samples using  pre-trained models;

* __Part3__: `../project` include scripts to train NSF, WaveNet, and WaveGlow using CMU-ARCTIC database.

* __Hands-on-NSF__.pdf: an optional doc on the tutorial. You may read it while working on Part1, 2, and 3. It can be [downloaded here](https://www.dropbox.com/sh/gf3zp00qvdp3row/AACeanvFQD5Gyu3a5I5jIV_-a/web/Hands-on-neural-vocoders-spcc2021.pdf?dl=1)


Alternatively, readers may try this order:

* __Part2__: play with pre-trained models in s*.ipynb;

* __Part1__: go through the details of building blocks in c*.ipynb;

* __Part3__: train to train your models on CMU-ARCTIC database.

Part1 and Part2 can be done on your personal computer with web browsers and a few Python tools; no GPU is required. Note that this tutorial has not been checked on a Windows PC. If you have any issues, please contact with me.

Part3 requires a Linux GPU server.

## <a name="part12"></a>2. Work on Part 1 and 2
Materials for part 1 and part 2 are in Jupyter notebooks. Readers can open them on personal desktop or laptop.

#### 2.1 Setup

If you use [conda](https://docs.conda.io/en/latest/miniconda.html), please use use `./env-cpu.yml` to install software:

```sh
# install software dependency
conda env create -f ./env-cpu.yml

# activate environment and add PYTHONPATH
source ./env-cpu.sh

# Download pre-trained models
bash 00_download_models.sh
```

If you don't use conda, you may install software in env-cpu.yml manually.

Jupyterlab since version 3.0 shows Table of Contents for the opened notebook. This is a highly recommended function. For more details, please check [this page](https://jupyterlab.readthedocs.io/en/stable/user/toc.html).

Pre-trained models of WaveGlow and Blow are quite large. They are shared through ~~Dropbox~~ [Zenodo](https://doi.org/10.5281/zenodo.6349636). If you cannot access ~~Dropbox~~ Zenodo, please let me know. You may also skip the pre-trained models if you don't want to try the analysis-synthesis examples in s2*.ipynb.

#### 2.2 Run Jupyter lab
Open the sh terminal and run shell command

```sh
jupyter lab
```

The browser will open, and the Jupyter notebook GUI will be displayed. You can click a jupyter notebook and start to run it. If you are not familiar with Jupyter notebook, please check docs on this [website](https://jupyter.org/). 


#### 2.3 Use HTML files

If you don't want to run Jupyter notebooks, please use any web browser to browse the static HTML files. HTML files are exported from the Jupyter notebooks. 


## <a name="part3"></a>3. Work on Part 3

Part 3 requires a Linux server with GPU card.

How-to is written in `../README`. For convenience, I summarize it here:

#### 3.1 Install software with support for GPU
If you use conda

```sh
# Go to the root directory of this repo
cd ..

# Install Pytorch with GPU and other software
conda env create -f env.yml

```

You may also manually install the software in env.yml.

#### 3.2 Run script

For example:

```sh
# Inside the root directory of this repo
cd project/01-nsf/cyc-noise-nsf-4

# Run script
bash 00_demo.sh > log_train 2>log_err &
```

This 00_demo.sh will download CMU-ARCTIC data, train one NSF model, and generate speech waveforms. The job will be running in the background. Please check log_train and log_err to monitor the process.

It may take 1 or more days to finish. 

You can check `../README` and Part3 in Hands-on-NSF.pdf for more details.


## <a name="faq"></a>4. FAQ

1. Jupyter Lab cannot be opened in the browser
   
    If you a similar problem like [this](https://github.com/jupyterlab/jupyterlab/issues/6921), please use another browser. 
    
    Simply copy the http address shown in the terminal that runs command Jupyter Lab and open it in another browser. 
    
    The http address can be found in the terminal:
    
    ```
    To access the notebook, open this file in a browser:
        file://...
    or copy and paste one of these URLs:
       http://localhost:8888/...
    or http://127.0.0.1:8888/...
    ```


2. Jupyter Lab does not have the right kernel

    If you have installed Jupyter Lab before, you may not find the Python kernel corresponding to the newly created Python environment. 
    
    Please check this [page](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-different-environments) to add the kernel to Jupyter Lab.


3. If you want to run the Jupyter notebook on a remote server

   You may check how-to on this [webpage](https://docs.anaconda.com/anaconda/user-guide/tasks/remote-jupyter-notebook/). 

4. If you are asked to submit Jupyter token or password
   
   Please check this [page](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html#automatic-password-setup) and set up a password

   ```sh

   $ jupyter notebook password
   Enter password:  
   Verify password: 
   ```

The end