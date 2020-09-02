# Hands-on materials for NSF models

Xin Wang, National Intitute of Informatics, 2020

## 1. Usage

Readers may check the materials in this order:
* __Part1__: c*.ipynb. These notebooks explain the building blocks of NSF models with codes and figures;

* __Part2__: s*.ipynb. These notebooks define NSF models using building blocks, load pre-trained models, and generate samples;

* __Part3__: Pytorch scripts in https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts. This part will show how to train NSF models on CMU-arctic database.

* __Hands-on-NSF__.pdf: a brief doc. This is to be read while working on Part1, 2, and 3. It can be [downloaded here](https://www.dropbox.com/sh/gf3zp00qvdp3row/AADKGBaLfVSbQVEIEsRUw75Sa/web/pytorch-Hands-on-NSF_v202008.pdf?raw=1)


Alternatively, readers may try this order:
* __Part2__ s*.ipynb: play with pre-trained NSF models;

* __Part1__ c*.ipynb: go through the details of building blocks;

* __Part3__: train NSF models on cmu-arctic using the Pytorch scripts.

Part1 and Part2 can be done on personal computer with web browsers and a few Python tools, no GPU is required.

Part3 requires a Linux GPU server.


## 2. Work on Part 1 and 2
Materials for part 1 and part 2 are written as Jupyter notebooks. Readers can open them on personal desktop or laptop.

#### 2.1 Install softwares

To open and run Jupyter notebooks
1. Please use Python3
2. Please install Jupyter notebook or Jupyter Lab https://jupyter.org/;
3. Please install pytorch (1.4 or later, cpu only), scipy (test on 1.4.1), numpy (test on 1.18.5), Matplotlib (test on 3.1.1);
4. Optionally, [librosa](https://librosa.org) is required to plot rainbow gram in s2_demonstration_music-nsf.ipynb.

If you use [conda](https://docs.conda.io/en/latest/miniconda.html), you may use `./env-cpu.yml` to install softwares and create a python environment:

```sh
conda env create -f ./env-cpu.yml
```

After restarting terminal, you can activate the python environment and run Jupyter lab

```sh
conda activate pytorch-cpu
jupyter lab
```


#### 2.2 Run Jupyter lab

After installing required softwares, please open terminal to go to this directory. Then 

```sh
jupyter lab
```

The browser will open, and you can click the jupyter notebooks and start to run it.

If you are not familiar with Jupyter notebook, please check docs on this [website](https://jupyter.org/). 

#### 2.3 Use HTML files

If you don't want to run Jupyter notebooks, please use any web browser to open the HTML files. HTML files are exported from Jupyter notebooks. They have the same contents as Jupyter notebooks.


## 3. Work on Part 3

Part3 requires a Linux server with GPU card.

#### 3.1 Install softwares
1. Python3
2. pytorch (1.4 or later) with GPU support
3. scipy (test on 1.4.1), numpy (test on 1.18.5)

If *conda* is used to manage python environment, you may try this [env.yml](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/master/env.yml) or this [env2.yml](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts/blob/master/env2.yml) to create an environment called *pytorch-1.4*:

```sh
conda env create -f env.yml
conda activate pytorch-1.4
```

#### 3.2 Run script
1. Download script if you haven't done so
```sh
git clone https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
```


2. Load installed Python environment (for example, pytorch-1.4 installed above)
```
conda activate pytorch-1.4
```


3. Run script in one project folder
```sh
cd project/cyc-noise-nsf-4
bash 00_demo.sh > log_train 2>log_err &
```

This 00_demo.sh will download CMU-arctic data, train one cyclic noise NSF model, and generate speech waveforms. The job will be running in background, and you can check log_train and log_err.

It may take 1 or more days to finish. 

Please check Hands-on-NSF.pdf for details on Part3.


### 4. FAQ

1. Jupyter Lab cannot be opened in the browser
    
    If you a similar problem like [this](https://github.com/jupyterlab/jupyterlab/issues/6921), please change another browser. 
    
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
    