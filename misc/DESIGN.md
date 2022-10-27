
# Project design and conventions


## Data format

* Waveform: 16/32-bit PCM or 32-bit float WAV that can be read by [scipy.io.wavfile.read](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.read.html) 

* Other data: binary, float-32bit, little endian ([numpy dtype <f4](https://numpy.org/doc/1.18/reference/generated/numpy.dtype.html)). The data can be read in python by:

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

More instructions can be found in this Jupyter notebook [here](https://colab.research.google.com/drive/1EO-ggi1U9f2zXwTiqg7AEljVx11JKta7?usp=sharing).


## Files in this repository

Name | Function
------------ | -------------
./core_scripts | scripts (Numpy or Pytorch code) to manage the training process, data io, etc.
./core_modules | finalized pytorch modules 
./sandbox | new functions and modules to be test
./project | project directories, and each folder correspond to one model for one dataset
./project/\*/\*/main.py | script to load data and run training and inference
./project/\*/\*/model.py | model definition based on Pytorch APIs
./project/\*/\*/config.py | configurations for training/val/test set data


The motivation is to separate the training and inference process, the model definition, and the data configuration. For example:

* To define a new model, change model.py

* To run on a new database, change config.py

The separation is not always strictly followed. 


## How the script works

The script starts with main.py and calls different functions for model training and inference. 

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

A detailed flowchart is [APPENDIX_1.md](./APPENDIX_1.md). 

---
That's all
