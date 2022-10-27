#!/usr/bin/env python
"""
config.py for project-NN-pytorch/projects

Usage: 
 For training, change Configuration for training stage
 For inference, change Configuration for inference stage
"""
import os

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

#########################################################
## Configuration for training stage
#########################################################
# Name of datasets
#  this will be used as the name of cache files created for each set
# 
# Name for the seed training set, in case you merge multiple data sets as
#  a single training set, just specify the name for each subset.
#  Here we only have 1 training subset
trn_set_name = ['asvspoof2019_toyset_trn']
# Name for the development set
val_set_name = ['asvspoof2019_toyset_val']


# For convenience, specify a path to the toy data set
#  because config*.py will be copied into model-*/config_AL_train_toyset/NN
#  we need to use ../../../ 
tmp = os.path.dirname(__file__) + '/../../../DATA/toy_example'

# File list for training and development sets
#   (text file, one file name per line, without name extension)
#   we need to provide one lst for each subset
#   trn_list[n] will correspond to trn_set_name[n]
# for training set
trn_list = [tmp + '/scp/train.lst']
# for development set
val_list = [tmp + '/scp/val.lst']

# Directories for input data
#   We need to provide the path to the directory that saves the input data.
#   We assume waveforms for training and development of one subset 
#   are stored in the same directory.
#   Hence, input_dirs[n] is for trn_set_name[n] and val_set_name[n]
#   
#   If you need to specify a separate val_input_dirs
#   val_input_dirs = [[PATH_TO_DEVELOPMENT_SET]]
# 
#   Each input_dirs[n] is a list, 
#   for example, input_dirs[n] = [wav, speaker_label, augmented_wav, ...]
#   
#   Here, input for each file is a single waveform
input_dirs = [[tmp + '/train_dev']]

# Dimensions of input features
#   What is the dimension of the input feature 
#   len(input_dims) should be equal to len(input_dirs[n]) 
#
#   Here, input for each file is a single waveform, dimension is 1
input_dims = [1]

# File name extension for input features
#   input_exts = [name_extention_of_feature_1, ...]
#   len(input_exts) should be equal to len(input_dirs[n]) 
# 
#   Here, input file extension is .wav
#   We use .wav not .flac
input_exts = ['.wav']

# Temporal resolution for input features
#   This is not relevant for CM but for other projects
#   len(input_reso) should be equal to len(input_dirs[n]) 
#   Here, it is 1 for waveform
input_reso = [1]

# Whether input features should be z-normalized
#   This is not relevant for CM but for other projects
#   len(input_norm) should be equal to len(input_dirs[n]) 
#   Here, it is False for waveform
#   We don't normalize the waveform
input_norm = [False]
    

# Similar configurations for output features
#  Here, we set output to empty because we will load
#  the target labels from protocol rather than output feature
#  '.bin' is also a place holder
output_dirs = [[] for x in input_dirs]
output_dims = [1]
output_exts = ['.bin']
output_reso = [1]
output_norm = [False]


# ===
# For active learning pool data
# ===
# Similar configurations as above
# 
# This is for demonstration, we still use the toy set as pool set.
# And we will merge the trainin and development sets as the pool set
#
# Name of the pool subsets
al_pool_set_name = ['pool_toyset_trn',  'pool_toyset_val']

# list of files for each pool subsets
al_pool_list = [tmp + '/scp/train.lst', tmp + '/scp/val.lst']

# list of input data directories
al_pool_in_dirs = [[tmp + '/train_dev'], 
                   [tmp + '/train_dev']]

al_pool_out_dirs = [[] for x in al_pool_in_dirs]


# ===
# Waveform configuration
# ===

# Waveform sampling rate
#  wav_samp_rate can be None if no waveform data is used
wav_samp_rate = 16000

# Truncating input sequences so that the maximum length = truncate_seq
#  When truncate_seq is larger, more GPU mem required
#  If you don't want truncating, please truncate_seq = None
truncate_seq = 64000

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = 8000
    
# Optional argument
#  This used to load protocol(s)
#  Multiple protocol files can be specified in the list
# 
#  Note that these protocols should cover all the 
#  training, development, and pool set data.
#  Otherwise, the code will raise an error
#  
#  Here, this protocol will cover all the data in the toy set
optional_argument = [tmp + '/protocol.txt']


# ===
# pre-trained SSL model
# ===
# We will load this pre-trained SSL model as the front-end
# 
# path to the SSL model (it is downloaded by 01_download.sh)
ssl_front_end_path = os.path.dirname(__file__) \
                     + '/../../../SSL_pretrained/xlsr_53_56k.pt'
# dimension of the SSL model output
#  this must be provided.
ssl_front_end_out_dim = 1024


#########################################################
## Configuration for inference stage
#########################################################
# This part is not used in this project
# They are place holders

test_set_name = trn_set_name + val_set_name

# List of test set data
# for convenience, you may directly load test_set list here
test_list = trn_list + val_list

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
test_input_dirs = input_dirs * 2

# Directories for output features, which are []
test_output_dirs = [[]] * 2

