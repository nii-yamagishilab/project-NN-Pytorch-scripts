#!/usr/bin/env python
"""
config.py 

Configuration file for training and evaluation.

Usage: 
 For training, change Configuration for training stage
 For inference,  change Configuration for inference stage

Please follow the instruction below to config this file
"""
import os

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

#########################################################
## Configuration for training stage
#########################################################

# Name of datasets (any string you wish to use)
#  after data preparation, trn/val_set_name are used to save statistics 
#  about the data sets
trn_set_name = 'asvspoof2021_trn_toy'
val_set_name = 'asvspoof2021_val_toy'

# for convenience
#  we will use resources in this directory
tmp = os.path.dirname(__file__) + '/../DATA/toy_example'

# File lists (text file, one data name per line, without name extension)
# trin_file_list: list of files for training set
trn_list = tmp + '/scp/train.lst'  
# val_file_list: list of files for validation set. It can be None
val_list = tmp + '/scp/val.lst'

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
#  here, we only use waveform as input to the code
input_dirs = [tmp + '/train_dev']

# Dimensions of input features
#  input_dims = [dimension_of_feature_1, dimension_of_feature_2, ...]
#  here, we only use waveform as input to the code, and the dimension
#  of waveform data is 1
input_dims = [1]

# File name extension for input features
#  input_exts = [name_extention_of_feature_1, ...]
#  here, we only use waveform, thus it is ".wav"
input_exts = ['.wav']

# Temporal resolution for input features
# input_reso = [reso_feature_1, reso_feature_2, ...]
#  this is used for other projects.
#  for waveform, we set it to 1
input_reso = [1]

# Whether input features should be z-normalized
#  input_norm = [normalize_feature_1, normalize_feature_2]
#  we don't z-normalize the waveform
input_norm = [False]
    
# This is for other projects,
#  we don't load target features for ASVspoof models using these config
#  we read target labels of ASVspoof trials in mode.py
#  but we need to fill in some placehoders
output_dirs = []
output_dims = [1]
output_exts = ['.bin']
output_reso = [1]
output_norm = [False]

# Waveform sampling rate
#  wav_samp_rate can be None if no waveform data is used
#  ASVspoof uses 16000 Hz
wav_samp_rate = 16000

# Truncating input sequences so that the maximum length = truncate_seq
#  When truncate_seq is larger, more GPU mem required
#  If you don't want truncating, please truncate_seq = None
#  For ASVspoof, we don't do truncate here
truncate_seq = None

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
#  For ASVspoof, we don't set minimum length of input trial
minimum_len = None
    

# Optional argument
#  We will use this optional_argument to read protocol file
#  When evaluating on a eval set without protocol file, set this to ['']
optional_argument = [tmp + '/protocol.txt']

#########################################################
## Configuration for inference stage
#########################################################
# similar options to training stage

test_set_name = 'asvspoof2021_test_toy'

# List of test set data
# for convenience, you may directly load test_set list here
test_list = tmp + '/scp/test.lst'

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
# directory of the evaluation set waveform
test_input_dirs = [tmp + '/eval']

# Directories for output features, which are []
test_output_dirs = []

