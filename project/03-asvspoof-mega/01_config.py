#!/usr/bin/env python
"""
config.py for project-NN-pytorch/projects

Usage: 
 For training, change Configuration for training stage
 For inference,  change Configuration for inference stage
"""
import os

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#########################################################
## Configuration for training stage
#########################################################

# Name of datasets (any string you wish to use)
#  after data preparation, trn/val_set_name are used to save statistics 
#  about the data sets
trn_set_name = 'asvspoof2019_trn'
val_set_name = 'asvspoof2019_val'

# for convenience
#  we will use resources in this directory
tmp = os.path.dirname(__file__) + '/../../DATA/asvspoof2019_LA'

# File lists (text file, one data name per line, without name extension)
# trin_file_list: list of files for training set
trn_list = tmp + '/scp/train.lst'  
# val_file_list: list of files for validation set. It can be None
val_list = tmp + '/scp/val.lst'

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
input_dirs = [tmp + '/train_dev']

# Dimensions of input features
# input_dims = [dimension_of_feature_1, dimension_of_feature_2, ...]
input_dims = [1]

# File name extension for input features
# input_exts = [name_extention_of_feature_1, ...]
#  If you have waveform in *.flac, please use input_exts = ['.flac']
input_exts = ['.wav']

# Temporal resolution for input features
# input_reso = [reso_feature_1, reso_feature_2, ...]
#  for waveform modeling, temporal resolution of input acoustic features
#  may be = waveform_sampling_rate * frame_shift_of_acoustic_features
#  for example, 80 = 16000 Hz * 5 ms 
input_reso = [1]

# Whether input features should be z-normalized
# input_norm = [normalize_feature_1, normalize_feature_2]
input_norm = [False]
    
# Similar configurations for output features
output_dirs = []
output_dims = [1]
output_exts = ['.bin']
output_reso = [1]
output_norm = [False]

# Waveform sampling rate
#  wav_samp_rate can be None if no waveform data is used
wav_samp_rate = 16000

# Truncating input sequences so that the maximum length = truncate_seq
#  When truncate_seq is larger, more GPU mem required
# If you don't want truncating, please truncate_seq = None
truncate_seq = None

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = None
    

# Optional argument
#  Just a buffer for convenience
#  It can contain anything
optional_argument = [tmp + '/protocol.txt']

#########################################################
## Configuration for inference stage
#########################################################
# similar options to training stage

test_set_name = 'asvspoof2019_test'

# List of test set data
# for convenience, you may directly load test_set list here
test_list = tmp + '/scp/test.lst'

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
test_input_dirs = [tmp + '/eval']

# Directories for output features, which are []
test_output_dirs = []

