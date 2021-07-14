#!/usr/bin/env python
"""
config.py

This configuration file specifiess the input and output data
"""

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

#########################################################
## Configuration for training stage
#########################################################

# Name of datasets
#  after data preparation, trn/val_set_name are used to save statistics 
#  about the data sets
trn_set_name = ['cmu_all_trn']
val_set_name = ['cmu_all_val']

# for convenience
tmp1 = '../DATA/cmu-arctic-data-set'

# File lists (text file, one data name per line, without name extension)
# trin_file_list: list of files for training set
trn_list = [tmp1 + '/scp/train.lst']
# val_file_list: list of files for validation set. It can be None
val_list = [tmp1 + '/scp/val.lst']

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
input_dirs = [[tmp1 + '/5ms/melspec']]

# Dimensions of input features
# input_dims = [dimension_of_feature_1, dimension_of_feature_2, ...]
input_dims = [80]

# File name extension for input features
# input_exts = [name_extention_of_feature_1, ...]
# Please put ".f0" as the last feature
input_exts = ['.mfbsp']

# Temporal resolution for input features
# input_reso = [reso_feature_1, reso_feature_2, ...]
#  for waveform modeling, temporal resolution of input acoustic features
#  may be = waveform_sampling_rate * frame_shift_of_acoustic_features
#  for example, 80 = 16000 Hz * 5 ms 
input_reso = [80]

# Whether input features should be z-normalized
# input_norm = [normalize_feature_1, normalize_feature_2]
input_norm = [True]
    
# Similar configurations for output features
output_dirs = [[tmp1 + '/wav_16k_norm']]
output_dims = [1]
output_exts = ['.wav']
output_reso = [1]
output_norm = [False]

# Waveform sampling rate
#  wav_samp_rate can be None if no waveform data is used
wav_samp_rate = 16000

# Truncating input sequences so that the maximum length = truncate_seq
#  When truncate_seq is larger, more GPU mem required
# If you don't want truncating, please truncate_seq = None
truncate_seq = 16000

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = 16000
    

# Other optional arguments, the definition of which depends on a specific model
options = {}

#########################################################
## Configuration for inference stage
#########################################################
# similar options to training stage

test_set_name = ['cmu_all_test_tiny']

# List of test set data
# for convenience, you may directly load test_set list here
test_list = [['slt_arctic_b0474']]

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
test_input_dirs = [[tmp1 + '/5ms/melspec/']]

# Directories for output features, which are []
test_output_dirs = [[]]


