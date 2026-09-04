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
__copyright__ = "Copyright 2022, Xin Wang"

#########################################################
## Configuration for training stage
#########################################################

# Name of datasets
#  after data preparation, trn/val_set_name are used to save statistics 
#  about the data sets
trn_set_name = ['asvspoof2019_trn', 'espnet_trn']
val_set_name = ['asvspoof2019_val', 'espnet_val']

# for convenience
tmp1 = os.path.dirname(__file__) + '/../../../DATA/asvspoof2019_LA'
tmp2 = os.path.dirname(__file__) + '/../../../DATA/espnet_release'

# File lists (text file, one data name per line, without name extension)
# trin_file_list: list of files for training set
trn_list = [tmp1 + '/scp/train.lst', tmp2 + '/scp/train.lst']
# val_file_list: list of files for validation set. It can be None
val_list = [tmp1 + '/scp/val.lst', tmp2 + '/scp/val.lst']

# Directories for input features
# input_dirs = [[path_of_feature_1_trainset_1, path_of_feature_2_trainset_1.. ]
#               [path_of_feature_1_trainset_2, path_of_feature_2_trainset_2..]] 
# len(input_dirs) should = len(trn_list) = len(val_list)
#  we assume train and validation data are put in the same sub-directory
input_dirs = [[tmp1 + '/train_dev'], [tmp2 + '/train_dev']]

# Dimensions of input features
# input_dims = [dimension_of_feature_1, dimension_of_feature_2, ...]
# len(input_dims) should be len(input_dirs[0])
input_dims = [1]

# File name extension for input features
# input_exts = [name_extention_of_feature_1, ...]
# Please put ".f0" as the last feature
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
output_dirs = [[] for x in input_dirs]
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
truncate_seq = 64000

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = 8000

# Optional argument
#  Just a buffer for convenience
#  It can contain anything
# Here we use it to hold the path to the protocol.txt 
# They will be loaded by model.py
optional_argument = [tmp1 + '/protocol.txt', tmp2 + '/protocol.txt']

#import augment
#input_trans_fns = [[augment.wav_aug]]
#output_trans_fns = [[]]

#########################################################
## Configuration for inference stage (place holder)
#########################################################
# Please use config_test_*.py inference
# This part is just a place holder

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

