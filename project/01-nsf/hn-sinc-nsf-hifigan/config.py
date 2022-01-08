#!/usr/bin/env python
"""
config.py

To merge different corpora (or just one corpus), 

*_set_name are lists
*_list are lists of lists
*_dirs are lists of lists

"""
import os

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

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
input_dirs = [[tmp1 + '/5ms/melspec', tmp1 + '/5ms/f0']]

# Dimensions of input features
# input_dims = [dimension_of_feature_1, dimension_of_feature_2, ...]
input_dims = [80, 1]

# File name extension for input features
# input_exts = [name_extention_of_feature_1, ...]
# Please put ".f0" as the last feature
input_exts = ['.mfbsp', '.f0']

# Temporal resolution for input features
# input_reso = [reso_feature_1, reso_feature_2, ...]
#  for waveform modeling, temporal resolution of input acoustic features
#  may be = waveform_sampling_rate * frame_shift_of_acoustic_features
#  for example, 80 = 16000 Hz * 5 ms 
input_reso = [80, 80]

# Whether input features should be z-normalized
# input_norm = [normalize_feature_1, normalize_feature_2]
input_norm = [True, True]
    
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
truncate_seq = None

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = None
    
# Optional argument
#  Just a buffer for convenience
#  It can contain anything
optional_argument = ['']


# Data transformation function, you can import here
#  these functions are applied before casting np.array data into tensor arrays
#  
#input_trans_fns = [[func_for_mel, fun_for_f0]]
#output_trans_fns = [[func_for_wav]]
#

import numpy as np
def augfunc(data_in, data_out):
    """ Use this function to trim the input and output sequence
    data_in_modified, data_out_modified, length = augfunc(data_in, data_out)
    """ 
    orig_length = data_out.shape[0]
    ups_rate = input_reso[0]
    seg_length = ups_rate * 32
    if orig_length > seg_length:
        # random start
        rs = np.random.randint(0, (orig_length - seg_length))
        rs = rs  // ups_rate * ups_rate
        data_in_new = data_in[rs // ups_rate : (rs + seg_length) // ups_rate]
        data_out_new = data_out[rs : rs + seg_length]
        tmp_len = seg_length
    else:
        data_in_new = data_in
        data_out_new = data_out
        tmp_len = orig_length
    return data_in_new, data_out_new, tmp_len
    
        
    
input_output_trans_fn = augfunc


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
test_input_dirs = [[tmp1 + '/5ms/melspec/', tmp1 + '/5ms/f0/']]

# Directories for output features, which are []
test_output_dirs = [[]]

# Data transformation function, you can import here
#  these functions are applied before casting np.array data into tensor arrays
#
#test_input_trans_fns = [[func_for_mel, fun_for_f0]]
#test_output_trans_fns = [[func_for_wav]]
