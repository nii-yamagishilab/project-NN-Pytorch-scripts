#!/usr/bin/env python
"""
config.py

To merge different corpora (or just one corpus), 

*_set_name are lists
*_list are lists of lists
*_dirs are lists of lists

"""

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
# input_dirs = [[path_of_feature_1, path_of_feature_2, ..., ],
#               [path_of_feature_1, path_of_feature_2, ..., ], ...]
# len(input_dirs) should be equal to len(trn_set_name) and len(val_set_name)
# iput_dirs[N] is the path for trn_set_name[N] and val_set_name[N]
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
truncate_seq = 2400

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = 2400
    

# Other optional arguments, the definition of which depends on a specific model
# This may be imported by model.py
# 
# Here, we specify the framelenth, frameshift, and lpc_order for LPC analysis
# Note that these configurations can be different from what we used to extract
# the input acoustic features, but they should be compatible.
#
# For example, input acoustic features are extracted with a frameshift of 80
# then, we can use frameshift 160 here. Since frameshift for LPC features
# is x2 of input features, the model will down-sample the input features.
# See line 148 and 167 in model.py
options = {'framelength': 160, 'frameshift': 160, 'lpc_order': 15}

# input_trans_fns and output_trans_fns are used by the DataSet.
# When loading the data from the disk, the input data will be transformed 
# by input_trans_fns, output will be transformed by output_trans_fns.
# This is done before converting the data into pytorch tensor, which
# is more convient.
# 
# They are used by ../../../core_scripts/data_io/default_data_io.py:
# f_post_data_process()
# If you want to debug into these functions, remember turn off multi workers
# $: python -m pdb main.py --num-workers 0
#
# These are the rules:
# len(input_trans_fns) == len(output_trans_fns) == len(trn_set_name)
# input_trans_fns[N] is for trn_set_name[N] and val_set_name[N]
# len(input_trans_fns[N]) == len(input_exts)
# len(output_trans_fns[N]) == len(output_exts)
# input_trans_fns[N][M] is for the input_exts[M] of trn_set_name[N]
# ...
import block_lpcnet
input_trans_fns = [[]]
output_trans_fns = [[lambda x: block_lpcnet.get_excit(x, 160, 160, 15)]]

#########################################################
## Configuration for inference stage
#########################################################
# similar options to training stage

test_set_name = ['cmu_all_test_tiny']

# List of test set data (in the same way as trn_list or val_list)
# Or, for convenience, you may directly load test_set list
test_list = [['slt_arctic_b0474']]

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
test_input_dirs = [[tmp1 + '/5ms/melspec', tmp1 + '/5ms/f0']]

# Directories for output features, which are [[]]
test_output_dirs = [[]]

# 
#test_output_trans_fns = output_trans_fns
