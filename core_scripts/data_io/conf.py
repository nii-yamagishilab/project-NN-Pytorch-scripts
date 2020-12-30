#!/usr/bin/env python
"""
config.py

Configurations for data_io

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
import torch.utils.data

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

# ---------------------------
# Numerical configuration
# ---------------------------
# data type for host
h_dtype = np.float32
# data type string format for numpy
h_dtype_str = '<f4'

# data type for device (GPU)
d_dtype = torch.float32

# std_floor
std_floor = 0.00000001


# ---------------------------
# File name configuration
# ---------------------------
# name of the mean/std file for input features
mean_std_i_file = 'mean_std_input.bin'
# name of the mean/std file for output features
mean_std_o_file = 'mean_std_output.bin'
# name of the the uttrerance length file
data_len_file = 'utt_length.dic'


# ---------------------------
# F0 extention and unvoiced value
# ---------------------------
# dictionary: key is F0 file extention, value is unvoiced value
f0_unvoiced_dic = {'.f0' : 0}


# ---------------------------
# Data configuration
# ---------------------------
# minimum length of data. Sequence shorter than this will be ignored
data_seq_min_length = 40

# default configuration for torch.DataLoader
default_loader_conf = {'batch_size':1, 'shuffle':False, 'num_workers':0}
