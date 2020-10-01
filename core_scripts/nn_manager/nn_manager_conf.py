#!/usr/bin/env python
"""
nn_manager_conf

A few definitions of nn_manager

"""
from __future__ import print_function

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


# Name of keys in checkpoint
# Epochs saved as checkpoint will have these fields
class CheckPointKey:
    state_dict = 'state_dict'
    info = 'info'
    optimizer = 'optimizer' 
    trnlog = 'train_log'
    vallog = 'val_log'


# Methods that a Model should have 
nn_model_keywords = {
    'prepare_mean_std': "method to initialize mean/std",
    'normalize_input': "method to normalize input features",
    'normalize_target': "method to normalize target features",
    'denormalize_output': "method to de-normalize output features",
    'forward': "main method for forward"
}
