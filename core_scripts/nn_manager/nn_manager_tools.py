#!/usr/bin/env python
"""
nn_manager

A simple wrapper to run the training / testing process

"""
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import torch

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#############################################################

def f_state_dict_wrapper(state_dict, data_parallel=False):
    """ a wrapper to take care of state_dict when using DataParallism

    f_model_load_wrapper(state_dict, data_parallel):
    state_dict: pytorch state_dict
    data_parallel: whether DataParallel is used
    
    https://discuss.pytorch.org/t/solved-keyerror-unexpected-
    key-module-encoder-embedding-weight-in-state-dict/1686/3
    """
    if data_parallel is True:
        # if data_parallel is used
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('module'):
                # if key is not starting with module, add it
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('module'):
                name = k
            else:
                # remove module.
                name = k[7:] 
            new_state_dict[name] = v
        return new_state_dict

