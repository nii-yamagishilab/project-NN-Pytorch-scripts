#!/usr/bin/env python
"""
startup_config

Startup configuration utilities

"""
from __future__ import absolute_import

import os
import sys
import torch
import importlib
import random
import numpy as np

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def set_random_seed(random_seed, args=None):
    """ set_random_seed(random_seed, args=None)
    
    Set the random_seed for numpy, python, and cudnn
    
    input
    -----
      random_seed: integer random seed
      args: argue parser
    """
    
    # initialization                                       
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    #For torch.backends.cudnn.deterministic
    #Note: this default configuration may result in RuntimeError
    #see https://pytorch.org/docs/stable/notes/randomness.html    
    if args is None:
        cudnn_deterministic = True
        cudnn_benchmark = False
    else:
        cudnn_deterministic = args.cudnn_deterministic_toggle
        cudnn_benchmark = args.cudnn_benchmark_toggle
    
        if not cudnn_deterministic:
            print("cudnn_deterministic set to False")
        if cudnn_benchmark:
            print("cudnn_benchmark set to True")
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
    return
