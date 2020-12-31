#!/usr/bin/env python
"""
debug.py

Tools to help debugging

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import datetime
import torch

from core_scripts.data_io import io_tools as nii_io

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def qw(data, path=None):
    """ write data tensor into a temporary buffer
    """
    if path is None:
        path = 'debug/temp.bin'
    
    try:
        os.mkdir(os.path.dirname(path))
    except OSError:
        pass
    
    if hasattr(data, 'detach'):
        nii_io.f_write_raw_mat(data.detach().to('cpu').numpy(), path)
    elif hasattr(data, 'cpu'):
        nii_io.f_write_raw_mat(data.to('cpu').numpy(), path)
    elif hasattr(data, 'numpy'):
        nii_io.f_write_raw_mat(data.numpy(), path)
    else:
        nii_io.f_write_raw_mat(data, path)


if __name__ == '__main__':
    print("Debugging tools")
