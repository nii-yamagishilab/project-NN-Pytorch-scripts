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
import numpy as np
import torch

from core_scripts.data_io import io_tools as nii_io

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

def convert_data_for_debug(data):
    """ data_new = convert_data_for_debug(data)
    For debugging, it is convenient to has a data in numpy format

    Args
    ----
      data: tensor

    Return
    ------
      data_new: numpy array
    """
    if hasattr(data, 'detach'):
        return data.detach().to('cpu').numpy()
    elif hasattr(data, 'cpu'):
        return data.to('cpu').numpy()
    elif hasattr(data, 'numpy'):
        return data.numpy()
    else:
        return data

def qw(data, path=None):
    """ write data tensor into a temporary buffer
    
    Args
    ----
      data: a pytorch tensor or numpy tensor
      path: str, path to be write the data
            if None, it will be "./debug/temp.bin"
    Return
    ------
      None
    """
    if path is None:
        path = 'debug/temp.bin'
    
    try:
        os.mkdir(os.path.dirname(path))
    except OSError:
        pass

    # write to IO
    nii_io.f_write_raw_mat(convert_data_for_debug(data), path)
    return
                        
def check_para(pt_model):
    """ check_para(pt_model)
    Quickly check the statistics on the parameters of the model
    
    Args
    ----
      pt_model: a Pytorch model defined based on torch.nn.Module
    
    Return
    ------
      None
    """
    mean_buf = [p.mean() for p in pt_model.parameters() if p.requires_grad]
    std_buf = [p.std() for p in pt_model.parameters() if p.requires_grad]
    print(np.array([convert_data_for_debug(x) for x in mean_buf]))
    print(np.array([convert_data_for_debug(x) for x in std_buf]))
    return


if __name__ == '__main__':
    print("Debugging tools")
