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


class data_probe:
    """ data_probe is defined to collect intermediate data
    produced from the inference or training stage
    """
    def __init__(self):
        # a list to store all intermediate data
        self.data_buf = []
        # a single array to store the data
        self.data_concated = None
        # default data convert method
        self.data_convert_method = convert_data_for_debug
        # default method to dump method
        self.data_dump_method = nii_io.pickle_dump
        # dump file name extension
        self.dump_file_ext = '.pkl'
        return

    def add_data(self, input_data):
        """ add_data(input_data)
        Add the input data to a data list. Data will be automatically
        converted by self.data_convert_method

        input
        -----
          input_data: tensor, or numpy.array        
        """
        self.data_buf.append(self.data_convert_method(input_data))
        return

    def _merge_data(self):
        """ merge_data()
        Merge the data in the list to a big numpy array table.
        Follow the convention of this project, we assume data has shape
        (batchsize, length, feat_dim)
        """
        self.data_concated = np.concatenate(self.data_buf, axis=1)
        return

    def _dump_file_path(self, file_path):
        """ add additional infor to the ump file path
        """
        time_tag = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return file_path + '_' + time_tag + self.dump_file_ext

    def dump(self, output_path='./debug/data_dump'):
        """ dump(output_path='./debug/data_dump')
        input
        -----
          output_path: str, path to store the dumped data
        """
        # add additional infor to output_path name
        output_path_new = self._dump_file_path(output_path)
        try:
            os.mkdir(os.path.dirname(output_path_new))
        except OSError:
            pass

        ## merge data if it has not been done
        #if self.data_concated is None:
        #    self.merge_data()
        #nii_io.f_write_raw_mat(self.data_concated, output_path_new)

        self.data_dump_method(self.data_buf, output_path_new)
        print("Data dumped to {:s}".format(output_path_new))
        
        self.data_concated = None
        return

if __name__ == '__main__':
    print("Debugging tools")
