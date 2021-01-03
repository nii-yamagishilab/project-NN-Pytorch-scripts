#!/usr/bin/env python
"""
customized sampler

1. Block shuffler based on sequence length
   Like BinnedLengthSampler in https://github.com/fatchord/WaveRNN
   e.g., data length [1, 2, 3, 4, 5, 6] -> [3,1,2, 6,5,4] if block size =3
"""

from __future__ import absolute_import

import os
import sys
import numpy as np

import torch
import torch.utils.data
import torch.utils.data.sampler as torch_sampler

import core_scripts.math_tools.random_tools as nii_rand_tk
import core_scripts.other_tools.display as nii_warn

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

# name of the sampler
g_str_sampler_bsbl = 'block_shuffle_by_length'

###############################################
# Sampler definition
###############################################

class SamplerBlockShuffleByLen(torch_sampler.Sampler):
    """ Sampler with block shuffle based on sequence length
    e.g., data length [1, 2, 3, 4, 5, 6] -> [3,1,2, 6,5,4] if block size =3
    """
    def __init__(self, buf_dataseq_length, batch_size):
        """ SamplerBlockShuffleByLength(buf_dataseq_length, batch_size)
        args
        ----
          buf_dataseq_length: list or np.array of int, 
                              length of each data in a dataset
          batch_size: int, batch_size
        """
        if batch_size == 1:
            mes = "Sampler block shuffle by length requires batch-size>1"
            nii_warn.f_die(mes)

        # hyper-parameter, just let block_size = batch_size * 3
        self.m_block_size = batch_size * 4
        # idx sorted based on sequence length
        self.m_idx = np.argsort(buf_dataseq_length)
        return
    
    def __iter__(self):
        """ Return a iterator to be iterated. 
        """
        tmp_list = list(self.m_idx.copy())

        # shuffle within each block
        # e.g., [1,2,3,4,5,6], block_size=3 -> [3,1,2,5,4,6]
        nii_rand_tk.f_shuffle_in_block_inplace(tmp_list, self.m_block_size)

        # shuffle blocks
        # e.g., [3,1,2,5,4,6], block_size=3 -> [5,4,6,3,1,2]
        nii_rand_tk.f_shuffle_blocks_inplace(tmp_list, self.m_block_size)

        # return a iterator, list is iterable but not a iterator
        # https://www.programiz.com/python-programming/iterator
        return iter(tmp_list)


    def __len__(self):
        """ Sampler requires __len__
        https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
        """
        return len(self.m_idx)

if __name__ == "__main__":
    print("Definition of customized_sampler")
