#!/usr/bin/env python
"""
random.py

Tools related to randomness

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import random

import core_scripts.other_tools.display as nii_display

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

##############
# Shuffling tools
##############

def f_shuffle_slice_inplace(input_list, slice_start=None, slice_stop=None):
    """ shuffle_slice(input_list, slice_start, slice_stop)
    
    Shuffling input list (in place) in the range specified by slice_start
    and slice_stop.

    Based on Knuth shuffling 
    https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle

    input
    -----
      input_list: list
      slice_start: int, start idx of the range to be shuffled
      slice_end: int, end idx of the range to be shuffled
      
      Both slice_start and slice_end should be in the style of python index
      e.g., shuffle_slice(input_list, 0, N) will shuffle the slice input[0:N]
    
      When slice_start / slice_stop is None,
      slice_start = 0 / slice_stop = len(input_list)

    output
    ------
      none: shuffling is done in place
    """
    if slice_start is None or slice_start < 0:
        slice_start = 0 
    if slice_stop is None or slice_stop > len(input_list):
        slice_stop = len(input_list)
        
    idx = slice_start
    while (idx < slice_stop - 1):
        idx_swap = random.randrange(idx, slice_stop)
        # naive swap
        tmp = input_list[idx_swap]
        input_list[idx_swap] = input_list[idx]
        input_list[idx] = tmp
        idx += 1
    return

def f_shuffle_in_block_inplace(input_list, block_size):
    """
    f_shuffle_in_block_inplace(input_list, block_size)
    
    Shuffle the input list (in place) by dividing the list input blocks and 
    shuffling within each block
    
    Example:
    >>> data = [1,2,3,4,5,6]
    >>> random_tools.f_shuffle_in_block_inplace(data, 3)
    >>> data
    [3, 1, 2, 5, 4, 6]

    input
    -----
      input_list: input list
      block_size: int
    
    output
    ------
      None: shuffling is done in place
    """
    if block_size <= 1:
        # no need to shuffle if block size if 1
        return
    else:
        list_length = len(input_list)
        # range( -(- x // y) ) -> int(ceil(x / y))
        for iter_idx in range( -(-list_length // block_size) ):
            # shuffle within each block
            f_shuffle_slice_inplace(
                input_list, iter_idx * block_size, (iter_idx+1) * block_size)
        return

def f_shuffle_blocks_inplace(input_list, block_size):
    """ 
    f_shuffle_blocks_inplace(input_list, block_size)
    
    Shuffle the input list (in place) by dividing the list input blocks and 
    shuffling blocks
    
    Example:
     >> data = np.arange(1, 7)
     >> f_shuffle_blocks_inplace(data, 3)
     >> print(data)
     [4 5 6 1 2 3]

    input
    -----
      input_list: input list
      block_size: int
    
    output
    ------
      None: shuffling is done in place
    """
    # new list
    tmp_list = input_list.copy()

    block_number = len(input_list) // block_size
    
    shuffle_block_idx = [x for x in range(block_number)]
    random.shuffle(shuffle_block_idx)

    new_idx = None
    for iter_idx in range(block_size * block_number):
        block_idx = iter_idx // block_size
        in_block_idx = iter_idx % block_size
        new_idx = shuffle_block_idx[block_idx] * block_size + in_block_idx
        input_list[iter_idx] = tmp_list[new_idx]
    return

if __name__ == "__main__":
    print("Definition of randomness tools")
