#!/usr/bin/env python
"""
Library of utilities for printing latex table
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#####################
## Latex table
#####################

def print_table(data_array, column_tag, row_tag):
    """
    print a latex table given the data and tags
    
    input
    -----
      data_array: np.array [M, N]
      column_tag: tag on top of the matrix
      row_tag: tags in each row
    
    output
    ------
      None
    """
    print(r"\begin{tabular}{" + ''.join(['c' for x in column_tag + ['']]) + r"}")
    def print_one_row(content_buffer):
        print(" & ".join(content_buffer) + r"\\ ")

    print_one_row([""] + column_tag)

    row = data_array.shape[0]
    col = data_array.shape[1]
    for row_idx in np.arange(row):
        row_content = [row_tag[row_idx]]
        for col_idx in np.arange(col):
            row_content.append("%1.3f" % (data_array[row_idx, col_idx]))
        print_one_row(row_content)
    print(r"\end{tabular}")
    return
