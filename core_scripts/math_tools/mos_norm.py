#!/usr/bin/env python
"""
mos_norm.py

MOS score normalization tools to analysis listening test results.

This is dumped from scripts

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import random

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

####################
# rank-normalization
#
# [1] Andrew Rosenberg, and Bhuvana Ramabhadran. 2017. 
#     Bias and Statistical Significance in Evaluating Speech Synthesis with 
#     Mean Opinion Scores. In Proc. Interspeech, 3976–3980.
####################

def _rank_norm_mapping_function(data, data_range=[1, 10]):
    """ data_map = _rank_norm_mapping_function(data, data_range=[1, 10])
    
    input
    -----
      data: np.array, (N,), MOS scores to compute the mapping function
      data_range: [int, int], 
            the lowest and highest possible values for the data
    output
    ------
      data_map: dict,  data_map[SCORE] -> rank_normed_score
    """
    # 
    data_map = {}
    
    # step1. sort the scores
    sorted_data = np.sort(data, kind='quicksort')
    
    # step2. assigna rank (staring from 1) to each score
    sorted_rank = np.arange(len(sorted_data)) + 1
    
    # step3. compute the normalized scores
    for score_value in np.arange(data_range[0], data_range[1]+1):
        # indices are the ranks of the score 
        #  (the score may appear multiple times, indices are their ranks)
        indices = sorted_rank[sorted_data == score_value]
        if indices.shape[0]:
            # (mean of rank - 1) / N
            data_map[score_value] = (np.mean(indices) - 1) / data.shape[0]
        else:
            # if the score does not appear in the data
            data_map[score_value] = -1

    return data_map

def rank_norm(data, data_range):
    """ ranked_score = rank_norm(data)
    input
    -----
      data: np.array, (N,), MOS scores to compute the mapping function
      data_range: [int, int], 
                 the lowest and highest possible values for the data
                 (even though the value may not appear in the data)
                 It can be [1, 5] for most of the MOS test
    output
    ------
      data_map: dict,  data_map[SCORE] -> rank_normed_score
      
    Example
    -------
    >>> data = np.array([2, 1, 2, 10, 4, 5, 6, 4, 5, 7])
    >>> rank_norm(data, [1, 10])
    [0.15, 0.0, 0.15, 0.9, 0.35, 0.55, 0.7, 0.35, 0.55, 0.8]
    """
    data_map = _rank_norm_mapping_function(data, data_range)
    data_new = [data_map[x] for x in data]
    return data_new


if __name__ == "__main__":
    print("Tools to normalize MOS scores")
