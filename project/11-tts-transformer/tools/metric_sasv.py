#!/usr/bin/python3
"""
Modules for SASV evaluation
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

from tools import metric_biometric

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"

def compute_sasv_eer(scores, labels, 
                     target_label = 1, 
                     nontar_label = 2,
                     spoof_label = 0):
    """
    sasv_eer, asv_eer, cm_eer, sasv_thr, asv_thr, cm_thr = compute_sasv_eer(
       scores, labels target_label = 1, nontar_label = 2, spoof_label = 0)

    input
    -----
      scores:       np.array, (#N, ), scores of trial lists with N entries
      labels:       np.array, (#N, ), target label of trial lists
      target_label: int, value of the target label, default 1
      nontar_label: int, value of the non-target label, default 2
      spoof_label:  int, value of the spoofed label, default 0

    output
    ------
      sasv_eer:     float, SASV EER value
      asv_eer:      float, ASV EER value
      cm_eer:       float, CM EER value
      sasv_thr:     float, threshold for SASV EER
      asv_thr:      float, threshold for ASV EER
      cm_thr:       float, threshold for CM EER
    """

    # get the scores
    tar_scores = scores[np.where(labels == target_label)]
    other_scores = scores[np.where(labels != target_label)]
    nontar_scores = scores[np.where(labels == nontar_label)]
    spoof_scores = scores[np.where(labels == spoof_label)]
    
    # comptue the EERs
    sasv_eer, sasv_thr = metric_biometric.compute_eer(tar_scores, other_scores)
    asv_eer, asv_thr = metric_biometric.compute_eer(tar_scores, nontar_scores)
    cm_eer, cm_thr = metric_biometric.compute_eer(tar_scores, spoof_scores)

    return sasv_eer, asv_eer, cm_eer, sasv_thr, asv_thr, cm_thr


if __name__ == "__main__":
    print(__doc__)
