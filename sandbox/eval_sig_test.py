#!/usr/bin/env python
"""
eval_sig_test

Utilities for statistical test on EERs
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import core_scripts.math_tools.sig_test as sig_test

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"



##################################
# Functions for computing z-values
#
##################################
def compute_z_independent(far_a, frr_a, far_b, frr_b, NI, NC):
    """z = compute_HTER_independent(hter_a, hter_b, NI, NC)
    
    Bengio, S. & Mariéthoz, J. A statistical significance test for 
    person authentication. in Proc. Odyssey (2004). 
    
    Fig2. independent case
    
    input
    -----
      far_a: float, far of system a, which be >=0 and <=1
      frr_a: float, frr of system a, which be >=0 and <=1
      far_b: float, far of system b, which be >=0 and <=1
      frr_b: float, frr of system b, which be >=0 and <=1
      NI: int, the number of impostor accesses.
      NC: int, the number of client accesses.
      
    output
    ------
      z: float, statitics of the hypothesis test
    """
    # 
    hter_a = (far_a + frr_a)/2
    hter_b = (far_b + frr_b)/2
    denominator  = (far_a * (1 - far_a) + far_b * (1 - far_b)) / 4 / NI 
    denominator += (frr_a * (1 - frr_a) + frr_b * (1 - frr_b)) / 4 / NC 
    return np.abs(hter_a - hter_b) / np.sqrt(denominator)
    
    
def compute_z_dependent(far_ab, frr_ab, far_ba, frr_ba, NI, NC):
    """z = compute_HTER_independent(hter_a, hter_b, NI, NC)
    
    Bengio, S. & Mariéthoz, J. A statistical significance test for 
    person authentication. in Proc. Odyssey (2004). 
    
    Fig2. dependent case
    
    input
    -----
      far_ab: float, see paper
      frr_ab: float, see paper 
      far_ba: float, see paper 
      frr_ba: float, see paper 
      NI: int, the number of impostor accesses.
      NC: int, the number of client accesses.
      
    output
    ------
      z: float, statitics of the hypothesis test
    """
    # 
    if far_ab == far_ba and frr_ab == frr_ba:
        return 0
    else:
        denominator = np.sqrt((far_ab + far_ba) / (4 * NI) 
                              + (frr_ab + frr_ba) / (4 * NC))
        return np.abs(far_ab + frr_ab - far_ba - frr_ba) / denominator


def get_eer(scores_positive, scores_negative):
    """eer, threshold = get_eer(scores_positive, scores_negative)
    
    compute Equal Error Rate given input scores
    
    input
    -----
      scores_positive: np.array, scores of positive class
      scores_negative: np.array, scores of negative class
    
    output
    ------
      eer: float, equal error rate
      threshold: float, the threshold for the err
      
    """
    return compute_eer(scores_positive, scores_negative)
    
    
def get_far_frr_dependent(bona_score_a, spoof_score_a, threshold_a, 
                          bona_score_b, spoof_score_b, threshold_b, 
                          NI, NC):
    """
    """
    far_ab_idx = np.bitwise_and(spoof_score_a < threshold_a, 
                                spoof_score_b >= threshold_b)
    far_ba_idx = np.bitwise_and(spoof_score_a >= threshold_a, 
                                spoof_score_b < threshold_b)
    frr_ab_idx = np.bitwise_and(bona_score_a >= threshold_a, 
                                bona_score_b < threshold_b)
    frr_ba_idx = np.bitwise_and(bona_score_a < threshold_a, 
                                bona_score_b >= threshold_b)

    far_ab = np.sum(far_ab_idx) / NI
    far_ba = np.sum(far_ba_idx) / NI
    frr_ab = np.sum(frr_ab_idx) / NC
    frr_ba = np.sum(frr_ba_idx) / NC
    return far_ab, far_ba, frr_ab, frr_ba


#######
# API for EER testing
# 
#######

def sig_test_holm_bonf_method(eer_bags, NC, NI, significance_level=0.05,
                              flag_reverse_indicator=False):
    """test_results = sig_test_holm_bonf_method(eer_bags, sig_level, NI, NC):

    input
    -----
      eer_bags: np.array, shape (N, M), where N is the number of systems, 
                and M is the number of random runs
                M can be 1, which means no multi-runs
      NC: int, number of bona fide trials in test set
      NI: int, number of spoofed trials in test set
      sig_level: float, significance level, default 0.05
      flag_reverse_indicator: bool, by default, no significance difference
                 is indicated by value 1.0 in output test_results
                 flag_reverse_indicator = True will set no-sig-diff
    
    output
    ------
      test_results: np.array, shape (N*M, N*M), 
                    test_results[i*M+j, l*M+n] shows the significance 
                    test between the j-th run of i-th system
                    and the n-th run of the l-th system
    
    Note: 
      test_results[i*M+j, l*M+n] == True: accept NULL hypothesis, 
      no significant different
    """
    # get the reject/accept
    #significance_level = 0.05

    num_system = eer_bags.shape[0]
    runs = eer_bags.shape[1]
    
    z_values = np.zeros([num_system * runs, num_system * runs])
    test_results = np.zeros(z_values.shape)

    for sys_1_idx in range(num_system):
        for sys_2_idx in range(num_system):
            # compute z_values 
            # significance test must be conducted within this pair of system
            z_value_tmp = np.zeros([runs, runs])
            for run_idx1 in range(runs):
                for run_idx2 in range(runs):
                    idx1 = sys_1_idx * runs + run_idx1
                    idx2 = sys_2_idx * runs + run_idx2
                    z_values[idx1, idx2] = compute_z_independent(
                        eer_bags[sys_1_idx, run_idx1], 
                        eer_bags[sys_1_idx, run_idx1], 
                        eer_bags[sys_2_idx, run_idx2], 
                        eer_bags[sys_2_idx, run_idx2], 
                        NI, NC)
                    z_value_tmp[run_idx1, run_idx2] = z_values[idx1, idx2]
    # save results
    if not flag_reverse_indicator:
        test_results = sig_test.reject_null_holm_bonferroni(
            z_values, z_values.size, significance_level)
    else:
        test_results = sig_test.reject_null_holm_bonferroni(
            z_values, z_values.size, significance_level, 
            accept_value = False, reject_value = True)
    return test_results


if __name__ == "__main__":
    print("Scripts eval_sig_test")
