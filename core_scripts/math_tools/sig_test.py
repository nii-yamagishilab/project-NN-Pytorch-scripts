#!/usr/bin/env python
"""
sig_test.py

Tools for significance test.

This is used in tutorials/b2_anti_spoofing/chapter_a1_stats_test.ipynb
"""
from __future__ import absolute_import

import os
import sys
import numpy as np
from scipy import stats

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

##########
# Functions
##########

def compute_Z_alpha2(significance_level, alternative='two-sided'):
    """ z_alpha = compute_Z_alpha2(significance_level, alternative)
    
    input
    -----
      significance_level: float
      alternative:        string, less, greatr or two-sided
    
    output
    ------
      Z_alpha2:           float
    
    Example
    -------
      print("Z_alpha/2 for 90% CI (alpha=0.10): {:.4f}".format(
          compute_Z_alpha2(0.10)))
      print("Z_alpha/2 for 95% CI (alpha=0.05): {:.4f}".format(
          compute_Z_alpha2(0.05)))
      print("Z_alpha/2 for 99% CI (alpha=0.01): {:.4f}".format(
          compute_Z_alpha2(0.01)))
    """
    if alternative == 'less':
        Z_alpha2 = stats.norm.ppf(significance_level)
    elif alternative == 'greater':
        Z_alpha2 = stats.norm.isf(significance_level)
    elif alternative == 'two-sided':
        Z_alpha2 = stats.norm.isf(significance_level/2)
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")
    return Z_alpha2



##########
# Tools for  family-wise hypothesis testing and correction
##########

def reject_null_bonferroni_naive(
        z_values, num_test, significance_level, alternative='two-sided', 
        accept_value=True, reject_value=False):
    """result = reject_null_bonferroni_naive(z_values, significance_level)
    
    native bonferroni correction
    https://en.wikipedia.org/wiki/Bonferroni_correction
    
    input
    -----
      z_values: np.array, an array of z_value
      num_test: int, number of tests in the experiment
      signifiance_level: float, common choise is 0.1, 0.05, or 0.01

    output
    ------
      result: np.array, same size as z_values, 
         if result[i] is True if z_value[i] is larger than the threshold
    """
    #num_test = z_values.size
    corrected_conf_level = significance_level / num_test
    Z_alpha2 = compute_Z_alpha2(corrected_conf_level, alternative)

    # reject null hypothesis
    idx_reject = z_values > Z_alpha2
    # acccept null hypothesis
    idx_accept = z_values <= Z_alpha2
    
    result = np.zeros(z_values.shape)
    result[idx_accept] = accept_value
    result[idx_reject] = reject_value
    return result


def reject_null_sidak(
        z_values, num_test, significance_level, alternative='two-sided', 
        accept_value=True, reject_value=False):
    """ similar API to reject_null_bonferroni_naive
    
    See Hervé Abdi, and others. Bonferroni and Šidák 
    Corrections for Multiple Comparisons. 
    Encyclopedia of Measurement and Statistics 3. 
    Sage Thousand Oaks, CA: 103–107. 2007.
    """
    #num_test = z_values.size
    #FWER = 1 - (1 - significance_level) ** (1 / num_test)
    corrected_conf_level = 1 - (1 - significance_level) ** (1 / num_test)
    Z_alpha2 = compute_Z_alpha2(corrected_conf_level, alternative)
    
    idx_reject = z_values >= Z_alpha2
    idx_accept = z_values < Z_alpha2
    result = np.zeros(z_values.shape)
    result[idx_accept] = accept_value
    result[idx_reject] = reject_value
    return result


def reject_null_holm_bonferroni(
        z_values, num_test, significance_level, alternative='two-sided', 
        accept_value=True, reject_value=False):
    """ similar API to reject_null_bonferroni_naive
    
    See https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    """
    # get the index of each z_value in the sorted array
    # large z corresponds to small p
    # we start with the case where the z is the largest
    order = np.argsort(z_values.flatten())[::-1]
    
    # a result buffer, by default, set all to accept
    result_buff = np.zeros(order.shape) + accept_value
    
    # conduct the test 
    for test_idx, data_idx in enumerate(order):
        # the test coming first should receive a more strict Z_alpha
        corrected_conf_level = significance_level / (num_test - test_idx)
        Z_alpha2 = compute_Z_alpha2(corrected_conf_level, alternative)
        #print(corrected_conf_level)
        if z_values.flatten()[data_idx] < Z_alpha2:
            # if we cannot reject, stop testing
            result_buff[data_idx] = accept_value
            break
        else:
            result_buff[data_idx] = reject_value
    
    return np.reshape(result_buff, z_values.shape)


def reject_null_holm_bonferroni_given_p_value(
        p_values, num_test, significance_level, 
        accept_value=True, reject_value=False):
    """ result = reject_null_holm_bonferroni_given_p_value(
          p_values, num_test, significance_level, 
          accept_value = True, reject_value = False)
    https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    
    input
    -----
      p_values:           np.array, any shape, of p-values for multiple test
      num_test:           int,      number of elements in the p_values matrix
      significance_level: float
      accept_value:       int, float, or bool, anything. Default True
                          It is used to indicate 
                          "accept the null hypothesis"
      reject_value:       int, float, or bool, anything. Default False
                          It is used to indicate 
                          "reject the null hypothesis"
    output
    ------
      result:             np.array, same shape as p_values
                          If result[i, ...] accepts the null hypothesis, 
                          its value will be set to accept_value
                          Otherwise, the value will be equal to reject_value

    See Example in 

    >> p_values = np.array([[0.01, 0.04], [0.03, 0.005]])
    >> print(reject_null_holm_bonferroni_given_p_value(p_values, 4, 0.05))
    Compare p-value 0.005 with corrected alpha 0.0125 reject NULL
    Compare p-value 0.010 with corrected alpha 0.0167 reject NULL
    Compare p-value 0.030 with corrected alpha 0.0250 accept NULL, stop
    [[0. 1.]
     [1. 0.]]
    """
    # get the index of each p_value in the sorted array
    # small p comes first
    order = np.argsort(p_values.flatten())
    
    # a result buffer, by default, set all to accept
    result_buff = np.zeros(order.shape) + accept_value
    
    # conduct the test 
    for test_idx, data_idx in enumerate(order):
        # the test coming first should receive a more strict Z_alpha
        corrected_conf_level = significance_level / (num_test - test_idx)
        
        print("Compare p-value {:.3f} with corrected alpha {:.4f}".format(
            p_values.flatten()[data_idx], corrected_conf_level), end=' ')
        # here smaller p_value is more significant
        if p_values.flatten()[data_idx] > corrected_conf_level:
            # if we cannot reject, stop testing
            result_buff[data_idx] = accept_value
            print("accept NULL, stop")
            break
        else:
            result_buff[data_idx] = reject_value
            print("reject NULL")
    
    return np.reshape(result_buff, p_values.shape)


if __name__ == "__main__":
    print("tools for significance test")
