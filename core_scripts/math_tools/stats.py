#!/usr/bin/env python
"""
stats.py

Tools to calcualte statistics

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import core_scripts.other_tools.display as nii_display
import core_scripts.data_io.conf as nii_dconf

import torch

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def f_var2std(var):
    """
    std = f_var2std(var)
    Args:
     var: np.arrary, variance
    
    Return:
     std: np.array, standard-devitation

    std = sqrt(variance), std[std<floor] = 1.0
    """
    negative_idx = var < 0
    std = np.sqrt(var)
    std[negative_idx] = 1.0
    floored_idx = std < nii_dconf.std_floor
    std[floored_idx] = 1.0
    return std
    

def f_online_mean_std(data, mean_old, var_old, cnt_old):
    """ 
    mean, var, count=f_online_mean_var(data, mean, var, num_count):
    
    online algorithm to accumulate mean and var
    
    input
    -----
      data: input data as numpy.array, in shape [length, dimension]
      mean: mean to be updated, np.array [dimension]
      var: var to be updated, np.array [dimension]
      num_count: how many data rows have been calculated before 
        this calling.
    
    output
    ------
      mean: mean, np.array [dimension]
      var: var, np.array [dimension]
      count: accumulated data number, = num_count + data.shape[0]

    Ref. parallel algorithm                                                 
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance  
    """
    try:
        # how many time steps (number of rows) in this data
        cnt_this = data.shape[0]

        # if input data is empty, don't update
        if cnt_this == 0:
            return mean_old, var_old, cnt_old
        
        if data.ndim == 1:
            # single dimension data, 1d array
            mean_this = data.mean()
            var_this = data.var()
            dim = 1
        else:
            # multiple dimension data, 2d array
            mean_this = data.mean(axis=0)
            var_this = data.var(axis=0)
            dim = data.shape[1]
            
        # difference of accumulated mean and data mean
        diff_mean = mean_this - mean_old

        # new mean and var
        new_mean = np.zeros([dim], dtype=nii_dconf.h_dtype)
        new_var = np.zeros([dim], dtype=nii_dconf.h_dtype)

        # update count
        updated_count = cnt_old + cnt_this
        
        # update mean
        new_mean = mean_old + diff_mean * (float(cnt_this) /
                                           (cnt_old + cnt_this))
        # update var
        if cnt_old == 0:
            # if this is the first data
            if data.ndim == 1:
                # remember that var is array, not scalar
                new_var[0] = var_this
            else:
                new_var = var_this
        else:
            # not first data
            new_var = (var_old * (float(cnt_old) / updated_count) 
                       + var_this * (float(cnt_this)/ updated_count) 
                       + (diff_mean * diff_mean
                          / (float(cnt_this)/cnt_old 
                             + float(cnt_old)/cnt_this
                             + 2.0)))
        # done
        return new_mean, new_var, updated_count
        
    except ValueError:
        if data.ndim > 1:
            if data.shape[1] != mean_old.shape[0] or \
               data.shape[1] != var_old.shape[0]:
                nii_display.f_print("Dimension incompatible", "error")
                nii_display.f_die("Error in online mean var calculation")
        else:
            if mean_old.shape[0] != 1 or \
               var_old.shape[0] != 1:
                nii_display.f_print("Dimension incompatible", "error")
                nii_display.f_die("Error in online mean var calculation")
            


def f_online_mean_cov(data, mean_old, cov_old, cnt_old):
    """ 
    mean, cov, count=f_online_mean_cov(data, mean, cov, num_count):
    
    online algorithm to accumulate mean and cov
    
    input
    -----
      data: input data as numpy.array, in shape [length, dimension]
      mean: mean to be updated, np.array [dimension]
      cov: cov to be updated, np.array [dimension, dimension]
      num_count: how many data rows have been calculated before 
        this calling.
    
    output
    ------
      mean: mean, np.array [dimension]
      cov: cov, np.array [dimension, dimension]
      count: accumulated data number, = num_count + data.shape[0]

    Note that the returned cov is biased. 

    Ref. parallel algorithm                                                 
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance  
    """
    if data.ndim == 1:
        nii_display.f_print("Dimension incompatible", "error")
        nii_display.f_die("Error in online mean cov calculation")
    
    try:
        # how many time steps (number of rows) in this data
        cnt_this = data.shape[0]
        dim = data.shape[1]
        
        # if input data is empty, don't update
        if cnt_this == 0:
            return mean_old, cov_old, cnt_old
        
        # multiple dimension data, 2d array
        mean_this = data.mean(axis=0)
        
        # assumpe number of columns be the number of variables
        if cnt_this == 1:
            cov_this = np.zeros([dim, dim], dtype=nii_dconf.h_dtype)
        else:
            cov_this = np.cov(data.T)
                
        # difference of accumulated mean and data mean
        diff_mean = mean_this - mean_old

        # new mean and cov
        new_mean = np.zeros([dim], dtype=nii_dconf.h_dtype)
        new_cov = np.zeros([dim, dim], dtype=nii_dconf.h_dtype)

        # update count
        updated_count = cnt_old + cnt_this
        
        # update mean
        new_mean = mean_old + diff_mean * (float(cnt_this) /
                                           (cnt_old + cnt_this))
        # update cov
        if cnt_old == 0:
            new_cov = cov_this
        else:
            # not first data
            new_cov = (cov_old * (float(cnt_old) / updated_count) 
                       + cov_this * (float(cnt_this)/ updated_count) 
                       + (np.outer(diff_mean, diff_mean)
                          / (float(cnt_this)/cnt_old 
                             + float(cnt_old)/cnt_this
                             + 2.0)))
        # done
        return new_mean, new_cov, updated_count
        
    except ValueError:
        if data.ndim > 1:
            if data.shape[1] != mean_old.shape[0] or \
               data.shape[1] != cov_old.shape[0]:
                nii_display.f_print("Dimension incompatible", "error")
                nii_display.f_die("Error in online mean cov calculation")
        else:
            if mean_old.shape[0] != 1 or \
               cov_old.shape[0] != 1:
                nii_display.f_print("Dimension incompatible", "error")
                nii_display.f_die("Error in online mean cov calculation")
        


def cov_tensor(X):
    """cov = cov_tensor(X)
    compute covariance of tensor
    
    input
    -----
      X: tensor, (batch, length, dim)
      
    output
    ------
      cov: tensor, (batch, dim, dim)
      
    Note that this function returns biased cov
    """
    X = data
    D = X.shape[1]
    mean = torch.mean(X, dim=1).unsqueeze(1)
    X = X - mean
    return 1/D * torch.matmul(X.transpose(-1, -2), X)


def f_online_mean_cov_tensor(data, mean_old, cov_old, cnt_old):
    """ 
    mean, cov, count=f_online_mean_cov(data, mean, cov, num_count):
    
    online algorithm to accumulate mean and cov
    
    input
    -----
      data: tensor, in shape [batch, length, dimension]
      mean: mean to be updated, tensor [batch, dimension]
      cov: cov to be updated, tensor [batch, dimension, dimension]
      num_count: how many data rows have been calculated before 
        this calling.
    
    output
    ------
      mean: mean, tensor [batch, dimension]
      cov: cov, tensor [batch, dimension, dimension]
      count: accumulated data number, = num_count + data.shape[0]

    Note that the returned cov is biased. 

    Example:
      data = torch.randn([2, 10, 5])

      mean_start = torch.zeros([2, 5])
      cov_start = torch.zeros([2, 5, 5])
      cnt = 0

      mean_old = mean_start
      cov_old = cov_start

      for idx in range(10):
        new_mean, new_var, cnt = f_online_mean_cov_tensor(data[:, idx:idx+1, :],
                  mean_old, cov_old, cnt)
        mean_old = new_mean
        cov_old = new_var
    
    Ref. parallel algorithm                                                 
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    # how many time steps (number of rows) in this data
    batch, cnt_this, dim = data.shape

    # if input data is empty, don't update
    if cnt_this == 0:
        return mean_old, cov_old, cnt_old

    # Mean and Cov for this data
    # multiple dimension data, 2d array
    mean_this = data.mean(dim=1)
    # assumpe number of columns be the number of variables
    if cnt_this == 1:
        cov_this = torch.zeros(
            [dim, dim], dtype=data.dtype, device=data.device)
    else:
        cov_this = cov(data.T)

    # Update 
    # difference of accumulated mean and data mean
    diff_mean = mean_this - mean_old

    # new mean and cov
    new_mean = torch.zeros([dim], dtype=data.dtype, device=data.device)
    new_cov = torch.zeros([dim, dim], dtype=data.dtype, device=data.device)

    # update count
    updated_count = cnt_old + cnt_this

    # update mean
    new_mean = mean_old + diff_mean * (float(cnt_this) /
                                       (cnt_old + cnt_this))
    # update cov
    if cnt_old == 0:
        new_cov = cov_this
    else:
        # not first data
        new_cov = (cov_old * (float(cnt_old) / updated_count) 
                   + cov_this * (float(cnt_this)/ updated_count) 
                   + (torch.bmm(diff_mean.unsqueeze(-1), diff_mean.unsqueeze(1))
                      / (float(cnt_this)/cnt_old 
                         + float(cnt_old)/cnt_this
                         + 2.0)))
    # done
    return new_mean, new_cov, updated_count



if __name__ == "__main__":
    pass
