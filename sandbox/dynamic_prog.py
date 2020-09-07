#!/usr/bin/env python
"""
nn_manager_gan

A simple wrapper to run the training / testing process for GAN

"""
from __future__ import print_function
import os
import sys
import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

import core_scripts.other_tools.debug as nii_debug

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#############################################################
def viterbi_decode(init_prob, trans_prob, obser_prob, 
                   eps=torch.finfo(torch.float32).eps, return_more=False):
    if type(init_prob) is torch.Tensor:
        _log_func = torch.log
        _torch_flag = True
    else:
        _log_func = np.log
        _torch_flag = False
    log_init_prob = _log_func(init_prob + eps)
    log_trans_prob = _log_func(trans_prob + eps) 
    log_obser_prob = _log_func(obser_prob + eps) 
    
    n_time, n_state = log_obser_prob.shape
    if log_trans_prob.shape[0] != n_state or log_trans_prob.shape[0] != n_state:
        print("Viterbi decoding: transition prob matrix invalid")
        sys.exit(1)
    if log_init_prob.shape[0] != n_state:
        print("Viterbi decoding: init prob matrix invalid")
        sys.exit(1)
    
    if _torch_flag:
        prob_mat = torch.zeros_like(log_obser_prob)
        state_mat = torch.zeros_like(log_obser_prob, dtype=torch.int)
        best_state = torch.zeros([n_time], dtype=torch.int, 
                                 device = init_prob.device)
        _argmax = torch.argmax
        tmp_idx = torch.arange(0, n_state, dtype=torch.long)
    else:
        prob_mat = np.zeros(log_obser_prob.shape)
        state_mat = np.zeros(log_obser_prob.shape, dtype=np.int)
        best_state = np.zeros([n_time], dtype=np.int)
        _argmax = np.argmax
        tmp_idx = np.arange(0, n_state, dtype=np.int)
    

    prob_mat[0, :] = log_init_prob + log_obser_prob[0, :]
    for time_idx in np.arange(1, n_time):
        trout_prob = prob_mat[time_idx - 1] + log_trans_prob.T
        #print(time_idx)
        tmp_best = _argmax(trout_prob, axis=1)
        state_mat[time_idx] = tmp_best
        prob_mat[time_idx] = trout_prob[tmp_idx, tmp_best] \
                             + log_obser_prob[time_idx]
        #for state_idx in np.arange(n_state):
        #    tmp_best = _argmax(trout_prob[state_idx])
        #    state_mat[time_idx, state_idx] = tmp_best
        #    prob_mat[time_idx, state_idx] = trout_prob[state_idx, tmp_best] \
        #                                    +log_obser_prob[time_idx, state_idx]
    
    best_state[-1] = _argmax(prob_mat[-1, :])
    for time_idx in np.arange(n_time-2, -1, -1):
        best_state[time_idx] = state_mat[time_idx+1, best_state[time_idx+1]]
    if return_more:
        return best_state, prob_mat, state_mat
    else:
        return best_state
