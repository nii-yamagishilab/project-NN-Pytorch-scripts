#!/usr/bin/env python
"""
Functions for dynamic programming

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
    """ Routine to do Viterbi decoding
    
    viterbi_decode(init_prob, trans_prob, obser_prob, 
                   eps=torch.finfo(torch.float32).eps, return_more=False):
    
    Input:
       init_prob: initialia state probability
                  tensor or np.arrary, in shape (N), for N states
       trans_prob: transition probability
                  tensor or np.array, in shape (N, N)
                  trans_prob(i, j): P(state=j | prev_state=i)
       obser_prob: observation probability
                  tensor or np.array, in shape (T, N), for T time sptes
        
       return_more: True: return best_states, prob_mat, state_trace
                    False: return best_states
    Output:
       best_states: best state sequence tensor or np.array, in shape (T)
       prob_mat: probablity matrix in shape (T, N), where (t, j) denotes
                 max_{s_1:t-1} P(o_1:t, s_1:t-1, s_t=j)
       state_mat: in shape (T, N), where (t, j) denotes
                 argmax_i P(o_1:t, s_1:t-2, s_t-1=i, s_t=j)
    """ 
    
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
        best_states = torch.zeros([n_time], dtype=torch.int, 
                                 device = init_prob.device)
        _argmax = torch.argmax
        tmp_idx = torch.arange(0, n_state, dtype=torch.long)
    else:
        prob_mat = np.zeros(log_obser_prob.shape)
        state_mat = np.zeros(log_obser_prob.shape, dtype=np.int)
        best_states = np.zeros([n_time], dtype=np.int)
        _argmax = np.argmax
        tmp_idx = np.arange(0, n_state, dtype=np.int)
    

    prob_mat[0, :] = log_init_prob + log_obser_prob[0, :]
    for time_idx in np.arange(1, n_time):
        trout_prob = prob_mat[time_idx - 1] + log_trans_prob.T

        # this version is faster?
        #print(time_idx)
        tmp_best = _argmax(trout_prob, axis=1)
        state_mat[time_idx] = tmp_best
        prob_mat[time_idx] = trout_prob[tmp_idx, tmp_best] \
                             + log_obser_prob[time_idx]
        
        # seems to be too slow
        #for state_idx in np.arange(n_state):
        #    tmp_best = _argmax(trout_prob[state_idx])
        #    state_mat[time_idx, state_idx] = tmp_best
        #    prob_mat[time_idx, state_idx] = trout_prob[state_idx, tmp_best] \
        #                                   +log_obser_prob[time_idx, state_idx]
    
    best_states[-1] = _argmax(prob_mat[-1, :])
    for time_idx in np.arange(n_time-2, -1, -1):
        best_states[time_idx] = state_mat[time_idx+1, best_states[time_idx+1]]
    if return_more:
        return best_states, prob_mat, state_mat
    else:
        return best_states
