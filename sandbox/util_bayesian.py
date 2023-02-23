#!/usr/bin/env python
"""
util_bayesian.py

Utilities for bayeisan neural network
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

import torch
import torch.nn.functional as torch_nn_func

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"


######
# utils to save guide and model Pyro
# not used anymore
######
def save_model_guide(model, guide, path_model, path_guide): 
    #torch.save(dnn_net.state_dict(), "mnist_cnn_{:03d}.pt".format(idx))
    torch.save({"model" : model.state_dict(), 
                "guide" : guide}, path_model)
    pyro.get_param_store().save(path_guide)
    return
    
def load_model_guide(model, path_model, path_guide):
    pretrained = torch.load(path_model)
    model.load_state_dict(pretrained['model'])
    guide = pretrained['guide']
    pyro.get_param_store().load(path_guide)
    return guide


######
# Utils to compute metrics for Bayesian inference
######
def _xent(y, dim=-1, log_floor = 0.000001):
    """xe = xent(y, dim)
    input: y, tensor, (..., num_classes), probablity matrix
    input: dim, int, along which dimension we do xent? default -1
    output: xe, tensor, (..., 1), xe = -sum_j y[j] log y[j]
    """
    logfloor = torch.zeros_like(y)
    logfloor[y < log_floor] = log_floor
    return -torch.sum(y * torch.log(y + logfloor), dim=dim, keepdim=True)


def xent(p):
    """mi = xent(p)
    This measures total uncertainty
    input: p, tensor, (sammple_N, batch, num_classes), probablity
    output: xe, tensor, (batch, 1)
    """
    # step1. Bayesian model average p(y | x, D) = E_{q_w}[p(y | w, x)]
    #        ->  1/N sum_i p(y | w_i, x)
    # mp (batch, num_classes)
    mp = p.mean(dim=0)

    # step2. cross entropy over p(y | x, D)
    # xe (batch, 1)
    xe = _xent(mp)
    return xe
    
def compute_epstemic_uncertainty(y):
    """mi = mutual_infor(y)
    This measures epstemic uncertainty
    input: y, tensor, (sammple_N, batch, num_classes), probablity
    output: mi, tensor, (batch, 1)
    """
    # cross entropy over BMA prob, see xent() above
    xe = xent(y)
    # cross entropy over each individual sample, ve (sample_N, batch, 1)
    # for w_i, compute ent_i = xent(p(y | w_i, x))
    # then, ve = 1/N sum_i ent_i
    ve = torch.mean(_xent(y), dim=0)
    # xe - ve
    mi = xe - ve
    return mi


def compute_aleatoric_uncertainty(y):
    """mi = mutual_infor(y)
    This measures aleatoric uncertainty
    input: y, tensor, (sammple_N, batch, num_classes), probablity
    output: mi, tensor, (batch, 1)
    """
    ve = torch.mean(_xent(y), dim=0)
    return ve
    

def compute_logit_from_prob(y, log_floor=0.0000001):
    """logit = compute_logit_from_prob(y)
    input: y, tensor, any shape, probablity of being positive
    output: logit, tensor, same shape as y, sigmoid(logit) is y
    """
    logfloor = torch.zeros_like(y)
    logfloor[y < log_floor] = log_floor
    
    tmp = 1 / (y + logfloor) - 1
    logfloor = logfloor * 0
    logfloor[tmp < log_floor] = log_floor
    
    logit = - torch.log(tmp + logfloor)
    return logit
    

#####
# wrapper
#####

def compute_llr_eps_ale(logits, idx_pos=1):
    """llr, eps, ale = compute_llr_eps_ale(logits)
    
    input: logits, tensor (sampling_num, batch, 2)
           idx_pos, int, which dimension is the positive class?
           (default 1, which means logits[:, :, 1])
    output: llr, tensor, (batch, 1)
            eps, tensor, (batch, 1)
            ale, tensor, (batch, 1)
    """
    # -> (sampling_num, batch, 2)
    prob = torch_nn_func.softmax(logits, dim=-1)

    # to LLR
    # 1. average prob over the samples to (batch, num_class)
    # 2. compute the llr
    averaged_prob = torch.mean(prob, dim=0)
    # unsqueeze to make the shape consistent
    llr = compute_logit_from_prob(averaged_prob[..., idx_pos]).unsqueeze(-1)
    
    # get uncertainty
    eps = compute_epstemic_uncertainty(prob)
    ale = compute_aleatoric_uncertainty(prob)
    return llr, eps, ale



if __name__ == "__main__":
    print("Package for util_bayesian")

