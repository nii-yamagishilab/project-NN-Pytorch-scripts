#!/usr/bin/env python
"""
one class

One-class learning towards generalized voice spoofing detection
Zhang, You and Jiang, Fei and Duan, Zhiyao
arXiv preprint arXiv:2010.13995
"""
from __future__ import print_function

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f
from torch.nn import Parameter


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

###################
class OCAngleLayer(torch_nn.Module):
    """ Output layer to produce activation for 

    """
    def __init__(self, in_planes, w_posi=0.9, w_nega=0.2, alpha=20.0):
        super(OCAngleLayer, self).__init__()
        self.in_planes = in_planes
        self.w_posi = 0.9
        self.w_nega = 0.5
        self.out_planes = 1
        
        self.weight = Parameter(torch.Tensor(in_planes, self.out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        
        self.alpha = alpha

    def forward(self, input, flag_angle_only=False):
        """
        Compute oc-softmax activations
        
        input:
        ------
        input tensor (batchsize, input_dim)

        output:
        -------
        tuple of tensor ((batchsize, output_dim), (batchsize, output_dim))
        """
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        # w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, 1)
        inner_wx = input.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
                
        if flag_angle_only:
            pos_score = cos_theta
            neg_score = None
        else:
            pos_score = self.alpha * (self.w_posi - cos_theta)
            neg_score = -1 * self.alpha * (self.w_nega - cos_theta)
        
        #
        return pos_score, neg_score

    
class OCSoftmaxWithLoss(torch_nn.Module):
    """
    OCSoftmaxWithLoss()
    
    """
    def __init__(self):
        super(OCSoftmaxWithLoss, self).__init__()
        self.m_loss = torch_nn.Softplus()

    def forward(self, inputs, target):
        """ 
        """
        # assume target is binary, positive = 1, negaitve = 0
        # inputs[0] positive score, inputs[1] negative score
        output = inputs[0] * target + inputs[1] * (1-target)
        loss = self.m_loss(output)

        return loss

if __name__ == "__main__":
    print("Definition of Am-softmax loss")
