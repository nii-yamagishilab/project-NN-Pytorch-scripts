#!/usr/bin/env python
"""
a_softmax layers

copied from https://github.com/Joyako/SphereFace-pytorch

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
class AngleLayer(torch_nn.Module):
    """ Output layer to produce activation for Angular softmax layer
    AngleLayer(in_dim, output_dim, m=4):

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    m:          angular-softmax paramter

    
    Method: (|x|cos, phi) = forward(x)
    
    x: (batchsize, input_dim)
    
    cos: (batchsize, output_dim)
    phi: (batchsize, output_dim)
    
    Note:
    cos[i, j]: cos(\theta) where \theta is the angle between
               input feature vector x[i, :] and weight vector w[j, :]
    phi[i, j]: -1^k cos(m \theta) - 2k
    """
    def __init__(self, in_planes, out_planes, m=4):
        super(AngleLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        # cos(m \theta) = f(cos(\theta))
        self.cos_val = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x,
        ]

    def forward(self, input, flag_angle_only=False):
        """
        Compute a-softmax activations
        
        input:
        ------
        input tensor (batchsize, input_dim)
        flag_angle_only: true:  return cos(\theta), phi(\theta)
                         false: return |x|cos(\theta), |x|phi(\theta)
                         default: false
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
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1) \
                    / w_modulus.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)
        
        # cos(m \theta)
        cos_m_theta = self.cos_val[self.m](cos_theta)

        with torch.no_grad():
            # theta (batchsie, output_dim)
            theta = cos_theta.acos()
            # k is deterministic here
            # k * pi / m <= theta <= (k + 1) * pi / m
            k = (self.m * theta / 3.14159265).floor()
            minus_one = k * 0.0 - 1
        
        # phi_theta (batchsize, output_dim)
        # Phi(yi, i) = (-1)**k * cos(myi,i) - 2 * k
        phi_theta = (minus_one ** k) * cos_m_theta - 2 * k
        
        if flag_angle_only:
            cos_x = cos_theta
            phi_x = phi_theta
        else:
            cos_x = cos_theta * x_modulus.view(-1, 1)
            phi_x = phi_theta * x_modulus.view(-1, 1)

        # ((batchsie, output_dim), (batchsie, output_dim))
        return cos_x, phi_x

    
class AngularSoftmaxWithLoss(torch_nn.Module):
    """
    AngularSoftmaxWithLoss()
    This is a loss function. 

    Method:
    loss = forward(input, target)
    
    input: a pair of cos(\theta) and phi(\theta), 
           calculated by AngularLinear
           cos(\theta) and phi(\theta) shape: (batchsize, class_num)

    target: target labels (batchsize)
    
    """
    def __init__(self, gamma=0):
        super(AngularSoftmaxWithLoss, self).__init__()
        self.gamma = gamma
        self.iter = 0
        self.lambda_min = 5.0
        self.lambda_max = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        """ 
        """
        self.iter += 1
        # target (batchsize, 1)
        target = target.long().view(-1, 1)
        
        with torch.no_grad():
            index = torch.zeros_like(input[0])
            # index[i][target[i][j]] = 1
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = index.bool()
    
        # output (batchsize, output_dim)
        # Tricks
        # output(\theta_yi) 
        # = (lambda*cos(\theta_yi) + ((-1)**k * cos(m * \theta_yi) - 2*k))
        #    /(1 + lambda)
        # = cos(\theta_yi) 
        #   - cos(\theta_yi) / (1 + lambda) + Phi(\theta_yi) / (1 + lambda)
        self.lamb = max(self.lambda_min, 
                        self.lambda_max / (1 + 0.1 * self.iter))
        output = input[0] * 1.0
        output[index] -= input[0][index] * 1.0 / (1 + self.lamb)
        output[index] += input[1][index] * 1.0 / (1 + self.lamb)

        # softmax loss
        logit = torch_f.log_softmax(output, dim=1)
        # select the ones specified by target
        logit = logit.gather(1, target).view(-1)
        # additional
        pt = logit.data.exp()
        loss = -1 * (1 - pt) ** self.gamma * logit
        loss = loss.mean()

        return loss

if __name__ == "__main__":
    print("Definition of A-softmax loss")
