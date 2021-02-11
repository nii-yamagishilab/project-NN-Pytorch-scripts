#!/usr/bin/env python
"""
additive margin softmax layers

Wang, F., Cheng, J., Liu, W. & Liu, H. 
Additive margin softmax for face verification. IEEE Signal Process. Lett. 2018

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
class AMAngleLayer(torch_nn.Module):
    """ Output layer to produce activation for Angular softmax layer
    AMAngleLayer(in_dim, output_dim, s=20, m=0.9):

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    s:          scaler
    m:          margin
    
    Method: (|x|cos, phi) = forward(x)
    
      x: (batchsize, input_dim)
    
      cos: (batchsize, output_dim)
      phi: (batchsize, output_dim)
    
    Note:
      cos[i, j]: cos(\theta) where \theta is the angle between
                 input feature vector x[i, :] and weight vector w[j, :]
      phi[i, j]: -1^k cos(m \theta) - 2k
    
    
    Usage example:  
      batchsize = 64
      input_dim = 10
      class_num = 2

      l_layer = AMAngleLayer(input_dim, class_num)
      l_loss = AMSoftmaxWithLoss()


      data = torch.rand(batchsize, input_dim, requires_grad=True)
      target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
      target = target.to(torch.long)

      scores = l_layer(data)
      loss = l_loss(scores, target)

      loss.backward()
    """
    def __init__(self, in_planes, out_planes, s=20, m=0.9):
        super(AMAngleLayer, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        
        self.weight = Parameter(torch.Tensor(in_planes, out_planes))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        
        self.m = m
        self.s = s

    def forward(self, input, flag_angle_only=False):
        """
        Compute am-softmax activations
        
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
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
                
        if flag_angle_only:
            cos_x = cos_theta
            phi_x = cos_theta
        else:
            cos_x = self.s * cos_theta
            phi_x = self.s * (cos_theta - self.m) 

        # ((batchsie, output_dim), (batchsie, output_dim))
        return cos_x, phi_x

    
class AMSoftmaxWithLoss(torch_nn.Module):
    """
    AMSoftmaxWithLoss()
    See usage in __doc__ of AMAngleLayer
    """
    def __init__(self):
        super(AMSoftmaxWithLoss, self).__init__()
        self.m_loss = torch_nn.CrossEntropyLoss()

    def forward(self, input, target):
        """ 
        input:
        ------
          input: tuple of tensors ((batchsie, out_dim), (batchsie, out_dim))
                 output from AMAngleLayer
        
          target: tensor (batchsize)
                 tensor of target index
        output:
        ------
          loss: scalar
        """
        # target (batchsize)
        target = target.long() #.view(-1, 1)
        
        
        # create an index matrix, i.e., one-hot vectors
        with torch.no_grad():
            index = torch.zeros_like(input[0])
            # index[i][target[i][j]] = 1
            index.scatter_(1, target.data.view(-1, 1), 1)
            index = index.bool()
        
        # use the one-hot vector as index to select
        # input[0] -> cos
        # input[1] -> phi
        # if target_i = j, ouput[i][j] = phi[i][j], otherwise cos[i][j]
        # 
        output = input[0] * 1.0
        output[index] -= input[0][index] * 1.0
        output[index] += input[1][index] * 1.0
        
        # cross entropy loss
        loss = self.m_loss(output, target)

        return loss

if __name__ == "__main__":
    print("Definition of Am-softmax loss")
