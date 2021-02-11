#!/usr/bin/env python
"""
P2sGrad: 
Zhang, X. et al. P2sgrad: Refined gradients for optimizing deep face models. 
in Proc. CVPR 9906-9914, 2019

I think the grad defined in Eq.(11) is equivalent to define a MSE loss with 0
or 1 as target:

\mathcal{L}_i = \sum_{j=1}^{K} (\cos\theta_{i,j} - \delta(j == y_i))^2 

The difference from a common MSE is that the network output is cos angle.
"""
from __future__ import print_function

import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_f
from torch.nn import Parameter

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

###################
class P2SActivationLayer(torch_nn.Module):
    """ Output layer that produces cos\theta between activation vector x
    and class vector w_j

    in_dim:     dimension of input feature vectors
    output_dim: dimension of output feature vectors 
                (i.e., number of classes)
    

    
    Usage example:
      batchsize = 64
      input_dim = 10
      class_num = 5

      l_layer = P2SActivationLayer(input_dim, class_num)
      l_loss = P2SGradLoss()

      data = torch.rand(batchsize, input_dim, requires_grad=True)
      target = (torch.rand(batchsize) * class_num).clamp(0, class_num-1)
      target = target.to(torch.long)

      scores = l_layer(data)
      loss = l_loss(scores, target)

      loss.backward()
    """
    def __init__(self, in_dim, out_dim):
        super(P2SActivationLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.weight = Parameter(torch.Tensor(in_dim, out_dim))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        return

    def forward(self, input_feat):
        """
        Compute P2sgrad activation
        
        input:
        ------
          input_feat: tensor (batchsize, input_dim)

        output:
        -------
          tensor (batchsize, output_dim)
          
        """
        # normalize the weight (again)
        # w (feature_dim, output_dim)
        w = self.weight.renorm(2, 1, 1e-5).mul(1e5)
        
        # normalize the input feature vector
        # x_modulus (batchsize)
        # sum input -> x_modules in shape (batchsize)
        x_modulus = input_feat.pow(2).sum(1).pow(0.5)
        # w_modules (output_dim)
        # w_moduls should be 1, since w has been normalized
        w_modulus = w.pow(2).sum(0).pow(0.5)

        # W * x = ||W|| * ||x|| * cos())))))))
        # inner_wx (batchsize, output_dim)
        inner_wx = input_feat.mm(w)
        # cos_theta (batchsize, output_dim)
        cos_theta = inner_wx / x_modulus.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)

        # done
        return cos_theta

    
class P2SGradLoss(torch_nn.Module):
    """P2SGradLoss() MSE loss between output and target one-hot vectors
    
    See usage in __doc__ of P2SActivationLayer
    """
    def __init__(self):
        super(P2SGradLoss, self).__init__()
        self.m_loss = torch_nn.MSELoss()

    def forward(self, input_score, target):
        """ 
        input
        -----
          input_score: tensor (batchsize, class_num)
                 cos\theta given by P2SActivationLayer(input_feat)
          target: tensor (batchsize)
                 target[i] is the target class index of the i-th sample

        output
        ------
          loss: scaler
        """
        # target (batchsize, 1)
        target = target.long() #.view(-1, 1)
        
        # filling in the target
        # index (batchsize, class_num)
        with torch.no_grad():
            index = torch.zeros_like(input_score)
            # index[i][target[i][j]] = 1
            index.scatter_(1, target.data.view(-1, 1), 1)
    
        # MSE between \cos\theta and one-hot vectors
        loss = self.m_loss(input_score, index)

        return loss

if __name__ == "__main__":
    print("Definition of P2SGrad Loss")
