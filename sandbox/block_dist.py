##!/usr/bin/env python
"""
Module definition for distributions

Definition of distributions for generative models.
Each module should define two methods: forward and inference.
1. forward(input_feat, target): computes distribution given input_feat and
likelihood given target_data  
2. inference(input_feat): computes distribution given input_feat and draw sample

Note that Modules defined in core_modules/*.py are for discrminative models.
There is no method for inference. But they may be combined with this code
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


class DistCategorical(torch_nn.Module):
    """Output layer that implements categorical distribution
    
    This Module implements two methods: forward and inference.
    
      forward(input_feat, target): computes the categorical
      distribution given input_feat and likelihood given target_data
      
      inference(input_feat): computes the categorical
      distribution given input_feat and generate output
      
    Example:
        dim = 4
        prob_vec = torch.rand([2, 3, dim])
        prob_vec[0, 1, 0] += 9.9
        prob_vec[0, 2, 1] += 9.9
        prob_vec[0, 0, 2] += 9.9

        prob_vec[1, 1, 1] += 9.9
        prob_vec[1, 2, 2] += 9.9
        prob_vec[1, 0, 0] += 9.9

        target = torch.tensor([[[2], [0], [1]], [[0], [1], [2]]])

        l_cat = DistCategorical(dim)
        samples = l_cat.inference(prob_vec)
        print(prob_vec)
        print(samples)

        loss = l_cat.forward(prob_vec, target)
        print(loss)
    """
    def __init__(self, category_size):
        """ DistCategorical(category_size)

        Args
        ----
          category_size: int, number of category
        """
        super(DistCategorical, self).__init__()
        
        self.category_size = category_size
        self.loss = torch_nn.CrossEntropyLoss()

    def _check_input(self, input_feat):
        """ check whether input feature vector has the correct dimension
        torch.dist does not check, it will gives output no matter what
        the shape of input_feat 
        """
        if input_feat.shape[-1] != self.category_size:
            mes = "block_dist.DistCategorical expects input_feat with "
            mes += "last dimension of size {:d}. ".format(self.category_size)
            mes += "But receives {:d}".format(input_feat.shape[-1])
            raise Exception(mes)
        return True

    def forward(self, input_feat, target):
        """ likelihood = forward(input_feat, target)
        
        input
        -----
          input_feat: tensor (batchsize, length, categorize_size)
            tensor to be converted into categorical distribution
          target: (batchsize, length, dim=1)
            tensor to be used to evaluate the likelihood

        output
        ------
          likelihood: tensor scaler
        """
        self._check_input(input_feat)
        # transpose input_feat to (batchsize, cateogrical_size, length)
        # squeeze target to (batchsize, length)
        return self.loss(input_feat.transpose(1, 2), target.squeeze(-1))
        
    def inference(self, input_feat):
        """ sample = inference(input_feat)
        
        input
        -----
          input_feat: tensor (batchsize, length, categorize_size)
            tensor to be converted into categorical distribution

        output
        ------
          sample: (batchsize, length, dim=1)
        """
        # check
        self._check_input(input_feat)

        # compute probability
        prob_vec = torch_nn_func.softmax(input_feat, dim=2)

        # distribution
        distrib = torch.distributions.Categorical(prob_vec)
        
        # draw samples and save
        sample = torch.zeros(
            [input_feat.shape[0], input_feat.shape[1], 1],
            dtype=input_feat.dtype, device=input_feat.device)
        sample[:, :, 0] = distrib.sample()
        
        return sample



if __name__ == "__main__":
    print("Definition of distributions modules")
