##!/usr/bin/env python
"""
ResNet model
Modified based on https://github.com/joaomonteirof/e2e_antispoofing

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"


class ResNetBlock1D(torch_nn.Module):
    """ ResNetBlock1D(inplane, outplane, dilation=1, stride=1, 
       kernel=[1, 3, 1], expansion = 4)

    Args
    ----
       inplane:  int, input feature dimension.
       outplane: int, output feature dimension.
       dilation: int, convolution dilation
       stride:   int, stride size
       kernel:   [int, int, int], the kernel size of the 3 conv layers
       expansion: int, ratio for the bottleneck
    """
    def __init__(self, inplane, outplane, dilation=1, stride=1, 
                 kernel=[1, 3, 1], expansion = 4, act_type='ReLU'):
        super(ResNetBlock1D, self).__init__()
        
        # 
        self.ins = inplane
        self.outs = outplane
        self.expansion = expansion
        self.hid = self.ins // expansion
        
        dl = dilation
        #
        # block1 (batch, input_dim, length) -> (batch, hid_dim, length)
        pad = self._get_pad(1, dilation, kernel[0])
        self.conv1 = torch_nn.Sequential(
            torch_nn.Conv1d(self.ins, self.hid, kernel[0], 1, pad, dilation),
            torch_nn.BatchNorm1d(self.hid),
            self._get_act(act_type))
        
        # block2 (batch, hid_dim, length) -> (batch, hid_dim, length // stride)
        pad = self._get_pad(stride, dilation, kernel[1])
        self.conv2 = torch_nn.Sequential(
            torch_nn.Conv1d(self.hid, self.hid, kernel[1], stride, pad, dl),
            torch_nn.BatchNorm1d(self.hid),
            self._get_act(act_type))

        # block3
        pad = self._get_pad(1, dilation, kernel[2])
        self.conv3 = torch_nn.Sequential(
            torch_nn.Conv1d(self.hid, self.outs, kernel[2], 1, pad, dl),
            torch_nn.BatchNorm1d(self.outs))
        
        self.output_act = self._get_act(act_type)

        # change input dimension if necessary
        if self.ins != self.outs or stride != 1:
            pad = self._get_pad(stride, dilation, kernel[1])
            self.changeinput = torch_nn.Sequential(
                torch_nn.Conv1d(self.ins,self.outs, kernel[1], stride, pad, dl),
                torch_nn.BatchNorm1d(self.outs))
        else:
            self.changeinput = torch_nn.Identity()
        return
    
    def _get_act(self, act_type):
        if act_type == 'LeakyReLU':
            return torch_nn.LeakyReLU()
        elif act_type == 'ELU':
            return torch_nn.ELU()
        elif act_type == 'GELU':
            return torch_nn.GELU()
        else:
            return torch_nn.ReLU()
        
    def _get_pad(self, stride, dilation, kernel):
        pad = (dilation * (kernel - 1) + 1 - stride) // 2
        return pad
        
    def forward(self, input_data):
        """output = ResNetBlock(input_data)
        
        input: tensor, (batchsize, dimension, length)
        output: tensor, (batchsize, dimension, length)
        """
        output = self.conv1(input_data)
        output = self.conv2(output)
        output = self.conv3(output)
        output = output + self.changeinput(input_data)
        output = self.output_act(output)
        return output




class ResNet1D(torch_nn.Module):
    """
    """
    def __init__(self, inplane, outplanes, kernels, dilations, strides, ratios,
                 block_module = ResNetBlock1D, act_type = 'ReLU'):
        super(ResNet1D, self).__init__()
       
        # 
        tmp_ins = [inplane] + outplanes[:-1]
        tmp_outs = outplanes
        layer_list = []
        
        for indim, outdim, kernel, dilation, stride, expand in zip(
            tmp_ins, tmp_outs, kernels, dilations, strides, ratios):
            layer_list.append(
                block_module(indim, outdim, dilation, stride, kernel, 
                             expand, act_type))
        
        self.m_layers = torch_nn.Sequential(*layer_list)
        return
    
    def forward(self, input_data, length_first=True):
        """ output = ResNet(input_data, swap_dim=True)

        input
        -----
          input_data:  tensor, (batch, input_dim, length), 
                               or (batch, length, input_dim)
          length_first: bool, True, this is used when input_data is 
                              (batch, length, input_dim). Otherwise, False
        output
        ------
          output_data: tensor, (batch, length, input_dim) if length_first True
                               else (batch, input_dim, length) 
        """
        if length_first:
            return self.m_layers(input_data.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        else:
            return self.m_layers(input_data)


if __name__ == "__main__":
    print("Implementation of ResNet for 1D signals")
