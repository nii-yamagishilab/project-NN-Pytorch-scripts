##!/usr/bin/env python
"""
ResNet model
Modified based on https://github.com/joaomonteirof/e2e_antispoofing

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
from scipy import signal as scipy_signal

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init
import sandbox.block_nn as nii_nn

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

## 

class PreActBlock(torch_nn.Module):
    """ Pre-activation version of the BasicBlock
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBlock, self).__init__()
        # input batchnorm
        self.bn1 = torch_nn.BatchNorm2d(in_planes)
        # conv1
        self.conv1 = torch_nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False)
        
        self.bn2 = torch_nn.BatchNorm2d(planes)
        self.conv2 = torch_nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch_nn.Sequential(
                torch_nn.Conv2d(in_planes, self.expansion * planes, 
                                kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = torch_nn_func.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(torch_nn_func.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(torch_nn.Module):
    """ Pre-activation version of the original Bottleneck module.
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super(PreActBottleneck, self).__init__()
        # 
        self.bn1 = torch_nn.BatchNorm2d(in_planes)
        self.conv1 = torch_nn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False)
        
        self.bn2 = torch_nn.BatchNorm2d(planes)
        self.conv2 = torch_nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn3 = torch_nn.BatchNorm2d(planes)
        self.conv3 = torch_nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = torch_nn.Sequential(
                torch_nn.Conv2d(in_planes, self.expansion*planes, 
                          kernel_size=1, stride=stride, bias=False))
        

    def forward(self, x):
        out = torch_nn_func.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(torch_nn_func.relu(self.bn2(out)))
        out = self.conv3(torch_nn_func.relu(self.bn3(out)))
        out += shortcut
        return out

def conv3x3(in_planes, out_planes, stride=1):
    return torch_nn.Conv2d(in_planes, out_planes, 
                           kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return torch_nn.Conv2d(in_planes, out_planes, 
                           kernel_size=1, stride=stride, bias=False)

RESNET_CONFIGS = {'18': [[2, 2, 2, 2], PreActBlock],
                  '28': [[3, 4, 6, 3], PreActBlock],
                  '34': [[3, 4, 6, 3], PreActBlock],
                  '50': [[3, 4, 6, 3], PreActBottleneck],
                  '101': [[3, 4, 23, 3], PreActBottleneck]
                  }


class ResNet(torch_nn.Module):
    def __init__(self, enc_dim, resnet_type='18', nclasses=2):
        self.in_planes = 16
        super(ResNet, self).__init__()

        layers, block = RESNET_CONFIGS[resnet_type]

        self._norm_layer = torch_nn.BatchNorm2d

        # laye 1
        self.conv1 = torch_nn.Conv2d(1, 16, kernel_size=(9, 3), 
                                     stride=(3, 1), padding=(1, 1), bias=False)
        self.bn1 = torch_nn.BatchNorm2d(16)
        self.activation = torch_nn.ReLU()

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv5 = torch_nn.Conv2d(
            512 * block.expansion, 256, kernel_size=(3, 3), 
            stride=(1, 1), padding=(0, 1), bias=False)
        
        self.bn5 = torch_nn.BatchNorm2d(256)
        self.fc = torch_nn.Linear(256 * 2, enc_dim)
        
        if nclasses >= 2:
            self.fc_mu = torch_nn.Linear(enc_dim, nclasses) 
        else:
            self.fc_mu = torch_nn.Linear(enc_dim, 1)

        self.initialize_params()
        self.attention = nii_nn.SelfWeightedPooling(256)


    def initialize_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch_init.kaiming_normal_(layer.weight, a=0, mode='fan_out')
            elif isinstance(layer, torch.nn.Linear):
                torch_init.kaiming_uniform_(layer.weight)
            elif isinstance(layer, torch.nn.BatchNorm2d) or \
                 isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight.data.fill_(1)
                layer.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = torch_nn.Sequential(
                conv1x1(self.in_planes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
            
        layers = []
        layers.append(
            block(self.in_planes, planes, stride, downsample, 
                  1, 64, 1, norm_layer))
        
        self.in_planes = planes * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, 1, groups=1, 
                      base_width=64, dilation=False, norm_layer=norm_layer))

        return torch_nn.Sequential(*layers)

    def forward(self, x, without_pooling=False):

        x = self.conv1(x)
        x = self.activation(self.bn1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.activation(self.bn5(x)).squeeze(2)

        if without_pooling:
            return x
        else:
            stats = self.attention(x.permute(0, 2, 1).contiguous())

            feat = self.fc(stats)
            
            mu = self.fc_mu(feat)

            return feat, mu

if __name__ == "__main__":
    print("Definition of Resnet for anti-spoofing")
