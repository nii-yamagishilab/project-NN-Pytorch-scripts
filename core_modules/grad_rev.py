#!/usr/bin/env python
"""
grad_rev.py

Definition of gradient reverse layer

Copied from https://cyberagent.ai/blog/research/11863/
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

# 
class GradientReversalFunction(torch.autograd.Function):
    """ https://cyberagent.ai/blog/research/11863/
    """
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(scale)
        return x

    @staticmethod
    def backward(ctx, grad):
        scale, = ctx.saved_tensors
        return scale * -grad, None
    
class GradientReversal(torch_nn.Module):
    """ https://cyberagent.ai/blog/research/11863/
    """
    def __init__(self, scale: float):
        super(GradientReversal, self).__init__()
        self.scale = torch.tensor(scale)

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.scale)
    
