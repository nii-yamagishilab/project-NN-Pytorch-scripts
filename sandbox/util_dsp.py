#!/usr/bin/env python
"""
util_dsp.py

Utilities for signal processing

Code adapted from
https://github.com/fatchord/WaveRNN/blob/master/utils/distribution.py

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def label_2_float(x, bits):
    """Convert integer numbers to float values
    
    Note: dtype conversion is not handled

    Args:
    -----
       x: data to be converted Tensor.long or int, any shape. 
       bits: number of bits, int
    
    Return:
    -------
       tensor.float
    
    """
    return 2 * x / (2**bits - 1.) - 1.

def float_2_label(x, bits):
    """Convert float wavs back to integer (quantization)
    
    Note: dtype conversion is not handled

    Args:
    -----
       x: data to be converted Tensor.float, any shape. 
       bits: number of bits, int
    
    Return:
    -------
       tensor.float
    
    """
    #assert abs(x).max() <= 1.0
    peak = torch.abs(x).max()
    if peak > 1.0:
        x /= peak
    x = (x + 1.) * (2**bits - 1) / 2
    return torch.clamp(x, 0, 2**bits - 1)


def mulaw_encode(x, quantization_channels, scale_to_int=True):
    """Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding

    Args:
       x (Tensor): Input tensor, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law companded to int
         False: return mu-law in (-1, 1)
        
    Returns:
        Tensor: Input after mu-law encoding
    """
    # mu 
    mu = quantization_channels - 1.0
    
    # no check on the value of x
    if not x.is_floating_point():
        x = x.to(torch.float)
    mu = torch.tensor(mu, dtype=x.dtype, device=x.device)
    x_mu = torch.sign(x) * torch.log1p(mu * torch.abs(x)) / torch.log1p(mu)
    if scale_to_int:
        x_mu = ((x_mu + 1) / 2 * mu + 0.5).to(torch.int64)
    return x_mu

def mulaw_decode(x_mu, quantization_channels, input_int=True):
    """Adapted from torchaudio
    https://pytorch.org/audio/functional.html mu_law_encoding

    Args:
        x_mu (Tensor): Input tensor
        quantization_channels (int): Number of channels

    Returns:
        Tensor: Input after mu-law decoding (float-value waveform (-1, 1))
    """
    mu = quantization_channels - 1.0
    if not x_mu.is_floating_point():
        x_mu = x_mu.to(torch.float)
    mu = torch.tensor(mu, dtype=x_mu.dtype, device=x_mu.device)
    if input_int:
        x = ((x_mu) / mu) * 2 - 1.0
    else:
        x = x_mu
    x = torch.sign(x) * (torch.exp(torch.abs(x) * torch.log1p(mu)) - 1.0) / mu
    return x


if __name__ == "__main__":
    print("util_dsp.py")
