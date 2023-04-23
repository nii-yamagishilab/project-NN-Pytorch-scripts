#!/usr/bin/env python
"""
This file contains code for RawNet2

Hemlata Tak, Jose Patino, Massimiliano Todisco, Andreas Nautsch, 
Nicholas Evans, and Anthony Larcher. End-to-End Anti-Spoofing with RawNet2. 
In Proc. ICASSP, 6369--6373. 2020.

Implementation based on RawNet in
https://github.com/asvspoof-challenge/2021/

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import core_scripts.other_tools.debug as nii_debug

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

class SincConv2(torch_nn.Module):
    """
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, num_filters, kernel_size, in_channels=1,
                 sample_rate = 16000, num_freq_bin = 257,
                 stride = 1, dilation = 1, 
                 flag_pad = True, flag_trainable=False):
        """
        SincConv2(num_filters, kernel_size, in_channels=1,
                 sample_rate = 16000, num_freq_bins = 257,
                 stride = 1, dilation = 1, 
                 flag_pad = True, flag_trainable=False)
        Args
        ----
          num_filters: int, number of sinc-filters
          kernel_size: int, length of each sinc-filter
          in_channels: int, dimension of input signal, 
                       (batchsize, length, in_channels)
          sample_rate: int, sampling rate
          num_freq_bin: number of frequency bins, not really important
                        here. Default 257
          stride:      int, stride of convoluiton, default 1
          dilation:    int, dilaion of conv, default 1
          flag_pad:    bool, whether pad the sequence to make input and 
                       output have equal length, default True
          flag_trainable: bool, whether the filter is trainable
                       default False
            
        
        Num_filters and in_channels decide the output tensor dimension
        If input is (batchsize, length, in_channels), output will be
        (batchsize, length, in_channels * num_filters)
        
        This is done through depwise convolution, 
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        i.e., each input dimension will go through all the num_filters.
        """
        super(SincConv2,self).__init__()
        
        self.m_out_channels = num_filters
        self.m_in_channels = in_channels
        self.m_sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        self.m_kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.m_kernel_size = self.m_kernel_size + 1

        self.m_stride = stride
        self.m_dilation = dilation
        
        # Pad to original length
        if flag_pad:
            self.m_padding = dilation * (self.m_kernel_size - 1) + 1 - stride
            if stride % 2 == 0:
                print("Warning: padding in SincCov is not perfect because of")
                print("stride {:d}".format(stride))
            self.m_padding = self.m_padding // 2
        else:
            self.m_padding = 0
        
        
        
        # initialize filterbanks using Mel scale
        f = int(self.m_sample_rate / 2) * np.linspace(0, 1, num_freq_bin)
        # Hz to mel conversion
        fmel = self.to_mel(f)   
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.m_out_channels+1)
         # Mel to Hz conversion
        filbandwidthsf = self.to_hz(filbandwidthsmel) 
        
        # mel band
        self.m_mel = filbandwidthsf
        # time index
        self.m_hsupp = torch.arange(-(self.m_kernel_size-1)/2, 
                                    (self.m_kernel_size-1)/2+1)
        # filter coeffs
        self.m_filters = torch.zeros(self.m_out_channels, self.m_kernel_size)
        
        # create filter coefficient
        for i in range(self.m_out_channels):
            fmin = self.m_mel[i]
            fmax = self.m_mel[i+1]
            hHigh = np.sinc(2 * fmax * self.m_hsupp / self.m_sample_rate)
            hHigh = (2 * fmax / self.m_sample_rate) * hHigh
            hLow = np.sinc(2 * fmin * self.m_hsupp / self.m_sample_rate)
            hLow = (2 * fmin / self.m_sample_rate) * hLow
            # band pass filters
            hideal = hHigh - hLow
            
            # applying windowing
            self.m_filters[i,:] = torch.tensor(
                np.hamming(self.m_kernel_size) * hideal)
        
        # repeat to (output_channels * in_channels)
        self.m_filters = self.m_filters.repeat(self.m_in_channels, 1)
        
        # save as model parameter
        self.m_filters = self.m_filters.view(
            self.m_out_channels * self.m_in_channels, 1, self.m_kernel_size)
        self.m_filters = torch_nn.Parameter(
            self.m_filters, requires_grad=flag_trainable)

        return
    
    def forward(self,x):
        """SincConv(x)
        
        input
        -----
          x: tensor, shape (batchsize, length, feat_dim)
          
        output
        ------
          y: tensor, shape (batchsize, length, output-channel)
        """
        return torch_nn_func.conv1d(
            x.permute(0, 2, 1), self.m_filters, stride=self.m_stride,
                        padding=self.m_padding, dilation=self.m_dilation,
                        bias=None, groups=x.shape[-1]).permute(0, 2, 1)



class FMS(torch_nn.Module):
    """filter-wise feature map scaling
    Hemlata Tak, Jose Patino, Massimiliano Todisco, Andreas Nautsch, 
    Nicholas Evans, and Anthony Larcher. 
    End-to-End Anti-Spoofing with RawNet2. 
    In Proc. ICASSP, 6369--6373. 2020.
    
    Example:
    l_fms = FMS(5)
    with torch.no_grad():
        data = torch.randn(2, 1000, 5)
        out = l_fms(data)
    """
    def __init__(self, feat_dim):
        """FMS(feat_dim)
        
        Args
        ----
          feat_dim: int, dimension of input, in shape (batch, length, dim)
        """
        super(FMS, self).__init__()
        self.m_dim = feat_dim
        self.m_pooling = torch_nn.AdaptiveAvgPool1d(1)
        self.m_dim_change = torch_nn.Linear(feat_dim, feat_dim)
        self.m_act = torch_nn.Sigmoid()
        return
    
    def forward(self, x):
        """FMS(x)
        input
        -----
          x: tensor, (batch, length, dim)
          
        output
        -----
          y: tensor, (batch, length, dim)
        """
        if x.shape[-1] != self.m_dim:
            print("FMS expects data of dim {:d}".format(self.m_dim))
            sys.exit(1)
            
        # pooling expects (batch, dim, length)
        # y will be (batch, dim, 1)
        y = self.m_pooling(x.permute(0, 2, 1))
        
        # squeeze to (batch, dim), unsqueeze to (batch, 1, dim, )
        y = self.m_act(self.m_dim_change(y.squeeze(-1))).unsqueeze(1)
        
        # scaling and shifting
        return (x * y + y)
    

class Residual_block(torch_nn.Module):
    """Residual block used in RawNet2 for Anti-spoofing
    """
    def __init__(self, nb_filts, flag_bn_input = False):
        """Residual_block(bn_filts, flga_bn_input)
        Args
        ----
          bn_filts: list of int, [input_channel, output_channel]
          flag_bn_input: bool, whether do BatchNorm and LReLU
                        default False
        """
        super(Residual_block, self).__init__()
        
        # whether batch normalize input
        if flag_bn_input:
            self.bn1 = torch_nn.Sequential(
                torch_nn.BatchNorm1d(num_features = nb_filts[0]),
                torch_nn.LeakyReLU(negative_slope=0.3))
        else:
            self.bn1 = None
        
        self.conv = torch_nn.Sequential(
            torch_nn.Conv1d(in_channels = nb_filts[0],
                            out_channels = nb_filts[1],
                            kernel_size = 3,
                            padding = 1,
                            stride = 1),
            torch_nn.BatchNorm1d(num_features = nb_filts[1]),
            torch_nn.Conv1d(in_channels = nb_filts[1],
                            out_channels = nb_filts[1],
                            padding = 1,
                            kernel_size = 3,
                            stride = 1)
        )
        
        # for dimension change
        if nb_filts[0] != nb_filts[1]:
            self.dim_change = torch_nn.Conv1d(
                in_channels = nb_filts[0],
                out_channels = nb_filts[1],
                padding = 0,
                kernel_size = 1,
                stride = 1)
        else:
            self.dim_change = None
            
        # maxpooling
        self.mp = torch_nn.MaxPool1d(3)
        
        return
    
    def forward(self, x):
        """ y= Residual_block(x)
        
        input
        -----
          x: tensor, (batchsize, length, dim)
          
        output
        ------
          y: tensor, (batchsize, length, dim2)
        """
        identity = x.permute(0, 2, 1)
        
        if self.bn1 is None:
            out = x.permute(0, 2, 1) 
        else:
            out = self.bn1(x.permute(0, 2, 1))

        out = self.conv(out)
        
        if self.dim_change is not None:
            identity = self.dim_change(identity)
            
        out += identity
        out = self.mp(out)
        return out.permute(0, 2, 1)
    
class RawNet(torch_nn.Module):
    """RawNet based on 
    https://github.com/asvspoof-challenge/2021/
    """
    def __init__(self, num_sinc_filter, sinc_filter_len, in_dim, sampling_rate, 
                 res_ch_1, res_ch_2, gru_node, gru_layer, emb_dim, num_class):
        super(RawNet, self).__init__()

        # sinc filter layer
        self.m_sinc_conv = SincConv2(
            num_sinc_filter, 
            kernel_size = sinc_filter_len,
            in_channels = in_dim, 
            sample_rate = sampling_rate, 
            flag_pad = False, 
            flag_trainable=False)
        
        # res block group
        self.m_resgroup = torch_nn.Sequential(
            nii_nn.BatchNorm1DWrapper(num_sinc_filter),
            torch_nn.SELU(),
            Residual_block([num_sinc_filter, res_ch_1], flag_bn_input=False),
            FMS(res_ch_1),
            Residual_block([res_ch_1, res_ch_1], flag_bn_input=True),
            FMS(res_ch_1),
            Residual_block([res_ch_1, res_ch_2], flag_bn_input=True),
            FMS(res_ch_2),
            Residual_block([res_ch_2, res_ch_2], flag_bn_input=True),
            FMS(res_ch_2),
            Residual_block([res_ch_2, res_ch_2], flag_bn_input=True),
            FMS(res_ch_2),
            Residual_block([res_ch_2, res_ch_2], flag_bn_input=True),
            FMS(res_ch_2),
        )
        
        # GRU part
        self.m_before_gru = torch_nn.Sequential(
            nii_nn.BatchNorm1DWrapper(res_ch_2),
            torch_nn.SELU()
        )
        self.m_gru = torch_nn.GRU(input_size = res_ch_2,
                                  hidden_size = gru_node,
                                  num_layers = gru_layer,
                                  batch_first = True)
        
        self.m_emb = torch_nn.Linear(in_features = gru_node, 
                                     out_features = emb_dim)

        
        # output score
        self.m_output = torch_nn.Linear(in_features = emb_dim,
                                        out_features = num_class, 
                                        bias=True)
        # 
        self.logsoftmax = torch_nn.LogSoftmax(dim=1)
        return
    
    def _compute_embedding(self, x):
        """
        input
        -----
          x: tensor, (batch, length, dim)
          
        output
        ------
          y: tensor, (batch, emb_dim)
        """
        batch, length, dim  = x.shape
        # 
        x = self.m_sinc_conv(x)
        x = self.m_resgroup(x)
        x, _ = self.m_gru(self.m_before_gru(x))
        return self.m_emb(x[:, -1, :])
        
        
    def _compute_score(self, emb, inference=True):
        """
        input
        -----
          emb: tensor, (batch, emb_dim)
          
        output
        ------
          score: tensor, (batch, num_class)
          
        Score here refers to 
        """
        # we should not use logsoftmax if we will use CrossEntropyLoss
        flag_logsoftmax = False

        if inference:
            # no softmax
            return self.m_output(emb)
        elif flag_logsoftmax:
            # Logsoftmax for training loss
            # this is used when the training criterion is NLLoss
            return self.logsoftmax(self.m_output(emb))
        else:
            return self.m_output(emb)
    
    def forward(self, x):
        """
        input
        -----
          x: tensor, (batch, length, dim)
          
        output
        ------
          y: tensor, (batch, num_class)
        
        y is the log-probablity after softmax
        """
        emb = self._compute_embedding(x)
        return self._compute_score(emb, inference=False)
    
    def inference(self, x):
        """
        input
        -----
          x: tensor, (batch, length, dim)
          
        output
        ------
          y: tensor, (batch, num_class)
        
        y is the input activation to softmax
        """
        emb = self._compute_embedding(x)
        return self._compute_score(emb, inference=True)

if __name__ == "__main__":
    print("Definition of RawNet2")
