#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

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

###############
## Utility functions
##
def f_padsize_conv1d_stride1(dilation_size, kernel_size):
    """
    pad_size = padsize_conv1d_stride1(dilation_size, kernel_size):
    How many points to pad so that input and output of Conv1D has the
    same temporal length.
    Assume stride = 1, kernel_size is odd number
    
    Note: this is for padding on both sides of the data
    See https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
    """
    if kernel_size % 2 == 0:
        print("Padding_conv1d_stride1: expect kernel_size as odd number")
        sys.exit(1)
    return dilation_size * (kernel_size - 1) // 2

##############
# Building blocks (torch.nn modules + dimension operation)
# 
    
# For blstm
class BLSTMLayer(torch_nn.Module):
    """ Wrapper over dilated conv1D
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    We want to keep the length the same
    """
    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = torch_nn.LSTM(input_dim, output_dim // 2, \
                                     bidirectional=True)
    def forward(self, x):
        # permute to (length, batchsize=1, dim)
        blstm_data, _ = self.l_blstm(x.permute(1, 0, 2))
        # permute it backt to (batchsize=1, length, dim)
        return blstm_data.permute(1, 0, 2)
        
        
# 1D dilated convolution that keep the input/output length
class Conv1dKeepLength(torch_nn.Conv1d):
    """ Wrapper for causal convolution
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is applied
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True):
        super(Conv1dKeepLength, self).__init__(
            input_dim, output_dim, kernel_s, stride=stride,
            padding = 0, dilation = dilation_s, groups=groups, bias=bias)

        self.causal = causal
        # input & output length will be the same        
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le

        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()
        
    def forward(self, data):
        # permute to (batchsize=1, dim, length)
        # add one dimension (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length)
        x = torch_nn_func.pad(data.permute(0, 2, 1).unsqueeze(2), \
                              (self.pad_le, self.pad_ri, 0, 0)).squeeze(2)
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output.permute(0, 2, 1)


# Moving average
class MovingAverage(Conv1dKeepLength):
    """ Wrapper to define a moving average smoothing layer
    """
    def __init__(self, feature_dim, window_len, causal=False):
        super(MovingAverage, self).__init__(
            feature_dim, feature_dim, 1, window_len, causal,
            groups=feature_dim, bias=False, tanh=False)
        # set the weighting coefficients
        torch_nn.init.constant_(self.weight, 1/window_len)
        # turn off grad for this layer
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, data):
        return super(MovingAverage, self).forward(data)

    
# Up sampling
class UpSampleLayer(torch_nn.Module):
    """ Wrapper over up-sampling
    Input tensor: (batchsize=1, length, dim)
    Ouput tensor: (batchsize=1, length * up-sampling_factor, dim)
    """
    def __init__(self, feature_dim, up_sampling_factor, smoothing=False):
        super(UpSampleLayer, self).__init__()
        # wrap a up_sampling layer
        self.scale_factor = up_sampling_factor
        self.l_upsamp = torch_nn.Upsample(scale_factor=self.scale_factor)
        if smoothing:
            self.l_ave1 = MovingAverage(feature_dim, self.scale_factor)
            self.l_ave2 = MovingAverage(feature_dim, self.scale_factor)
        else:
            self.l_ave1 = torch_nn.Identity()
            self.l_ave2 = torch_nn.Identity()
        return
    
    def forward(self, x):
        # permute to (batchsize=1, dim, length)
        up_sampled_data = self.l_upsamp(x.permute(0, 2, 1))
        # permute it backt to (batchsize=1, length, dim)
        # and do two moving average
        return self.l_ave1(self.l_ave2(up_sampled_data.permute(0, 2, 1)))
    

# For filter block
class NeuralFilterBlock(torch_nn.Module):
    """ Wrapper over a single filter block
    """
    def __init__(self, signal_size, hidden_size, \
                 kernel_size=3, conv_num=10):
        super(NeuralFilterBlock, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv_num = conv_num
        self.dilation_size = [np.power(2, x) for x in np.arange(conv_num)]

        # ff layer to expand dimension
        self.l_ff_1 = torch_nn.Linear(signal_size, hidden_size, \
                                      bias=False)
        self.l_ff_1_tanh = torch_nn.Tanh()
        
        # dilated conv layers
        tmp = [Conv1dKeepLength(hidden_size, hidden_size, x, \
                                kernel_size, causal=True, bias=False) \
               for x in self.dilation_size]
        self.l_convs = torch_nn.ModuleList(tmp)
        
        # ff layer to de-expand dimension
        self.l_ff_2 = torch_nn.Linear(hidden_size, hidden_size//4,
                                      bias=False)
        self.l_ff_2_tanh = torch_nn.Tanh()
        self.l_ff_3 = torch_nn.Linear(hidden_size//4, signal_size,
                                      bias=False)
        self.l_ff_3_tanh = torch_nn.Tanh()        
        return

    def forward(self, signal, context):
        """ 
        Assume: signal (batchsize=1, length, signal_size)
                context (batchsize=1, length, hidden_size)
        Output: (batchsize=1, length, signal_size)
        """
        # expand dimension
        tmp_hidden = self.l_ff_1_tanh(self.l_ff_1(signal))
        
        # loop over dilated convs
        # output of a d-conv is input + context + d-conv(input)
        for l_conv in self.l_convs:
            tmp_hidden = tmp_hidden + l_conv(tmp_hidden) + context
            
        # to be consistent with legacy configuration in CURRENNT
        tmp_hidden = tmp_hidden * 0.1
        
        # compress the dimesion and skip-add
        tmp_hidden = self.l_ff_2_tanh(self.l_ff_2(tmp_hidden))
        tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
        output_signal = tmp_hidden + signal
        return output_signal
    
#####
## Model definition
## 

## For condition module only provide Spectral feature
class CondModule(torch_nn.Module):
    """ 
    """
    def __init__(self, input_dim, output_dim, up_sample, \
                 blstm_s = 64, cnn_kernel_s = 3):
        super(CondModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.blstm_s = blstm_s
        self.cnn_kernel_s = cnn_kernel_s

        self.l_blstm = BLSTMLayer(input_dim, self.blstm_s)
        self.l_conv1d = Conv1dKeepLength(self.blstm_s, output_dim, 1, \
                                         self.cnn_kernel_s)
        self.l_upsamp = UpSampleLayer(self.output_dim, self.up_sample, \
                                      True)

    def forward(self, x):
        return self.l_upsamp(self.l_conv1d(self.l_blstm(x)))

# For source module
class SourceModule(torch_nn.Module):
    """
    """
    def __init__(self, amplitude, up_samp):
        super(SourceModule, self).__init__()
        self.dim = 1
        self.amp = amplitude
        self.up_samp = up_samp
        self.l_upsamp = UpSampleLayer(self.dim, self.up_samp, False)
        # use bias=0
        self.l_linear = torch_nn.Linear(self.dim, self.dim, bias=False)
        self.l_tanh = torch_nn.Tanh()

    def forward(self, x):
        noise = torch.randn_like(self.l_upsamp(x)) * self.amp / 3 
        return self.l_tanh(self.l_linear(noise))
        
        
# For Filter module
class FilterModule(torch_nn.Module):
    """
    """
    def __init__(self, signal_size, hidden_size, block_num = 5, \
                 kernel_size = 3, conv_num_in_block = 10):
        super(FilterModule, self).__init__()        
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block

        tmp = [NeuralFilterBlock(signal_size, hidden_size, \
                                 kernel_size, conv_num_in_block) \
               for x in range(self.block_num)]
        self.l_blocks = torch_nn.ModuleList(tmp)

    def forward(self, signal, context):
        for l_block in self.l_blocks:
            signal = l_block(signal, context)
        return signal
        
        

## FOR MODEL
class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, mean_std=None):
        super(Model, self).__init__()

        torch.manual_seed(1)
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)

        #
        self.noise_amp = 0.001
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.hidden_dim = 64
        self.upsamp_rate = 288
        self.cnn_kernel_size = 3
        self.filter_block_num = 5
        self.cnn_num_in_block = 10
        
        # hidde layers
        self.m_source = SourceModule(self.noise_amp, self.upsamp_rate)
        self.m_condition = CondModule(self.input_dim, self.hidden_dim, \
                                      self.upsamp_rate, \
                                      cnn_kernel_s = self.cnn_kernel_size)
        self.m_filter = FilterModule(self.output_dim, self.hidden_dim,\
                                     self.filter_block_num, \
                                     self.cnn_kernel_size, \
                                     self.cnn_num_in_block)
        
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        """
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.zeros([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.zeros([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        """
        return y * self.output_std + self.output_mean
    
    def forward(self, x):
        """ definition of forward method 
        Assume x (batchsize=1, length, dim)
        Return output(batchsize=1, length)
        """
        # normalize the data
        x = self.normalize_input(x)
        cond_feature = self.m_condition(x)
        source_signal = self.m_source(x[:, :, 0:1])
        output = self.m_filter(source_signal, cond_feature)
        return output.squeeze(-1)
    
    
class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        self.frame_hops = [80, 40, 640]
        self.frame_lens = [320, 80, 1920]
        self.fft_n = [4096, 4096, 4096]
        self.win = torch.hann_window
        self.amp_floor = 0.00000001
        self.loss1 = torch_nn.MSELoss()
        self.loss2 = torch_nn.MSELoss()
        self.loss3 = torch_nn.MSELoss()
        self.loss = [self.loss1, self.loss2, self.loss3]
        
    def compute(self, output, target):
        """ Loss().compute(output, target) should return
        the Loss in torch.tensor format
        Assume output and target as (batchsize=1, length)
        """
        # convert from (batchsize=1, length, dim=1) to (1, length)
        if target.ndim == 3:
            target.squeeze_(-1)
            
        # compute loss
        loss = 0
        for frame_shift, frame_len, fft_p, loss_f in \
            zip(self.frame_hops, self.frame_lens, self.fft_n, self.loss):
            x_stft = torch.stft(output, fft_p, frame_shift, frame_len, \
                                window=self.win(frame_len), onesided=True)
            y_stft = torch.stft(target, fft_p, frame_shift, frame_len, \
                                window=self.win(frame_len), onesided=True)
            x_sp_amp = torch.log(torch.norm(x_stft, 2, -1).pow(2) + \
                                   self.amp_floor)
            y_sp_amp = torch.log(torch.norm(y_stft, 2, -1).pow(2) + \
                                   self.amp_floor)
            loss += loss_f(x_sp_amp, y_sp_amp)
        return loss

    
if __name__ == "__main__":
    print("Definition of model")

    
