##!/usr/bin/env python
"""
Major blocks defined for NSF
These blocks are originall defined in ../project/01_nsf/*/.model.py

Definition are gathered here for convience.

CondModule, SourceModule, and FilterModule are not copied here since 
they may change according to the model for certain application
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
__copyright__ = "Copyright 2021, Xin Wang"

#######
# Neural filter block
#######

class NeuralFilterBlock(torch_nn.Module):
    """ Wrapper over a single filter block
    NeuralFilterBlock(signal_size, hidden_size, kernel_size, conv_num=10)

    args
    ----
      signal_size: int, input signal is in shape (batch, length, signal_size)
      hidden_size: int, output of conv layers is (batch, length, hidden_size)
      kernel_size: int, kernel size of the conv layers
      conv_num: number of conv layers in this neural filter block (default 10)
    
      legacy_scale: Bool, whether load scale as parameter or magic number
                  To be compatible with old models that defines self.scale
                  No impact on the result, just different ways to load a 
                  fixed self.scale
    """
    def __init__(self, signal_size, hidden_size, kernel_size=3, conv_num=10,
                 legacy_scale = False):
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
        tmp = [nii_nn.Conv1dKeepLength(hidden_size, hidden_size, x, \
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

        # a simple scale: to be consistent with CURRENNT implementation
        if legacy_scale:
            # in case this scale is defined as model parameter in 
            # some old models
            self.scale = torch_nn.Parameter(
                torch.tensor([0.1]), requires_grad=False)
        else:
            # simple hyper-parameter should be OK
            self.scale = 0.1
        return

    def forward(self, signal, context):
        """ 
        input
        -----
          signal (batchsize, length, signal_size)
          context (batchsize, length, hidden_size)
          
          context is produced from the condition module
        
        output
        ------
          output: (batchsize, length, signal_size)
        """
        # expand dimension
        tmp_hidden = self.l_ff_1_tanh(self.l_ff_1(signal))
        
        # loop over dilated convs
        # output of a d-conv is input + context + d-conv(input)
        for l_conv in self.l_convs:
            tmp_hidden = tmp_hidden + l_conv(tmp_hidden) + context
            
        # to be consistent with legacy configuration in CURRENNT
        tmp_hidden = tmp_hidden * self.scale
        
        # compress the dimesion and skip-add
        tmp_hidden = self.l_ff_2_tanh(self.l_ff_2(tmp_hidden))
        tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
        output_signal = tmp_hidden + signal
        return output_signal

############################
# Source signal generator
############################

class SineGen(torch_nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0, 
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    
    args
    ----
      samp_rate: flaot, sampling rate in Hz
      harmonic_num: int, number of harmonic overtones (default 0, i.e., only F0)
      sine_amp: float, amplitude of sine-wavefrom (default 0.1)
      noise_std: float, std of Gaussian noise (default 0.003)
      voiced_threshold: int, F0 threshold for U/V classification (default 0)
                        F0 < voiced_threshold will be set as unvoiced regions
                         
      flag_for_pulse: Bool, whether this SinGen is used inside PulseGen 
                      (default False)
    
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    def __init__(self, samp_rate, harmonic_num = 0, sine_amp = 0.1, 
                 noise_std = 0.003, voiced_threshold = 0, flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        return

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv
            
    def _f02sine(self, f0_values):
        """ 
        input
        -----
          f0_values: (batchsize, length_in_time, dim)
          where dim is the number of fundamental tone plus harmonic overtones
         
          f0_values are supposed to be up-sampled. In other words, length should
          be equal to the number of waveform sampling points.

        output
        ------
          sine_values: (batchsize, length_in_times, dim)

        sine_values[i, :, k] is decided by the F0s in f0_values[i, :, k]
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1
        
        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2],\
                              device = f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
                
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x *2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - 
                                tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) 
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every 
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation
            
            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            
            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within 
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return  sines
    
    
    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        
        input
        -----
          F0: tensor(batchsize, length, dim=1)
              Input F0 should be discontinuous.
              F0 for unvoiced steps should be 0
       
        output
        ------
          sine_tensor: tensor(batchsize, length, output_dim)
          output uv: tensor(batchsize, length, 1)

        where output_dim = 1 + harmonic_num
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, \
                                    device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx+1] = f0_buf[:, :, 0] * (idx+2)
                
            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            
            # generate uv signal
            #uv = torch.ones(f0.shape)
            #uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)
            
            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            #.       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1-uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            
            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise

        return sine_waves, uv, noise





if __name__ == "__main__":
    print("Definition of major components in NSF")
