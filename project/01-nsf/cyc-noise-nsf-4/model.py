#!/usr/bin/env python
"""
model.py for Cyclic-noise-NSF
version: 4
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import core_scripts.other_tools.debug as nii_debug


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##############
# Building blocks (torch.nn modules + dimension operation)

#     
# For blstm
class BLSTMLayer(torch_nn.Module):
    """ Wrapper over BLSTM
    Input tensor:  (batchsize, length, dim_in)
    Output tensor: (batchsize, length, dim_out)
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
        
#        
# 1D dilated convolution that keep the input/output length
class Conv1dKeepLength(torch_nn.Conv1d):
    """ Wrapper for causal convolution
    Input tensor:  (batchsize, length, dim_in)
    Output tensor: (batchsize, length, dim_out)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is optional
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, \
                 tanh = True, pad_mode='constant'):
        super(Conv1dKeepLength, self).__init__(
            input_dim, output_dim, kernel_s, stride=stride,
            padding = 0, dilation = dilation_s, groups=groups, bias=bias)

        self.pad_mode = pad_mode

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
        # https://github.com/pytorch/pytorch/issues/1333
        x = torch_nn_func.pad(data.permute(0, 2, 1).unsqueeze(2), \
                              (self.pad_le, self.pad_ri,0,0), \
                              mode = self.pad_mode).squeeze(2)
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output.permute(0, 2, 1)

# 
# Moving average
class MovingAverage(Conv1dKeepLength):
    """ Wrapper to define a moving average smoothing layer
    Note: MovingAverage can be implemented using TimeInvFIRFilter too.
          Here we define another Module dicrectly on Conv1DKeepLength
    """
    def __init__(self, feature_dim, window_len, causal=False, \
                 pad_mode='replicate'):
        super(MovingAverage, self).__init__(
            feature_dim, feature_dim, 1, window_len, causal,
            groups=feature_dim, bias=False, tanh=False, \
            pad_mode=pad_mode)
        # set the weighting coefficients
        torch_nn.init.constant_(self.weight, 1/window_len)
        # turn off grad for this layer
        for p in self.parameters():
            p.requires_grad = False
            
    def forward(self, data):
        return super(MovingAverage, self).forward(data)

# 
# FIR filter layer
class TimeInvFIRFilter(Conv1dKeepLength):
    """ Wrapper to define a FIR filter over Conv1d
        FIR Filtering is conducted on each dimension (channel)
        independently, i.e., groups=channel_num in conv1d
    """ 
    def __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_trn=False):
        """ __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_trn=False)
        feature_dim: dimension of input data
        filter_coef: 1-D tensor of filter coefficients
        causal: FIR is causal or not (default: true)
        flag_trn: whether learn the filter coefficients (default false)

        Input data: (batchsize=1, length, feature_dim)
        Output data: (batchsize=1, length, feature_dim)
        """
        super(TimeInvFIRFilter, self).__init__(
            feature_dim, feature_dim, 1, filter_coef.shape[0], causal,
            groups=feature_dim, bias=False, tanh=False)
        
        if filter_coef.ndim == 1:
            # initialize weight using provided filter_coef
            with torch.no_grad():
                tmp_coef = torch.zeros([feature_dim, 1, 
                                        filter_coef.shape[0]])
                tmp_coef[:, 0, :] = filter_coef
                tmp_coef = torch.flip(tmp_coef, dims=[2])
                self.weight = torch.nn.Parameter(tmp_coef, 
                                                 requires_grad=flag_trn)
        else:
            print("TimeInvFIRFilter expects filter_coef as 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)

    def forward(self, data):                
        return super(TimeInvFIRFilter, self).forward(data)    

class TimeVarFIRFilter(torch_nn.Module):
    """ TimeVarFIRFilter
    Given sequences of filter coefficients and a signal, do filtering
    
    Filter coefs: (batchsize, signal_length, filter_order = K)
    Signal:       (batchsize, signal_length, 1)
    
    For batch 0:
     For n in [1, sequence_length):
       output(0, n, 1) = \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
       
    Note: filter coef (0, n, :) is only used to compute the output 
          at (0, n, 1)
    """
    def __init__(self):
        super(TimeVarFIRFilter, self).__init__()
    
    def forward(self, signal, f_coef):
        """ 
        Filter coefs: (batchsize=1, signal_length, filter_order = K)
        Signal:       (batchsize=1, signal_length, 1)
        
        Output:       (batchsize=1, signal_length, 1)
        
        For n in [1, sequence_length):
         output(0, n, 1) = \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)
          
        This method may be not efficient:
        
        Suppose signal [x_1, ..., x_N], filter [a_1, ..., a_K]
        output         [y_1, y_2, y_3, ..., y_N, *, * ... *]
               = a_1 * [x_1, x_2, x_3, ..., x_N,   0, ...,   0]
               + a_2 * [  0, x_1, x_2, x_3, ..., x_N,   0, ...,  0]
               + a_3 * [  0,   0, x_1, x_2, x_3, ..., x_N, 0, ...,  0]
        """
        signal_l = signal.shape[1]
        order_k = f_coef.shape[-1]

        # pad to (batchsize=1, signal_length + filter_order-1, dim)
        padded_signal = torch_nn_func.pad(signal, (0, 0, 0, order_k - 1))
        
        y = torch.zeros_like(signal)
        # roll and weighted sum, only take [0:signal_length]
        for k in range(order_k):
            y += torch.roll(padded_signal, k, dims=1)[:, 0:signal_l, :] \
                      * f_coef[:, :, k:k+1] 
        # done
        return y

class SignalsConv1d(torch_nn.Module):
    """ Filtering input signal with time invariant filter
    Note: FIRFilter conducted filtering given fixed FIR weight
          SignalsConv1d convolves two signals
    Note: this is based on torch.nn.functional.conv1d
    
    """                                                                  
    def __init__(self):
        super(SignalsConv1d, self).__init__()
        
    def forward(self, signal, system_ir):
        """ output = forward(signal, system_ir)
        
        signal:    (batchsize, length1, dim)
        system_ir: (length2, dim) 
        
        output:    (batchsize, length1, dim)
        """ 
        if signal.shape[-1] != system_ir.shape[-1]:
            print("Error: SignalsConv1d expects shape:")
            print("signal    (batchsize, length1, dim)")
            print("system_id (batchsize, length2, dim)")
            print("But received signal: {:s}".format(str(signal.shape)))
            print(" system_ir: {:s}".format(str(system_ir.shape)))
            sys.exit(1)
        padding_length = system_ir.shape[0] - 1
        groups = signal.shape[-1]
        
        # pad signal on the left 
        signal_pad = torch_nn_func.pad(signal.permute(0, 2, 1),\
                                       (padding_length, 0))
        # prepare system impulse response as (dim, 1, length2)
        # also flip the impulse response
        ir = torch.flip(system_ir.unsqueeze(1).permute(2, 1, 0), \
                        dims=[2])
        # convolute
        output = torch_nn_func.conv1d(signal_pad, ir, groups=groups)
        return output.permute(0, 2, 1)


# Sinc filter generator
class SincFilter(torch_nn.Module):
    """ SincFilter
        Given the cut-off-frequency, produce the low-pass and high-pass
        windowed-sinc-filters.
        If input cut-off-frequency is (batchsize=1, signal_length, 1),
        output filter coef is (batchsize=1, signal_length, filter_order).
        For each time step in [1, signal_length), we calculate one
        filter for low-pass sinc filter and another for high-pass filter.
        
        Example:
        import scipy
        import scipy.signal
        import numpy as np
        
        filter_order = 31
        cut_f = 0.2
        sinc_layer = SincFilter(filter_order)
        lp_coef, hp_coef = sinc_layer(torch.ones(1, 10, 1) * cut_f)
        
        w, h1 = scipy.signal.freqz(lp_coef[0, 0, :].numpy(), [1])
        w, h2 = scipy.signal.freqz(hp_coef[0, 0, :].numpy(), [1])
        plt.plot(w, 20*np.log10(np.abs(h1)))
        plt.plot(w, 20*np.log10(np.abs(h2)))
        plt.plot([cut_f * np.pi, cut_f * np.pi], [-100, 0])
    """
    def __init__(self, filter_order):
        super(SincFilter, self).__init__()
        # Make the filter oder an odd number
        #  [-(M-1)/2, ... 0, (M-1)/2]
        # 
        self.half_k = (filter_order - 1) // 2
        self.order = self.half_k * 2 +1
        
    def hamming_w(self, n_index):
        """ prepare hamming window for each time step
        n_index (batchsize=1, signal_length, filter_order)
            For each step, n_index.shape is [-(M-1)/2, ... 0, (M-1)/2]
            where,
            n_index[0, 0, :] = [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 1, :] = [-(M-1)/2, ... 0, (M-1)/2]
            ...
        output  (batchsize=1, signal_length, filter_order)
            output[0, 0, :] = hamming_window
            output[0, 1, :] = hamming_window
            ...
        """
        # Hamming window
        return 0.54 + 0.46 * torch.cos(2 * np.pi * n_index / self.order)
    
    def sinc(self, x):
        """ Normalized sinc-filter sin( pi * x) / pi * x
        https://en.wikipedia.org/wiki/Sinc_function
        
        Assume x (batchsize, signal_length, filter_order) and 
        x[0, 0, :] = [-half_order, - half_order+1, ... 0 ..., half_order]
        x[:, :, self.half_order] -> time index = 0, sinc(0)=1
        """
        y = torch.zeros_like(x)
        y[:,:,0:self.half_k]=torch.sin(np.pi * x[:, :, 0:self.half_k]) \
                              / (np.pi * x[:, :, 0:self.half_k])
        y[:,:,self.half_k+1:]=torch.sin(np.pi * x[:, :, self.half_k+1:])\
                               / (np.pi * x[:, :, self.half_k+1:])
        y[:,:,self.half_k] = 1
        return y
        
    def forward(self, cut_f):
        """ lp_coef, hp_coef = forward(self, cut_f)
        cut-off frequency cut_f (batchsize=1, length, dim = 1)
    
        lp_coef: low-pass filter coefs  (batchsize, length, filter_order)
        hp_coef: high-pass filter coefs (batchsize, length, filter_order)
        """
        # create the filter order index
        with torch.no_grad():   
            # [- (M-1) / 2, ..., 0, ..., (M-1)/2]
            lp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            # [[[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ],
            #  [[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ]]
            lp_coef = lp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            
            hp_coef = torch.arange(-self.half_k, self.half_k + 1, 
                                   device=cut_f.device)
            hp_coef = hp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)
            
            # temporary buffer of [-1^n] for gain norm in hp_coef
            tmp_one = torch.pow(-1, hp_coef)
            
        # unnormalized filter coefs with hamming window
        lp_coef = cut_f * self.sinc(cut_f * lp_coef) \
                  * self.hamming_w(lp_coef)
        
        hp_coef = (self.sinc(hp_coef) \
                   - cut_f * self.sinc(cut_f * hp_coef)) \
                  * self.hamming_w(hp_coef)
        
        # normalize the coef to make gain at 0/pi is 0 dB
        # sum_n lp_coef[n]
        lp_coef_norm = torch.sum(lp_coef, axis=2).unsqueeze(-1)
        # sum_n hp_coef[n] * -1^n
        hp_coef_norm = torch.sum(hp_coef * tmp_one, axis=2).unsqueeze(-1)
        
        lp_coef = lp_coef / lp_coef_norm
        hp_coef = hp_coef / hp_coef_norm
        
        # return normed coef
        return lp_coef, hp_coef

# 
# Up sampling
class UpSampleLayer(torch_nn.Module):
    """ Wrapper over up-sampling
    Input tensor: (batchsize, length, dim)
    Ouput tensor: (batchsize, length * up-sampling_factor, dim)
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
    

# Neural filter block (1 block)
class NeuralFilterBlock(torch_nn.Module):
    """ Wrapper over a single filter block
    """
    def __init__(self, signal_size, hidden_size, for_har_component=True,\
                 kernel_size=3, conv_num=10):
        super(NeuralFilterBlock, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv_num = conv_num
        self.dilation_s = [np.power(2, x) for x in np.arange(conv_num)]
        self.for_har = for_har_component

        # ff layer to expand dimension
        self.l_ff_1 = torch_nn.Linear(signal_size, hidden_size)
        self.l_ff_1_tanh = torch_nn.Tanh()
        
        # dilated conv layers
        tmp = [Conv1dKeepLength(hidden_size, hidden_size, x, \
                                kernel_size, causal=True, bias=True) \
               for x in self.dilation_s]
        self.l_convs = torch_nn.ModuleList(tmp)
                
        # ff layer to de-expand dimension
        self.l_ff_2 = torch_nn.Linear(hidden_size, hidden_size//4)
        self.l_ff_2_tanh = torch_nn.Tanh()
        self.l_ff_3 = torch_nn.Linear(hidden_size//4, signal_size)
        self.l_ff_3_tanh = torch_nn.Tanh()        

        # a simple scale
        self.scale = torch_nn.Parameter(torch.tensor([0.1]), 
                                        requires_grad=False)
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
        tmp_hidden = tmp_hidden * self.scale
        
        # compress the dimesion and skip-add
        tmp_hidden = self.l_ff_2_tanh(self.l_ff_2(tmp_hidden))
        if self.for_har:
            # if this block is used for harmonic component
            tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
            output_signal = tmp_hidden + signal
        else:
            # for noise component, no need to use skip-connection
            output_signal = self.l_ff_3(tmp_hidden)
        return output_signal
    
# 
# Sine waveform generator
class SineGen(torch_nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0, 
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """
    def __init__(self, samp_rate, harmonic_num = 0, 
                 sine_amp = 0.1, noise_std = 0.003,
                 voiced_threshold = 0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
    
    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv
            
    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
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
            # This will not change F0 of sine because (x-1) * 2*pi = x * 2*pi
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
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
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



class PulseGen(torch_nn.Module):
    """ Definition of Pulse train generator
    
    There are many ways to implement pulse generator. 
    Here, PulseGen is based on SinGen. For a perfect 
    """
    def __init__(self, samp_rate, pulse_amp = 0.1,
                 noise_std = 0.003, voiced_threshold = 0):
        super(PulseGen, self).__init__()
        self.pulse_amp = pulse_amp
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.noise_std = noise_std
        self.l_sinegen = SineGen(self.sampling_rate, harmonic_num=0,\
                                 sine_amp=self.pulse_amp, noise_std=0,\
                                 voiced_threshold=self.voiced_threshold,\
                                 flag_for_pulse=True)
    
    def forward(self, f0):
        """ Pulse train generator
        pulse_train, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output pulse_train: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        
        Note: self.l_sine doesn't make sure that the initial phase of 
        a voiced segment is np.pi, the first pulse in a voiced segment
        may not be at the first time step within a voiced segment
        """
        with torch.no_grad():
            sine_wav, uv, noise = self.l_sinegen(f0)
            
            # sine without additive noise
            pure_sine = sine_wav - noise
            
            # step t corresponds to a pulse if
            # sine[t] > sine[t+1] & sine[t] > sine[t-1] 
            # & sine[t-1], sine[t+1], and sine[t] are voiced
            # or 
            # sine[t] is voiced, sine[t-1] is unvoiced
            # we use torch.roll to simulate sine[t+1] and sine[t-1]
            sine_1 = torch.roll(pure_sine, shifts=1, dims=1)
            uv_1 = torch.roll(uv, shifts=1, dims=1)
            uv_1[:, 0, :] = 0
            sine_2 = torch.roll(pure_sine, shifts=-1, dims=1)
            uv_2 = torch.roll(uv, shifts=-1, dims=1)
            uv_2[:, -1, :] = 0
            
            loc = (pure_sine > sine_1) * (pure_sine > sine_2) \
                * (uv_1 > 0) * (uv_2 > 0) * (uv > 0) \
                + (uv_1 < 1) * (uv > 0)
            
            # pulse train without noise
            pulse_train = pure_sine * loc

            # additive noise to pulse train
            # note that noise from sinegen is zero in voiced regions
            pulse_noise = torch.randn_like(pure_sine) * self.noise_std
            
            # with additive noise on pulse, and unvoiced regions
            pulse_train += pulse_noise * loc + pulse_noise * (1 - uv)
        return pulse_train, sine_wav, uv, pulse_noise


class CyclicNoiseGen_v1(torch_nn.Module):
    """ CyclicnoiseGen_v1
    Cyclic noise with a single parameter of beta.
    Pytorch v1 implementation assumes f_t is also fixed
    """
    def __init__(self, samp_rate, 
                 noise_std = 0.003, voiced_threshold = 0):
        super(CyclicNoiseGen_v1, self).__init__()
        self.samp_rate = samp_rate
        self.noise_std = noise_std
        self.voiced_threshold = voiced_threshold
    
        self.l_pulse = PulseGen(samp_rate, pulse_amp=1.0, \
                                noise_std=noise_std, \
                                voiced_threshold=voiced_threshold)
        self.l_conv = SignalsConv1d()
        
    def noise_decay(self, beta, f0mean):
        """ decayed_noise = noise_decay(beta, f0mean)
        decayed_noise =  n[t]exp(-t * f_mean / beta / samp_rate)
        
        beta: (dim=1) or (batchsize=1, 1, dim=1)
        f0mean (batchsize=1, 1, dim=1)
        
        decayed_noise (batchsize=1, length, dim=1)
        """
        with torch.no_grad():
            # exp(-1.0 n / T) < 0.01 => n > -log(0.01)*T = 4.60*T 
            # truncate the noise when decayed by -40 dB
            length = 4.6 * self.samp_rate / f0mean
            length = length.int()
            time_idx = torch.arange(0, length, device=beta.device)
            time_idx = time_idx.unsqueeze(0).unsqueeze(2)
            time_idx = time_idx.repeat(beta.shape[0], 1, beta.shape[2])
        
        noise = torch.randn(time_idx.shape, device=beta.device)
            
        # due to Pytorch implementation, use f0_mean as the f0 factor
        decay = torch.exp(-time_idx * f0mean / beta / self.samp_rate)
        return noise * self.noise_std * decay
        
    def forward(self, f0s, beta):
        """ Producde cyclic-noise 
        """
        # pulse train
        pulse_train, sine_wav, uv, noise = self.l_pulse(f0s)
        pure_pulse = pulse_train - noise * (1.0 - uv)
        
        # decayed_noise (length, dim=1)
        if (uv<1).all():
            # all unvoiced
            cyc_noise = torch.zeros_like(sine_wav)
        else:
            f0mean = f0s[uv>0].mean()
            
            decayed_noise = self.noise_decay(beta, f0mean)[0, :, :]
            # convolute
            cyc_noise = self.l_conv(pure_pulse, decayed_noise)
        
        # add noise in invoiced segments
        cyc_noise = cyc_noise + noise * (1.0 - uv)
        return cyc_noise, pulse_train, sine_wav, uv, noise

#####
## Model definition
## 

## For condition module only provide Spectral feature to Filter block
class CondModuleHnSincNSF(torch_nn.Module):
    """ Condition module for hn-sinc-NSF

    Upsample and transform input features
    CondModuleHnSincNSF(input_dimension, output_dimension, 
                        up_sample_rate,
                        blstm_dimension = 64, cnn_kernel_size = 3)
    
    Spec, F0, cut_off_freq = CondModuleHnSincNSF(features, F0)

    Both input features should be frame-level features
    If x doesn't contain F0, just ignore the returned F0
    
    CondModuleHnSincNSF(input_dim, output_dim, up_sample, 
                        blstm_s = 64, cnn_kernel_s = 3, 
                        voiced_threshold = 0):

    input_dim: sum of dimensions of input features
    output_dim: dim of the feature Spec to be used by neural filter-block
    up_sample: up sampling rate of input features
    blstm_s: dimension of the features from blstm (default 64)
    cnn_kernel_s: kernel size of CNN in condition module (default 3)
    voiced_threshold: f0 > voiced_threshold is voiced, otherwise unvoiced
    """
    def __init__(self, input_dim, output_dim, up_sample, \
                 blstm_s = 64, cnn_kernel_s = 3, voiced_threshold = 0):
        super(CondModuleHnSincNSF, self).__init__()

        # input feature dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.blstm_s = blstm_s
        self.cnn_kernel_s = cnn_kernel_s
        self.cut_f_smooth = up_sample * 4
        self.voiced_threshold = voiced_threshold

        # the blstm layer
        self.l_blstm = BLSTMLayer(input_dim, self.blstm_s)

        # the CNN layer (+1 dim for cut_off_frequence of sinc filter)
        self.l_conv1d = Conv1dKeepLength(self.blstm_s, \
                                         self.output_dim + 1, \
                                         dilation_s = 1, \
                                         kernel_s = self.cnn_kernel_s)
        # Upsampling layer for hidden features
        self.l_upsamp = UpSampleLayer(self.output_dim + 1, \
                                      self.up_sample, True)
        
        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = UpSampleLayer(1, self.up_sample, False)

        # Another smoothing layer to smooth the cut-off frequency
        # for sinc filters. Use a larger window to smooth
        self.l_cut_f_smooth = MovingAverage(1, self.cut_f_smooth)

    def get_cut_f(self, hidden_feat, f0):
        """ cut_f = get_cut_f(self, feature, f0)
        feature: (batchsize, length, dim=1)
        f0: (batchsize, length, dim=1)        
        """ 
        # generate uv signal
        uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
        # hidden_feat is between (-1, 1) after conv1d with tanh
        # (-0.2, 0.2) + 0.3 = (0.1, 0.5)
        # voiced:   (0.1, 0.5) + 0.4 = (0.5, 0.9)
        # unvoiced: (0.1, 0.5) = (0.1, 0.5)
        return hidden_feat * 0.2 + uv * 0.4 + 0.3
        
    
    def forward(self, feature, f0):
        """ spec, f0 = forward(self, feature, f0)
        feature: (batchsize, length, dim)
        f0: (batchsize, length, dim=1), which should be F0 at frame-level
        
        spec: (batchsize, length, self.output_dim), at wave-level
        f0: (batchsize, length, 1), at wave-level
        """ 
        # Different from the paper, for simplicitiy, output of conv1d
        # is fed to the neural filter blocks without concatenating F0
        tmp = self.l_upsamp(self.l_conv1d(self.l_blstm(feature)))
        spec = tmp[:, :, 0:self.output_dim]

        # directly up-sample F0 without smoothing
        f0_upsamp = self.l_upsamp_F0(f0)

        # get the cut-off-frequency from output of CNN
        cut_f = self.get_cut_f(tmp[:, :, self.output_dim:], f0_upsamp)
        # smooth the cut-off-frequency using fixed average smoothing
        cut_f_smoothed = self.l_cut_f_smooth(cut_f)

        # return
        return spec, f0_upsamp, cut_f_smoothed

# For source module
class SourceModuleCycNoise_v1(torch_nn.Module):
    """ SourceModuleCycNoise_v1
    SourceModule(sampling_rate, noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    
    noise_std: std of Gaussian noise (default: 0.003)
    voiced_threshold: threhold to set U/V given F0 (default: 0)

    cyc, noise, uv = SourceModuleCycNoise_v1(F0_upsampled, beta)
    F0_upsampled (batchsize, length, 1)
    beta (1)
    cyc (batchsize, length, 1)
    noise (batchsize, length, 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, \
                 noise_std=0.003, voiced_threshod=0):
        super(SourceModuleCycNoise_v1, self).__init__()
        self.sampling_rate = sampling_rate
        self.noise_std = noise_std
        self.l_cyc_gen = CyclicNoiseGen_v1(sampling_rate, noise_std,
                                           voiced_threshod)
        
        
    def forward(self, f0_upsamped, beta):
        """
        cyc, noise, uv = SourceModuleCycNoise_v1(F0, beta)
        F0_upsampled (batchsize, length, 1)
        beta (1)
        cyc (batchsize, length, 1)
        noise (batchsize, length, 1)
        uv (batchsize, length, 1)
        """
        # source for harmonic branch
        cyc, pulse, sine, uv, add_noi = self.l_cyc_gen(f0_upsamped, beta)
        
        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.noise_std / 3
        return cyc, noise, uv
        
        
# For Filter module
class FilterModuleCycNoiseNSF(torch_nn.Module):
    """ Filter for cyclic noise nsf
    FilterModuleCycNoiseNSF(signal_size, hidden_size, sinc_order = 31,
                          block_num = 5, kernel_size = 3, 
                          conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    sinc_order: order of the sinc filter
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block

    Usage:
    out = FilterModuleCycNoiseNSF(har_source, noi_source, cut_f, context)
    har_source: source for harmonic branch (batchsize, length, dim=1)
    noi_source: source for noise branch (batchsize, length, dim=1)
    cut_f: cut-off-frequency of sinc filters (batchsize, length, dim=1)
    context: hidden features to be added (batchsize, length, dim)
    out: (batchsize, length, dim=1)    
    """
    def __init__(self, signal_size, hidden_size, sinc_order = 31, \
                 block_num = 5, kernel_size = 3, conv_num_in_block = 10):
        super(FilterModuleCycNoiseNSF, self).__init__()        
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block
        self.sinc_order = sinc_order
        
        # filter blocks for harmonic branch
        tmp = [NeuralFilterBlock(signal_size, hidden_size, True, \
                                 kernel_size, conv_num_in_block) \
               for x in range(self.block_num)]
        self.l_har_blocks = torch_nn.ModuleList(tmp)

        # filter blocks for noise branch (only one block, 5 sub-blocks)
        tmp = [NeuralFilterBlock(signal_size, hidden_size, False, \
                                 kernel_size, conv_num_in_block // 2) \
               for x in range(1)]
        self.l_noi_blocks = torch_nn.ModuleList(tmp)

        # sinc filter generators and time-variant filtering layer
        self.l_sinc_coef = SincFilter(self.sinc_order)
        self.l_tv_filtering = TimeVarFIRFilter()
        # done
        

    def forward(self, har_component, noi_component, cond_feat, cut_f):
        """
        """
        # harmonic component
        # 
        hidden_signals = []
        for l_har_block in self.l_har_blocks:
            hidden_signal = l_har_block(har_component, cond_feat)
            hidden_signals.append(hidden_signal)
            har_component = hidden_signal

        # noise componebt
        for l_noi_block in self.l_noi_blocks:
            noi_component = l_noi_block(noi_component, cond_feat)
        
        # get sinc filter coefficients
        lp_coef, hp_coef = self.l_sinc_coef(cut_f)

        # time-variant filtering
        har_signal = self.l_tv_filtering(har_component, lp_coef)
        noi_signal = self.l_tv_filtering(noi_component, hp_coef)

        # get output 
        return har_signal + noi_signal, hidden_signals
        
        

## FOR MODEL
class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        self.input_dim = in_dim
        self.output_dim = out_dim
        
        # configurations
        # amplitude of sine waveform (for each harmonic)
        self.sine_amp = 0.1
        # standard deviation of Gaussian noise for additive noise
        self.noise_std = 0.003
        # dimension of hidden features in filter blocks
        self.hidden_dim = 64
        # upsampling rate on input acoustic features (16kHz * 5ms = 80)
        # assume input_reso has the same value
        self.upsamp_rate = prj_conf.input_reso[0]
        # sampling rate (Hz)
        self.sampling_rate = prj_conf.wav_samp_rate
        # CNN kernel size in filter blocks
        self.cnn_kernel_s = 3
        # number of filter blocks (for harmonic branch)
        # noise branch only uses 1 block
        self.filter_block_num = 5
        # number of dilated CNN in each filter block
        self.cnn_num_in_block = 10

        # sinc filter order (odd number)
        self.sinc_order = 31
        # number of harmonics for sine mask
        # note: cyclis-noise-nsf doesn't use harmonic overtone in source
        self.harmonic_num = 7 
        # beta parameter for cyclic-noise
        self.beta = 0.870

        # the three modules
        self.m_cond = CondModuleHnSincNSF(self.input_dim, \
                                          self.hidden_dim, \
                                          self.upsamp_rate, \
                                          cnn_kernel_s=self.cnn_kernel_s)

        self.m_source = SourceModuleCycNoise_v1(self.sampling_rate, 
                                                self.noise_std)
        
        self.m_filter = FilterModuleCycNoiseNSF(self.output_dim, \
                                                self.hidden_dim, \
                                                self.sinc_order, \
                                                self.filter_block_num, \
                                                self.cnn_kernel_s, \
                                                self.cnn_num_in_block)

        # one additional module to generate sine mask
        self.m_sinemask = SineGen(self.sampling_rate, self.harmonic_num)
        
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, \
                         data_mean_std=None):
        """ Load mean/std of input/output features
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
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
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
        # assume x[:, :, -1] is F0, denormalize F0
        f0 = x[:, :, -1:]
        # normalize the input features data
        feat = self.normalize_input(x)

        # condition module
        # features_for_filter_block, up-sampled F0, cut-off-frequency
        cond_feat, f0_upsamp, cut_f = self.m_cond(feat, f0)

        # source module
        # here we assume beta is fixed
        beta = torch.ones(1, 1, 1, device=f0_upsamp.device) * self.beta
        # harmonic-source signal, noise-source signal, uv flag
        har_source, noi_source, uv = self.m_source(f0_upsamp, beta)
        
        # neural filter module (including sinc-based FIR filtering)
        # output signal, hidden signals
        output, hidden = self.m_filter(har_source, noi_source, \
                                       cond_feat, cut_f)
        
        if self.training:
            # hidden signals shape as (batchsize=1, length)
            hidden = [x.squeeze(-1) for x in hidden]
            # sine for masking
            with torch.no_grad():
                sine_mask, uv, noise = self.m_sinemask(f0_upsamp)
                sine_mask = (sine_mask - noise).mean(axis=-1)
            # return 
            return [output.squeeze(-1), hidden, sine_mask]
        else:
            return output.squeeze(-1)
    
    
class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        # frame shift (number of points)
        self.frame_hops = [80, 40, 640]
        # frame length
        self.frame_lens = [320, 80, 1920]
        # FFT length
        self.fft_n = [512, 128, 2048]
        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating
        self.amp_floor = 0.00001
        # floor to determine the frames to be masked
        self.mask_power_threshold = 0.0000001
        # loss function
        self.loss = torch_nn.MSELoss()

        return

    def _stft(self, signal, fft_p, frame_shift, frame_len):
        """ wrapper of torch.stft
        Remember to use onesided=True, pad_mode="constant"
        Signal (batchsize, length)
        Output (batchsize, fft_p/2+1, frame_num, 2)
        """ 
        
        # to be compatible with different torch versions
        if torch.__version__.split('.')[1].isnumeric() and \
           int(torch.__version__.split('.')[1]) < 7:
            return torch.stft(
                signal, fft_p, frame_shift, frame_len, 
                window=self.win(frame_len, dtype=signal.dtype, 
                                device=signal.device), 
                onesided=True, pad_mode="constant")
        else:
            return torch.stft(
                signal, fft_p, frame_shift, frame_len, 
                window=self.win(frame_len, dtype=signal.dtype, 
                                device=signal.device), 
                onesided=True, pad_mode="constant", return_complex=False)
            
    
    def _amp(self, x):
        """  _amp(stft)
        x_stft: (batchsize, fft_p/2+1, frame_num, 2)
        output: (batchsize, fft_p/2+1, frame_num)
        output[x, y, z] = log(x_stft[x, y, z, 1]^2 + x_stft[x, y, z, 2]^2
                              + floor)
        """
        return torch.log(torch.norm(x, 2, -1).pow(2) + self.amp_floor)

    def _mask_stft(self, mask_signal, fft_p, frame_shift, frame_len):
        """
        calculate the masking weights from input mask signal (sine)
        """
        # power spectrum of the masking signal (sine signal)
        x_stft = self._stft(mask_signal, fft_p, frame_shift, frame_len)
        # x_pow (batchsize, fft_p/2+1, frame_num)
        x_stft_pow = torch.norm(x_stft, 2, -1).pow(2)
        
        # get the normalizing weight for each frame
        # x_flag (batchsize, frame_num)
        x_flag = x_stft_pow.mean(axis=1) > self.mask_power_threshold
        # x_stft_max (batchsize, frame_num)
        x_stft_max = x_stft_pow.max(axis=1)[0]
        x_stft_max[~x_flag] = 1.0
        # x_stft_weight (batchsize, frame_num)
        x_stft_weight = 1 / x_stft_max * x_flag
        
        # normalizing the mask
        # mask_normed (batchsize, fft_p/2+1, frame_num, 2)
        mask_normed = torch.ones_like(x_stft)
        # normalize the mask, so that maximum mask weight = 1
        # mask_normed[:, :, :, 0] is used to mask the real-part 
        # of an spectrum
        # mask_normed[:, :, :, 1] is used to mask the imaginary-part 
        # of an spectrum
        mask_normed[:, :, :, 0] = x_stft_pow * x_stft_weight.unsqueeze(1)
        mask_normed[:, :, :, 1] = mask_normed[:, :, :, 0]
        return mask_normed

    def stft_amp(self, signal, fft_p, frame_shift, frame_len, mask=None):
        """ compute STFT log amplitude
        signal: (batchsize, length)
        output: (batchsize, fft_p/2+1, frame_num)
        
        mask:  (batchsize, fft_p/2+1, frame_num, 2)
        """
        x_stft = self._stft(signal, fft_p, frame_shift, frame_len)
        
        if mask is None:
            x_sp_amp = self._amp(x_stft)
        else:
            # apply mask if necessary
            # mask[:, :, :, 0] is used to mask the real-part 
            # of an spectrum
            # mask[:, :, :, 1] is used to mask the imaginary-part 
            # of an spectrum
            x_sp_amp = self._amp(x_stft * mask)
        return x_sp_amp
            

    def compute(self, outputs, target):
        """ Loss().compute(output, target) should return
        the Loss in torch.tensor format
        Assume output and target as (batchsize=1, length)
        """       
        # generated signal
        output = outputs[0]
        # hidden signals from each filter block in harmonic branch
        hiddens = outputs[1]
        # sine mask signal
        sinemask = outputs[2]

        # convert from (batchsize=1, length, dim=1) to (1, length)
        if target.ndim == 3:
            target.squeeze_(-1)

        # compute loss over target and output
        loss = 0
        for frame_s, frame_l, fft_p in \
            zip(self.frame_hops, self.frame_lens, self.fft_n):
            
            # between generated signal and target
            gen_sp_amp = self.stft_amp(output, fft_p, frame_s, frame_l)
            tar_sp_amp = self.stft_amp(target, fft_p, frame_s, frame_l)
            loss += self.loss(gen_sp_amp, tar_sp_amp)
            
            # masked spectral loss between hidden signals & target
            with torch.no_grad():
                # produce stft of sine mask
                mask = self._mask_stft(sinemask, fft_p, frame_s, frame_l)
                # apply mask to target signal
                tar_sp_masked_amp = self.stft_amp(target, fft_p, \
                                                  frame_s, frame_l, mask)
            for hidden in hiddens:
                h_sp_masked_amp = self.stft_amp(hidden, fft_p, frame_s, \
                                                frame_l, mask)
                loss += self.loss(h_sp_masked_amp, tar_sp_masked_amp)
        # done
        return loss

    
if __name__ == "__main__":
    print("Definition of model")

    
