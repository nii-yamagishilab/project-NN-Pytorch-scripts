#!/usr/bin/env python
"""
model.py for harmonic-plus-noise NSF
version: 1
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
        # https://github.com/pytorch/pytorch/issues/1333
        x = torch_nn_func.pad(data.permute(0, 2, 1).unsqueeze(2), \
                              (self.pad_le, self.pad_ri, 0, 0)).squeeze(2)
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

# 
# FIR filter layer
class TimeInvFIRFilter(Conv1dKeepLength):
    """ Wrapper to define a FIR filter over Conv1d
        Note: FIR Filtering is conducted on each dimension (channel)
        independently: groups=channel_num in conv1d
    """                                                                   
    def __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False):
        """ __init__(self, feature_dim, filter_coef, 
                 causal=True, flag_train=False)
        feature_dim: dimension of input data
        filter_coef: 1-D tensor of filter coefficients
        causal: FIR is causal or not (default: true)
        flag_train: whether train the filter coefficients (default: false)

        Input data: (batchsize, length, feature_dim)
        Output data: (batchsize, length, feature_dim)
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
                                                 requires_grad=flag_train)
        else:
            print("TimeInvFIRFilter expects filter_coef to be 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)

    def forward(self, data):                                              
        return super(TimeInvFIRFilter, self).forward(data)    

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
        tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
        output_signal = tmp_hidden + signal
        return output_signal
    
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
            # This will not change F0 of sine because (x-1) * 2*pi = x *2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - 
                                tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            
            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) \
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

#####
## Model definition
## 

## For condition module only provide Spectral feature to Filter block
class CondModule(torch_nn.Module):
    """ Conditiona module

    Upsample and transform input features
    CondModule(input_dimension, output_dimension, up_sample_rate,
               blstm_dimension = 64, cnn_kernel_size = 3)
    
    Spec, F0 = CondModule(features, F0)
    Both input features should be frame-level features

    If x doesn't contain F0, just ignore the returned F0
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
        self.l_upsamp = UpSampleLayer(self.output_dim, self.up_sample, 
                                      True)
        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = UpSampleLayer(1, self.up_sample, False)

    def forward(self, feature, f0):
        """ spec, f0 = forward(self, feature, f0)
        feature: (batchsize, length, dim)
        f0: (batchsize, length, dim=1), which should be F0 at frame-level
        
        spec: (batchsize, length, self.output_dim), at wave-level
        f0: (batchsize, length, 1), at wave-level
        """ 
        spec = self.l_upsamp(self.l_conv1d(self.l_blstm(feature)))
        f0 = self.l_upsamp_F0(f0)
        return spec, f0

# For source module
class SourceModuleHnNSF(torch_nn.Module):
    """ SourceModule for hn-nsf 
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1, 
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)

    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, 
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleHnNSF, self).__init__()
        
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(sampling_rate, harmonic_num,
                                 sine_amp, add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch_nn.Linear(harmonic_num+1, 1)
        self.l_tanh = torch_nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
        
        
# For Filter module
class FilterModuleHnNSF(torch_nn.Module):
    """ Filter for Hn-NSF
    FilterModuleHnNSF(signal_size, hidden_size, fir_coef,
                      block_num = 5,
                      kernel_size = 3, conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    fir_coef: list of FIR filter coeffs,
              (low_pass_1, low_pass_2, high_pass_1, high_pass_2)
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block


    output = FilterModuleHnNSF(harmonic_source, noise_source, uv, context)
    harmonic_source (batchsize, length, dim=1)
    noise_source  (batchsize, length, dim=1)
    context (batchsize, length, dim)
    uv (batchsize, length, dim)

    output: (batchsize, length, dim=1)    
    """
    def __init__(self, signal_size, hidden_size, filter_coef, \
                 block_num = 5, kernel_size = 3, conv_num_in_block = 10):
        super(FilterModuleHnNSF, self).__init__()        
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block
        self.filter_coef = filter_coef

        # filter blocks for harmonic branch
        tmp = [NeuralFilterBlock(signal_size, hidden_size, \
                                 kernel_size, conv_num_in_block) \
               for x in range(self.block_num)]
        self.l_har_blocks = torch_nn.ModuleList(tmp)

        # filter blocks for noise branch (only one block, 5 sub-blocks)
        tmp = [NeuralFilterBlock(signal_size, hidden_size, \
                                 kernel_size, conv_num_in_block // 2) \
               for x in range(1)]
        self.l_noi_blocks = torch_nn.ModuleList(tmp)

        # FIR filter groups
        # lp_v: filter for voiced region, harmonic component
        # lp_u: filter for unvoiced region, harmonic component
        # hp_v: filter for voiced region, noise component
        # hp_u: filter for unvoiced region, noise component
        self.l_fir_lp_v = TimeInvFIRFilter(signal_size, filter_coef[0])
        self.l_fir_lp_u = TimeInvFIRFilter(signal_size, filter_coef[1])
        self.l_fir_hp_v = TimeInvFIRFilter(signal_size, filter_coef[2])
        self.l_fir_hp_u = TimeInvFIRFilter(signal_size, filter_coef[3])
        

    def forward(self, har_component, noi_component, condition_feat, uv):
        """
        """
        # harmonic component
        for l_har_block in self.l_har_blocks:
            har_component = l_har_block(har_component, condition_feat)
        # noise componebt
        for l_noi_block in self.l_noi_blocks:
            noi_component = l_noi_block(noi_component, condition_feat)
        
        # harmonic + noise in time-domain
        # assume uv is {0, 1}, produce a weight vector for voiced/unvoiced
        # sigmoid is used to avoid {0, 1}, and uv is scaled to {-5, 5}
        w_voi = torch.sigmoid((uv - 0.5) * 10)
        w_unv = 1.0 - w_voi
        
        har_v = self.l_fir_lp_v(har_component)
        har_u = self.l_fir_lp_u(har_component)
        noi_v = self.l_fir_hp_v(noi_component)
        noi_u = self.l_fir_hp_u(noi_component)
        
        output = (har_v + noi_v) * w_voi + (har_u + noi_u) * w_unv
        return output
        
        

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
        self.cnn_kernel_size = 3
        # number of filter blocks (for harmonic branch)
        # noise branch only uses 1 block
        self.filter_block_num = 5
        # number of dilated CNN in each filter block
        self.cnn_num_in_block = 10
        # number of harmonic overtone in source
        self.harmonic_num = 7

        # fixed filter coefficients
        # computed using PM algorithm 
        # (tool: http://t-filter.engineerjs.com)
        #
        # low-pass for harmonic-component in voiced region
        #  16kHz, pass-band 0-5K, gain 1, ripple 5dB, 
        #         stop-band 7-8k, gain 0, ripple -40dB)
        lp_v = [0.08538414199291068, 0.04920229475534168,
                -0.1470178606967731, 0.24737764593887432,
                0.7103067853166558, 0.24737764593887432,
                -0.1470178606967731, 0.04920229475534168,
                0.08538414199291068]

        # low-pass for harmonic-copmonent in unvoiced region
        #  16kHz, pass-band 0-1K, gain 1, ripple 5dB, 
        #         stop-band 3-8k, gain 0, ripple -40dB)
        lp_u = [0.00936455546502, 0.0416254862901, 0.0878313219556,
                0.146086321198, 0.192602581136, 0.211221591449,
                0.192602581136, 0.146086321198, 0.0878313219556,
                0.0416254862901, 0.00936455546502]

        #
        # high-pass for noise-component in voiced region
        #  16kHz, pass-band 7-8K, gain 1, ripple 5dB, 
        #         stop-band 0-5k, gain 0, ripple -40dB)        
        hp_v = [-0.00936455546502148, 0.04162548629009957, 
                -0.08783132195564508, 0.1460863211980122,
                -0.19260258113649556, 0.21122159144894015,
                -0.19260258113649556, 0.1460863211980122,
                -0.08783132195564508, 0.04162548629009957,
                -0.00936455546502148]

        #
        # high-pass for noise-component in unvoiced region
        #  16kHz, pass-band 3-8K, gain 1, ripple 5dB, 
        #         stop-band 0-1k, gain 0, ripple -40dB)                
        hp_u = [0.0853841419929, -0.0492022947553, -0.147017860697, 
                -0.247377645939, 0.710306785317, -0.247377645939, 
                -0.147017860697, -0.0492022947553, 0.0853841419929]
        
        self.fir_filters = [torch.tensor(lp_v), torch.tensor(lp_u),
                            torch.tensor(hp_v), torch.tensor(hp_u)]
        
        
        # the three modules
        self.m_condition = CondModule(self.input_dim, self.hidden_dim, \
                                      self.upsamp_rate, \
                                      cnn_kernel_s = self.cnn_kernel_size)

        self.m_source = SourceModuleHnNSF(self.sampling_rate, 
                                          self.harmonic_num, 
                                          self.sine_amp, self.noise_std)
        
        self.m_filter = FilterModuleHnNSF(self.output_dim, 
                                          self.hidden_dim,\
                                          self.fir_filters,
                                          self.filter_block_num, \
                                          self.cnn_kernel_size, \
                                          self.cnn_num_in_block)
        # done
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
        # normalize the data
        feat = self.normalize_input(x)

        # condition module
        # features_for_filter_block, up-sampled F0
        cond_feat, f0_upsamped = self.m_condition(feat, f0)
        
        # source module
        # harmonic-source, noise-source (for noise branch), uv flag
        har_source, noi_source, uv = self.m_source(f0_upsamped)
        
        # filter module (including FIR filtering)
        # output signal
        output = self.m_filter(har_source, noi_source, cond_feat, uv)
        # output
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
        # loss
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
        for frame_shift, frame_len, fft_p in \
            zip(self.frame_hops, self.frame_lens, self.fft_n):
            x_stft = self._stft(output, fft_p, frame_shift, frame_len)
            y_stft = self._stft(target, fft_p, frame_shift, frame_len)
            x_sp_amp = self._amp(x_stft)
            y_sp_amp = self._amp(y_stft)
            loss += self.loss(x_sp_amp, y_sp_amp)
        return loss

    
if __name__ == "__main__":
    print("Definition of model")

    
