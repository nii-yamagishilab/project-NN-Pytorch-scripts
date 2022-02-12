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

import core_scripts.other_tools.debug as nii_debug
import sandbox.block_nn as nii_nn
import sandbox.block_nsf as nii_nsf


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

    
#####
## Model definition
##### 
class Conv1dNoPermute(torch_nn.Conv1d):
    """
    """
    def __init__(self, input_dim, output_dim, dilation_s, kernel_s, 
                 causal = False, stride = 1, groups=1, bias=True, tanh=True):
        super(Conv1dNoPermute, self).__init__(
            input_dim, output_dim, kernel_s, stride=stride,
            padding = dilation_s * (kernel_s - 1) if causal \
            else dilation_s * (kernel_s - 1) // 2, 
            dilation = dilation_s, groups=groups, bias=bias)

        self.l_ac = torch_nn.Tanh() if tanh else torch_nn.Identity()
        return

    def forward(self, input_data):
        data = input_data[0]
        cond = input_data[1]
        out = self.l_ac(
            super(Conv1dNoPermute, self).forward(data)[:, :, :data.shape[-1]])
        return [data + cond + out, cond]


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
        self.l_ff_p = torch_nn.Sequential(
            torch_nn.Linear(signal_size, hidden_size, bias=False),
            torch_nn.Tanh())

        # dilated conv layers
        tmp = [Conv1dNoPermute(hidden_size, hidden_size, x, 
                               kernel_size, causal=True, bias=False) \
               for x in self.dilation_size]
        self.l_convs = torch_nn.Sequential(*tmp)
        
        # ff layer to de-expand dimension
        self.l_ff_f = torch_nn.Sequential(
            torch_nn.Linear(hidden_size, hidden_size//4, bias=False),
            torch_nn.Tanh(),
            torch_nn.Linear(hidden_size//4, signal_size, bias=False),
            torch_nn.Tanh())

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

    def forward(self, input_data):
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
        signal, context = input_data[0], input_data[1]
        # expand dimension
        tmp_hidden = self.l_ff_p(signal)
        
        # loop over dilated convs
        # output of a d-conv is input + context + d-conv(input)
        tmp_hidden = self.l_convs(
            [tmp_hidden.permute(0, 2, 1),
             context.permute(0, 2, 1)])[0].permute(0, 2, 1)
            
        # to be consistent with legacy configuration in CURRENNT
        tmp_hidden = tmp_hidden * self.scale
        
        # compress the dimesion and skip-add
        output_signal = self.l_ff_f(tmp_hidden) + signal
        return [output_signal, context]


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

        # bi-LSTM
        self.l_blstm = nii_nn.BLSTMLayer(input_dim, self.blstm_s)
        self.l_conv1d = nii_nn.Conv1dKeepLength(
            self.blstm_s, output_dim, 1, self.cnn_kernel_s)

        self.l_upsamp = nii_nn.UpSampleLayer(
            self.output_dim, self.up_sample, True)
        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = nii_nn.UpSampleLayer(1, self.up_sample, False)

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
class SourceModuleMusicNSF(torch_nn.Module):
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

    Sine_source, noise_source = SourceModuleMusicNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """
    def __init__(self, sampling_rate, harmonic_num=0, sine_amp=0.1, 
                 add_noise_std=0.003, voiced_threshod=0):
        super(SourceModuleMusicNSF, self).__init__()
        
        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = nii_nsf.SineGen(
            sampling_rate, harmonic_num, sine_amp, 
            add_noise_std, voiced_threshod)

        # to merge source harmonics into a single excitation
        self.l_linear = torch_nn.Linear(harmonic_num+1, 1)
        self.l_tanh = torch_nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleMusicNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        #  sine fundamental component and harmonic overtones
        sine_wavs, uv, _ = self.l_sin_gen(x)
        #  merge into a single excitation
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv
        
        
# For Filter module
class FilterModuleMusicNSF(torch_nn.Module):
    """ Filter for Hn-NSF
    FilterModuleMusicNSF(signal_size, hidden_size, fir_coef,
                      block_num = 5,
                      kernel_size = 3, conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    fir_coef: list of FIR filter coeffs,
              (low_pass_1, low_pass_2, high_pass_1, high_pass_2)
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block


    output = FilterModuleMusicNSF(harmonic_source,noise_source,uv,context)
    harmonic_source (batchsize, length, dim=1)
    noise_source  (batchsize, length, dim=1)
    context (batchsize, length, dim)
    uv (batchsize, length, dim)

    output: (batchsize, length, dim=1)    
    """
    def __init__(self, signal_size, hidden_size, \
                 block_num = 5, kernel_size = 3, conv_num_in_block = 10):
        super(FilterModuleMusicNSF, self).__init__()        
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block

        # filter blocks for harmonic branch
        tmp = [NeuralFilterBlock(
            signal_size, hidden_size, kernel_size, conv_num_in_block) \
               for x in range(self.block_num)]
        self.l_har_blocks = torch_nn.Sequential(*tmp)


    def forward(self, har_component, noi_component, condition_feat, uv):
        """
        """
        # harmonic component
        #for l_har_block in self.l_har_blocks:
        #    har_component = l_har_block(har_component, condition_feat)
        #output = har_component
        output = self.l_har_blocks([har_component, condition_feat])[0]
        return output        
        

## FOR MODEL
class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        ######
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        ######

        # configurations
        self.sine_amp = 0.1
        self.noise_std = 0.001
        
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.hidden_dim = 64

        self.upsamp_rate = prj_conf.input_reso[0]
        self.sampling_rate = prj_conf.wav_samp_rate

        self.cnn_kernel_size = 3
        self.filter_block_num = 5
        self.cnn_num_in_block = 10
        self.harmonic_num = 16
        
        # the three modules
        self.m_condition = CondModule(self.input_dim, \
                                      self.hidden_dim, \
                                      self.upsamp_rate, \
                                      cnn_kernel_s = self.cnn_kernel_size)

        #self.m_source = SourceModuleMusicNSF(self.sampling_rate, 
        #                                     self.harmonic_num, 
        #                                     self.sine_amp, 
        #                                     self.noise_std)
        
        self.m_filter = FilterModuleMusicNSF(self.output_dim, 
                                             self.hidden_dim,\
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

        # normalize the data
        feat = self.normalize_input(x)

        # condition module
        #  place_holder is originally the up-sampled F0
        #  it is not used for noise-excitation model
        #  but it has the same shape as the upsampled souce signal
        #  it can help to create the noise_source below
        cond_feat, place_holder = self.m_condition(feat, x[:, :, -1:])

        with torch.no_grad():
            noise_source = torch.randn_like(place_holder) * self.noise_std / 3

        # source module
        #har_source, noi_source, uv = self.m_source(f0_upsamped)

        # filter module (including FIR filtering)
        output = self.m_filter(noise_source, None, cond_feat, None)

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
        self.fft_n = [4096, 4096, 4096]
        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating
        self.amp_floor = 0.00001
        # loss
        self.loss1 = torch_nn.MSELoss()
        self.loss2 = torch_nn.MSELoss()
        self.loss3 = torch_nn.MSELoss()
        self.loss = [self.loss1, self.loss2, self.loss3]
        #self.loss = torch_nn.MSELoss()
        
    def compute(self, output_orig, target_orig):
        """ Loss().compute(output, target) should return
        the Loss in torch.tensor format
        Assume output and target as (batchsize=1, length)
        """
        # convert from (batchsize=1, length, dim=1) to (1, length)
        if output_orig.ndim == 3:
            output = output_orig.squeeze(-1)
        else:
            output = output_orig
        if target_orig.ndim == 3:
            target = target_orig.squeeze(-1)
        else:
            target = target_orig
        
        # compute loss
        loss = 0
        for frame_shift, frame_len, fft_p, loss_f in \
            zip(self.frame_hops, self.frame_lens, self.fft_n, self.loss):
            x_stft = torch.stft(output, fft_p, frame_shift, frame_len, \
                                window=self.win(frame_len, \
                                                device=output_orig.device), 
                                onesided=True,
                                pad_mode="constant")
            y_stft = torch.stft(target, fft_p, frame_shift, frame_len, \
                                window=self.win(frame_len, 
                                                device=output_orig.device), 
                                onesided=True,
                                pad_mode="constant")
            x_sp_amp = torch.log(torch.norm(x_stft, 2, -1).pow(2) + \
                                 self.amp_floor)
            y_sp_amp = torch.log(torch.norm(y_stft, 2, -1).pow(2) + \
                                 self.amp_floor)
            loss += loss_f(x_sp_amp, y_sp_amp)
        return loss
    
if __name__ == "__main__":
    print("Definition of model")

    
