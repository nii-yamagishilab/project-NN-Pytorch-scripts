#!/usr/bin/env python
"""
util_frontend.py

Utilities for frontend feature extraction

It includes:
 LFCC: based on asvspoof.org baseline matlab code
 LFB: Linear filterbank feature
      Chen, T., Kumar, A., Nagarsheth, P., Sivaraman, G. & Khoury, E. 
      Generalization of Audio Deepfake Detection. in Proc. Odyssey 132-137 
      (2020). doi:10.21437/Odyssey.2020-19 

      According to the author's email:
      LFB = np.log(FilterBank(Amplitude(STFT(x))))
      There is no DCT. But it does have logarithm.

      Implemented based on LFCC API
 
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import sandbox.util_dsp as nii_dsp
import core_scripts.data_io.conf as nii_conf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

##################
## other utilities
##################
def stft_wrapper(x, fft_n, frame_shift, frame_length, window, 
                 pad_mode="constant", return_complex=False):
    """Due to the different signature of torch.stft, write a 
    wrapper to handle this

    input
    -----
      x: tensor, waveform, (batch, length)
      window: tensor, window coef, (frame_length, )
    
    output
    ------
      tensor: (batch, frame_num, bin_num, 2)
    
    """
    # there are better ways, but for convenience
    if torch.__version__.split('.')[1].isnumeric() and \
       int(torch.__version__.split('.')[1]) < 7:
        #  torch 1.6.*
        return torch.stft(x, fft_n, frame_shift, frame_length, 
                          window=window, onesided=True, pad_mode=pad_mode)
    else:
        #  torch > 1.7
        return torch.stft(x, fft_n, frame_shift, frame_length, 
                          window=window, onesided=True, pad_mode=pad_mode,
                          return_complex=return_complex)


def istft_wrapper(x, fft_n, frame_shift, frame_length, window, 
                 pad_mode="constant"):
    
    # there are better ways, but for convenience
    if torch.__version__.split('.')[1].isnumeric() and \
       int(torch.__version__.split('.')[1]) < 7:
        #  torch 1.6.*
        return torch.istft(x, fft_n, frame_shift, frame_length, 
                           window=window, onesided=True, pad_mode=pad_mode)
    else:
        #  torch > 1.7
        return torch.istft(x, fft_n, frame_shift, frame_length, 
                           window=window, onesided=True)




def trimf(x, params):
    """
    trimf: similar to Matlab definition
    https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle
    
    """
    if len(params) != 3:
        print("trimp requires params to be a list of 3 elements")
        sys.exit(1)
    a = params[0]
    b = params[1]
    c = params[2]
    if a > b or b > c:
        print("trimp(x, [a, b, c]) requires a<=b<=c")
        sys.exit(1)
    y = torch.zeros_like(x, dtype=nii_conf.d_dtype)
    if a < b:
        index = torch.logical_and(a < x, x < b)
        y[index] = (x[index] - a) / (b - a)
    if b < c:    
        index = torch.logical_and(b < x, x < c)              
        y[index] = (c - x[index]) / (c - b)
    y[x == b] = 1
    return y 
    
def delta(x):
    """ By default
    input
    -----
    x (batch, Length, dim)
    
    output
    ------
    output (batch, Length, dim)
    
    Delta is calculated along Length dimension
    """
    length = x.shape[1]
    output = torch.zeros_like(x)
    x_temp = torch_nn_func.pad(x.unsqueeze(1), (0, 0, 1, 1), 
                               'replicate').squeeze(1)
    output = -1 * x_temp[:, 0:length] + x_temp[:,2:]
    return output


def linear_fb(fn, sr, filter_num):
    """linear_fb(fn, sr, filter_num)
    create linear filter bank based on trim

    input
    -----
      fn: int, FFT points
      sr: int, sampling rate (Hz)
      filter_num: int, number of filters in filter-bank
    
    output
    ------
      fb: tensor, (fn//2+1, filter_num)

    Note that this filter bank is supposed to be used on 
    spectrum of dimension fn//2+1.

    See example in LFCC.
    """
    # build the triangle filter bank
    f = (sr / 2) * torch.linspace(0, 1, fn//2+1)
    filter_bands = torch.linspace(min(f), max(f), filter_num+2)
        
    filter_bank = torch.zeros([fn//2+1, filter_num])
    for idx in range(filter_num):
        filter_bank[:, idx] = trimf(
            f, [filter_bands[idx], 
                filter_bands[idx+1], 
                filter_bands[idx+2]])
    return filter_bank

#################
## LFCC front-end
#################

class LFCC(torch_nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy
    """
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=True, flag_for_LFB=False,
                 num_coef=None, min_freq=0, max_freq=1):
        """ Initialize LFCC
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          filter_num: int, number of filters in filter-bank

          with_energy: bool, (default False), whether replace 1st dim to energy
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default True), whether use delta and delta-delta
        
          for_LFB: bool (default False), reserved for LFB feature
          num_coef: int or None, number of coeffs to be taken from filter bank.
                    Note that this is only used for LFCC, i.e., for_LFB=False
                    When None, num_coef will be equal to filter_num
          min_freq: float (default 0), min_freq * sr // 2 will be the minimum 
                    frequency of extracted FFT spectrum
          max_freq: float (default 1), max_freq * sr // 2 will be the maximum 
                    frequency of extracted FFT spectrum
        """
        super(LFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        self.num_coef = num_coef

        # decide the range of frequency bins
        if min_freq >= 0 and min_freq < max_freq and max_freq <= 1:
            self.min_freq_bin = int(min_freq * (fn//2+1))
            self.max_freq_bin = int(max_freq * (fn//2+1))
            self.num_fft_bins = self.max_freq_bin - self.min_freq_bin 
        else:
            print("LFCC cannot work with min_freq {:f} and max_freq {:}".format(
                min_freq, max_freq))
            sys.exit(1)
        
        # build the triangle filter bank
        f = (sr / 2) * torch.linspace(min_freq, max_freq, self.num_fft_bins)
        filter_bands = torch.linspace(min(f), max(f), filter_num+2)
        
        filter_bank = torch.zeros([self.num_fft_bins, filter_num])
        for idx in range(filter_num):
            filter_bank[:, idx] = trimf(
                f, [filter_bands[idx], 
                    filter_bands[idx+1], 
                    filter_bands[idx+2]])
        self.lfcc_fb = torch_nn.Parameter(filter_bank, requires_grad=False)

        # DCT as a linear transformation layer
        self.l_dct = nii_dsp.LinearDCT(filter_num, 'dct', norm='ortho')

        # opts
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.flag_for_LFB = flag_for_LFB
        if self.num_coef is None:
            self.num_coef = filter_num

        # Add a buf to store window coefficients
        #  
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis 
        if self.with_emphasis:
            # to avoid side effect
            x_copy = torch.zeros_like(x) + x
            x_copy[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        else:
            x_copy = x
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        #x_stft = torch.stft(x_copy, self.fn, self.fs, self.fl, 
        #                    window=torch.hamming_window(self.fl).to(x.device), 
        #                    onesided=True, pad_mode="constant")
        x_stft = stft_wrapper(x_copy, self.fn, self.fs, self.fl,self.window_buf)

        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        
        if self.min_freq_bin > 0 or self.max_freq_bin < (self.fn//2+1):
            sp_amp = sp_amp[:, :, self.min_freq_bin:self.max_freq_bin]
        
        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.lfcc_fb) + 
                                 torch.finfo(torch.float32).eps)
        
        # DCT (if necessary, remove DCT)
        lfcc = self.l_dct(fb_feature) if not self.flag_for_LFB else fb_feature
        
        # Truncate the output of l_dct when necessary
        if not self.flag_for_LFB and self.num_coef != self.filter_num:
            lfcc = lfcc[:, :, :self.num_coef]
            

        # Add energy 
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2)+ 
                                 torch.finfo(torch.float32).eps)
            lfcc[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            lfcc_delta = delta(lfcc)
            lfcc_delta_delta = delta(lfcc_delta)
            lfcc_output = torch.cat((lfcc, lfcc_delta, lfcc_delta_delta), 2)
        else:
            lfcc_output = lfcc

        # done
        return lfcc_output

#################
## LFB front-end
#################

class LFB(LFCC):
    """ Linear filterbank feature
      Chen, T., Kumar, A., Nagarsheth, P., Sivaraman, G. & Khoury, E. 
      Generalization of Audio Deepfake Detection. in Proc. Odyssey 132-137 
      (2020). doi:10.21437/Odyssey.2020-19 
       
    """
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=False):
        """ Initialize LFB
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          filter_num: int, number of filters in filter-bank
          with_energy: bool, (default False), whether replace 1st dim to energy
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default True), whether use delta and delta-delta
        """
        super(LFB, self).__init__(fl, fs, fn, sr, filter_num, with_energy,
                                  with_emphasis, with_delta, flag_for_LFB=True)
        return
    
    def forward(self, x):
        """
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         lfb_output: tensor(batch, frame_num, dim_num)
        """
        return super(LFB, self).forward(x)


#################
## Spectrogram (FFT) front-end
#################

class Spectrogram(torch_nn.Module):
    """ Spectrogram front-end
    """
    def __init__(self, fl, fs, fn, sr, 
                 with_emphasis=True, with_delta=False, in_db=False):
        """ Initialize LFCC
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default False), whether use delta and delta-delta
          in_db: bool, (default False), use 20log10(amp)? if False, use amp
        """
        super(Spectrogram, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr

        # opts
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.in_db = in_db

        # buf to store window coefficients
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis 
        if self.with_emphasis:
            x[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        #x_stft = torch.stft(x, self.fn, self.fs, self.fl, 
        #                    window=torch.hamming_window(self.fl).to(x.device), 
        #                    onesided=True, pad_mode="constant")        
        x_stft = stft_wrapper(x, self.fn, self.fs, self.fl, self.window_buf)
        

        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        
        if self.in_db:
            sp_amp = torch.log10(sp_amp + torch.finfo(torch.float32).eps)

        # Add delta coefficients
        if self.with_delta:
            sp_delta = delta(sp_amp)
            sp_delta_delta = delta(sp_delta)
            sp_output = torch.cat((sp_amp, sp_delta, sp_delta_delta), 2)
        else:
            sp_output = sp_amp

        # done
        return sp_amp




#################
## MFCC front-end
#################
from core_scripts.data_io import dsp_tools

class MFCC(torch_nn.Module):
    """ Based on asvspoof.org baseline Matlab code.
    Difference: with_energy is added to set the first dimension as energy
    """
    def __init__(self, fl, fs, fn, sr, filter_num, 
                 with_energy=False, with_emphasis=True,
                 with_delta=True, flag_for_MelSpec=False,
                 num_coef=None, min_freq=0, max_freq=1):
        """ Initialize LFCC
        
        Para:
        -----
          fl: int, frame length, (number of waveform points)
          fs: int, frame shift, (number of waveform points)
          fn: int, FFT points
          sr: int, sampling rate (Hz)
          filter_num: int, number of filters in filter-bank

          with_energy: bool, (default False), whether replace 1st dim to energy
          with_emphasis: bool, (default True), whether pre-emphaze input wav
          with_delta: bool, (default True), whether use delta and delta-delta
        
          flag_for_MelSpec: bool (default False), reserved for LFB feature
          num_coef: int or None, number of coeffs to be taken from filter bank.
                    Note that this is only used for LFCC, i.e., for_LFB=False
                    When None, num_coef will be equal to filter_num
          min_freq: float (default 0), min_freq * sr // 2 will be the minimum 
                    frequency of extracted FFT spectrum
          max_freq: float (default 1), max_freq * sr // 2 will be the maximum 
                    frequency of extracted FFT spectrum
        """
        super(MFCC, self).__init__()
        self.fl = fl
        self.fs = fs
        self.fn = fn
        self.sr = sr
        self.filter_num = filter_num
        self.num_coef = num_coef

        # decide the range of frequency bins
        if min_freq >= 0 and min_freq < max_freq and max_freq <= 1:
            pass
        else:
            print("MFCC cannot work with min_freq {:f} and max_freq {:}".format(
                min_freq, max_freq))
            sys.exit(1)
        
        # opts
        self.with_energy = with_energy
        self.with_emphasis = with_emphasis
        self.with_delta = with_delta
        self.flag_for_MelSpec = flag_for_MelSpec
        if self.num_coef is None:
            self.num_coef = filter_num
            
        # get filter bank
        tmp_config = dsp_tools.Melspec(sr, fl, fs, fn, filter_num, 
                                       sr/2*min_freq, sr/2*max_freq)
        filter_bank = torch.tensor(tmp_config.melfb.T, dtype=nii_conf.d_dtype)
        self.mel_fb = torch_nn.Parameter(filter_bank, requires_grad=False)

        # DCT as a linear transformation layer
        if not self.flag_for_MelSpec:
            self.l_dct = nii_dsp.LinearDCT(filter_num, 'dct', norm='ortho')
        else:
            self.l_dct = None


        # Add a buf to store window coefficients
        #  
        self.window_buf = None
        return
    
    def forward(self, x):
        """
        
        input:
        ------
         x: tensor(batch, length), where length is waveform length
        
        output:
        -------
         lfcc_output: tensor(batch, frame_num, dim_num)
        """
        # pre-emphsis 
        if self.with_emphasis:
            # to avoid side effect
            x_copy = torch.zeros_like(x) + x
            x_copy[:, 1:] = x[:, 1:]  - 0.97 * x[:, 0:-1]
        else:
            x_copy = x
        
        if self.window_buf is None:
            self.window_buf = torch.hamming_window(self.fl).to(x.device)

        # STFT
        x_stft = stft_wrapper(x_copy, 
                              self.fn, self.fs, self.fl, self.window_buf)

        # amplitude
        sp_amp = torch.norm(x_stft, 2, -1).pow(2).permute(0, 2, 1).contiguous()
        
        # filter bank
        fb_feature = torch.log10(torch.matmul(sp_amp, self.mel_fb) + 
                                 torch.finfo(torch.float32).eps)
        
        # DCT (if necessary, remove DCT)
        if not self.flag_for_MelSpec:
            output = self.l_dct(fb_feature) 
        else:
            output = fb_feature
        
        # Truncate the output of l_dct when necessary
        if not self.flag_for_MelSpec and self.num_coef != self.filter_num:
            output = output[:, :, :self.num_coef]

        # Add energy 
        if self.with_energy:
            power_spec = sp_amp / self.fn
            energy = torch.log10(power_spec.sum(axis=2)+ 
                                 torch.finfo(torch.float32).eps)
            output[:, :, 0] = energy

        # Add delta coefficients
        if self.with_delta:
            output_delta = delta(output)
            output_delta_delta = delta(output_delta)
            output = torch.cat((output, output_delta, output_delta_delta), 2)
        else:
            pass

        # done
        return output


#######################
# spectrum substraction
#######################
def spectral_substraction(input_wav, noise_wav, ratio = 0.1,
                          fft_n = 512, frame_shift = 256, frame_length = 512):
    """
    output = spectrum_substraction(input_wav, noise_wav)
        
    input
    -----
      input_wav: tensor, (batch, length1, 1)
      noise_wav: tensor, (batch, length2, 1)
      ratio: float, default 0.1, ratio to be multiplied with noise spectrum
      fft_n: int, default 1024, fft length
      frame_shift: int, default 256, frame shift
      frame_length: int, default 512, frame_shift
    
    output
    ------
      output: tensor, de-noised waveform
    
    Note: this function do spectral_substraction. 
    noise_wav does not need to have the same length as input_wav.
    spectra amplitude of noise_wav will be averaged and subtracted from 
      input_wav stft spectra amplitude
    """    
    window = torch.hamming_window(frame_length).to(input_wav.device)
    
    # stft
    input_spec = stft_wrapper(
        input_wav.squeeze(-1), fft_n, frame_shift, frame_length, window, 
        return_complex=True)
    
    noise_spec = stft_wrapper(
        noise_wav.squeeze(-1), fft_n, frame_shift, frame_length, window, 
        return_complex=True)
        
    # input wave spectral amplitude and phase (batch, fft_n//2+1, length, )
    input_spec_abs = torch.abs(input_spec)
    input_spec_pha = torch.angle(input_spec)
    
    # noise spectral, averaged
    noise_spec_abs = torch.abs(noise_spec).mean(axis=-1).unsqueeze(-1)
    
    # spectral subtraction
    denoi_spec_abs = input_spec_abs - noise_spec_abs * ratio
    denoi_spec_abs = torch.clamp(denoi_spec_abs, min=0.00000001)
        
    # recover waveform
    input_spec = torch.complex(
        denoi_spec_abs * torch.cos(input_spec_pha),
        denoi_spec_abs * torch.sin(input_spec_pha),
    )
        
    output = istft_wrapper(
        input_spec, fft_n, frame_shift, frame_length, window)

    # adjust waveform length
    length = min([input_wav.shape[1], output.shape[1]])
    output_new = torch.zeros_like(input_wav)
    output_new[:, 0:length, 0] = output[:, 0:length]
    return output_new


if __name__ == "__main__":
    print("Definition of front-end for Anti-spoofing")
