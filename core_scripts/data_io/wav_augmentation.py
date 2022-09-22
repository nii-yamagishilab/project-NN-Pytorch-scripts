#!/usr/bin/env python
"""
Functions for waveform augmentation.

Note that 
1. functions here are based on numpy, and they are intended to be used
   before data are converted into torch tensors. 

   data on disk -> DataSet.__getitem__()  -----> Collate  ---->  Pytorch model
                             numpy.tensor           torch.tensor

   These functions don't work on pytorch tensors

2. RawBoost functions are based on those by H.Tak and M.Todisco

   See code here
   https://github.com/TakHemlata/RawBoost-antispoofing
   
   Hemlata Tak, Madhu R Kamble, Jose Patino, Massimiliano Todisco, and 
   Nicholas W D Evans. RawBoost: A Raw Data Boosting and Augmentation Method 
   Applied to Automatic Speaker Verification Anti-Spoofing. Proc. ICASSP. 2022

"""

from __future__ import absolute_import

import os
import sys
import copy

import numpy as np
from scipy import signal
from pathlib import Path

try:
    from pydub import AudioSegment
except ModuleNotFoundError:
    pass

import core_scripts.data_io.wav_tools as nii_wav_tools

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

################
# Tool 
################
def unify_length_shape(data, target_data):
    """ output = unify_length_shape(data, target_data)
    
    input
    -----
      data: np.array, either (L), or (L, 1)
      target_data: np.array, either (L) or (L, 1)

    output
    ------
      output: np.array that has same shape as target_data
    """
    output_buf = np.zeros_like(target_data)
    min_len = min([data.shape[0], target_data.shape[0]])
    if data.ndim == target_data.ndim:
        output_buf[:min_len] = data[:min_len]
    elif data.ndim == 1:
        output_buf[:min_len, 0] = data[:min_len]
    elif data.ndim == 2:
        output_buf[:min_len] = data[:min_len, 0]
    else:
        print("Implementation error in unify_length_shape")
        sys.exit(1)
    return output_buf

################
# Time domain 
################

def wav_rand_sil_trim(wav, 
                      sr, 
                      random_trim_sil = False, 
                      random_trim_nosil = False):
    """ output = wav_rand_sil_trim(
                      wav, sr,
                      random_trim_sil = False, 
                      random_trim_nosil = False)
    
    Randomly trim the leading and ending silence.
    
    input
    -----
      wav: np.array, (length, 1)
      sr: int, waveform sampling rate
      random_trim_sil: bool, randomly trim silence (default False)
      random_trim_nosil: bool, randomly trim no-silence segments
      
    output
    ------
      output:  np.array, (length, 1)
      start_idx:  int, starting time index in the input wav
      end_idx: int, ending time index in the input wav
           output <- wav[start_idx:end_idx]
    """
    # original length
    orig_len = wav.shape[0]
    
    # frame shift for silence detection, fixed here
    fs=80
    
    # get the time flag for silence region
    _, _, frame_tag = nii_wav_tools.silence_handler_wrapper(
        wav, sr, fs=fs, flag_only_startend_sil = True)

    # get the ending time of the leading silence 
    # get the starting time of the trailing silence
    #
    if len(np.flatnonzero(frame_tag)):
        # start and end position
        start_nonsil = np.flatnonzero(frame_tag)[0] * fs
        end_nonsil = np.flatnonzero(frame_tag)[-1] * fs
    else:
        # no silence, use the original entire data
        start_nonsil = 0
        end_nonsil = wav.shape[0]

    # if further randomly trim, 
    if random_trim_sil:
        prob = np.random.rand()
        start_nosil_new = int(start_nonsil * prob)
        end_nosil_new = int((orig_len - end_nonsil)*prob) + end_nonsil
    else:
        start_nosil_new = start_nonsil
        end_nosil_new = end_nonsil

    # get the non-silence region
    if start_nosil_new < end_nosil_new and start_nosil_new > 0:
        input_new = wav[start_nosil_new:end_nosil_new]
    else:
        input_new = wav
    
    return input_new, start_nosil_new, end_nosil_new


def wav_time_mask(input_data, wav_samp_rate):
    """ output = wav_time_mask(input_data, wav_samp_rate)
    
    Apply time mask and zero-out segments
    
    input
    -----
      input_data: np.array, (length, 1)
      wav_samp_rate: int, waveform sampling rate
    
    output
    ------
      output:  np.array, (length, 1)
    """
    # choose the codec
    seg_width = int(np.random.rand() * 0.2 * wav_samp_rate)
    start_idx = int(np.random.rand() * (input_data.shape[0] - seg_width))
    if start_idx < 0:
        start_idx = 0
    if (start_idx + seg_width) > input_data.shape[0]:
        seg_width = input_data.shape[0] - start_idx
    tmp = np.ones_like(input_data)
    tmp[start_idx:start_idx+seg_width] = 0
    return input_data * tmp


    
def batch_siltrim_for_multiview(input_data_batch, wav_samp_rate,
                                random_trim_sil=False, 
                                random_trim_nosil=False):
    """ output = batch_trim(input_data, wav_samp_rate)
    
    For multi-view data, trim silence
    
    input
    -----
      input_data: list of np.array, (length, 1)
      wav_samp_rate: int, waveform sampling rate
    
    output
    ------
      output:  list of np.array, (length, 1)
    """

    # output buffer
    output_data_batch = []
    
    # original length
    orig_len = input_data_batch[0].shape[0]

    # get the starting and ending of non-silence region
    #  (computed based on the first wave in the list)
    _, start_time, end_time = wav_rand_sil_trim(
        input_data_batch[0], wav_samp_rate, 
        random_trim_sil, random_trim_nosil)
    
    # do trimming on all waveforms in the input list
    if start_time < end_time and start_time > 0:
        for data in input_data_batch:
            output_data_batch.append(data[start_time:end_time])
    else:
        for data in input_data_batch:
            output_data_batch.append(data)
    return output_data_batch


def batch_pad_for_multiview(input_data_batch_, wav_samp_rate, length,
                            random_trim_nosil=False, repeat_pad=False):
    """ output = batch_pad_for_multiview(
          input_data_batch, wav_samp_rate, length, random_trim_nosil=False)
    
    If input_data_batch is a single trial, trim it to a fixed length
    For multi-view data, trim all the trials to a fixed length, using the same
    random start and end

    input
    -----
      input_data: list of np.array, (length, 1)
      wav_samp_rate: int, waveform sampling rate
    
    output
    ------
      output:  list of np.array, (length, 1)
    """

    # unify the length of input data before further processing
    def _ad_length(x, length, repeat_pad):
        # adjust the length of the input x
        if length > x.shape[0]:
            if repeat_pad:
                rt = int(length / x.shape[0]) + 1
                tmp = np.tile(x, (rt, 1))[0:length]
            else:
                tmp = np.zeros([length, 1])
                tmp[0:x.shape[0]] = x
        else:
            tmp = x[0:length]
        return tmp
    # use the first data in the list
    firstlen = input_data_batch_[0].shape[0]
    input_data_batch = [_ad_length(x, firstlen, repeat_pad) \
                        for x in input_data_batch_]

    # 
    new_len = input_data_batch[0].shape[0]    
    if repeat_pad is False:
        
        # if we simply trim longer sentence but not pad shorter sentence
        if new_len < length:
            start_len = 0
            end_len = new_len

        elif random_trim_nosil:
            start_len = int(np.random.rand() * (new_len - length))
            end_len = start_len + length

        else:
            start_len = 0
            end_len = length
        input_data_batch_ = input_data_batch

    else:
        
        if new_len < length:
            start_len = 0
            end_len = length
            rt = int(length / new_len) + 1
            # repeat multiple times
            input_data_batch_ = [np.tile(x, (rt, 1)) for x in input_data_batch]
        elif random_trim_nosil:
            start_len = int(np.random.rand() * (new_len - length))
            end_len = start_len + length
            input_data_batch_ = input_data_batch
        else:
            start_len = 0
            end_len = length
            input_data_batch_ = input_data_batch

    output_data_batch = [x[start_len:end_len] for x in input_data_batch_]
    return output_data_batch



##################
# Frequency domain
##################


def wav_freq_mask_fixed(input_data, wav_samp_rate, start_b, end_b):
    """ output = wav_freq_mask_fixed(input_data, wav_samp_rate, start_b, end_b)
    
    Mask the frequency range, fixed 
    
    input
    -----
      input_data: np.array, (length, 1)
      wav_samp_rate: int, waveform sampling rate
      start_b: float
      end_b: float

    output
    ------
      output:  np.array, (length, 1)
    """
    # order of the filder, fixed to be 10
    # change it to a random number later
    filter_order = 10
    
    if start_b < 0.01:
        sos = signal.butter(filter_order, end_b, 'highpass', output='sos')
    elif end_b > 0.99:
        sos = signal.butter(filter_order, start_b, 'lowpass',output='sos')
    else:
        sos = signal.butter(
            filter_order, [start_b, end_b], 'bandstop', output='sos')
        
    filtered = signal.sosfilt(sos, input_data[:, 0])

    # change dimension
    output = np.expand_dims(filtered, axis=1)

    return output


def wav_freq_mask(input_data, wav_samp_rate):
    """ output = wav_freq_mask(input_data, wav_samp_rate)
    
    Randomly mask the signal in frequency domain
    
    input
    -----
      input_data: np.array, (length, 1)
      wav_samp_rate: int, waveform sampling rate
    
    output
    ------
      output:  np.array, (length, 1)
    """
    # order of the filder, fixed to be 10
    # change it to a random number later
    filter_order = 10

    # maximum value of the bandwidth for frequency masking
    max_band_witdh = 0.2

    # actual band_w
    band_w = np.random.rand() * max_band_witdh
    
    if band_w < 0.05:
        # if the bandwidth is too small, do no masking
        output = input_data
    else:
        # start
        start_b = np.random.rand() * (1 - band_w)
        # end
        end_b = start_b + band_w
        output = wav_freq_mask_fixed(input_data, wav_samp_rate, start_b, end_b)

    return output


##################
# Compression codec
##################
def wav_codec(input_dat, wav_samp_rate):
    """ A wrapper to use pyDub and ffmpeg
    
    This requires pyDub and ffmpeg installed. 
    """
    tmpdir = '/tmp/xwtemp'
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    randomname = "{:s}/{:010d}".format(tmpdir, np.random.randint(100000))
    while os.path.isfile(randomname + '.empty'):
        randomname = "{:s}/{:010d}".format(tmpdir, np.random.randint(100000))
    Path(randomname + '.empty').touch()
    
    # write to file (16bit PCM)
    nii_wav_tools.waveFloatToPCMFile(input_dat[:, 0], randomname + '.wav')
    data = AudioSegment.from_wav(randomname + '.wav')

    # choose the codec
    rand_codec = np.random.randint(2)
    if rand_codec == 0:
        # mp3
        rand_bitrate = np.random.randint(6)
        if rand_bitrate == 0:
            rate = '16k'
        elif rand_bitrate == 1:
            rate = '32k'
        elif rand_bitrate == 2:
            rate = '64k'
        elif rand_bitrate == 3:
            rate = '128k'
        elif rand_bitrate == 4:
            rate = '256k'
        else:
            rate = '320k'
        data.export(randomname + '.tmp', format='mp3', 
                    codec='libmp3lame', bitrate=rate)
        data_codec = AudioSegment.from_mp3(randomname + '.tmp')
        output_dat = np.array(data_codec.get_array_of_samples() / 
                              np.power(2, 16-1))

        # clean
        #os.remove(randomname + '.wav')
        try:
            os.remove(randomname + '.tmp')
        except FileNotFoundError:
            pass
        #os.remove(randomname + '.empty')

    else:
        # opus ogg
        rand_bitrate = np.random.randint(5)
        if rand_bitrate == 0:
            rate = '16k'
        elif rand_bitrate == 1:
            rate = '32k'
        elif rand_bitrate == 2:
            rate = '64k'
        elif rand_bitrate == 3:
            rate = '128k'
        else:
            rate = '256k'
        data.export(randomname + '.tmp', format='opus', 
                    bitrate = rate, codec='libopus')
        
        data_codec = AudioSegment.from_file(
            randomname + '.tmp', format='ogg',  codec='libopus') 
        data_codec = data_codec.set_frame_rate(16000)
        output_dat = np.array(data_codec.get_array_of_samples() / 
                              np.power(2, 16-1))

        # clean
        #os.remove(randomname + '.wav')
        try:
            os.remove(randomname + '.tmp')
        except FileNotFoundError:
            pass
        #os.remove(randomname + '.empty')
    
    try:
        os.remove(randomname + '.wav')
        os.remove(randomname + '.empty')
    except FileNotFoundError:
        pass
    
    if output_dat.shape[0] != input_dat.shape[0]:
        output_dat_ = np.zeros_like(input_dat[:, 0])
        minlen = min([output_dat.shape[0], input_dat.shape[0]])
        output_dat_[0:minlen] = output_dat_[0:minlen]
        output_dat = output_dat_
    return np.expand_dims(output_dat, axis=1)

##################
# Waveform morphing
##################

def morph_wavform(wav1, wav2, para=0.5, method=2,
                  fl = 320, fs = 160, nfft = 1024):
    """ output = morph_waveform(wav1, wav2, method=2, para=0.5
                 fl = 320, fs = 160, nfft = 1024)

    input
    -----
      wav1: np.array, (L,) or (L,1), input waveform 1
      wav2: np.array, (L,) or (L,1), input waveform 2
      para: float, coefficient for morphing
      method: int, 
          method = 1, waveform level morphing
             output = wav1 * para + wav2 * (1-para)
          method = 2, spec amplitude morphing
             amplitude = wav1_amp * para + wav2_amp * (1-para)
             phase is from wav2
          method = 3, phase morphing
             ...
             amplitude is from wav1
          method = 4, both spec and phase

      fl: int, frame length for STFT analysis when method > 1
      fs: int, frame shift for STFT analysis when method > 1
      nfft: int, fft points for STFT analysis when method > 1

    output
    ------
      output: np.array, same shape as wav1 and wav2
      
    """
    length = min([wav1.shape[0], wav2.shape[0]])
    
    if wav1.ndim > 1:
        data1 = wav1[0:length, 0]
    else:
        data1 = wav1[0:length]
        
    if wav2.ndim > 1:
        data2 = wav2[0:length, 0]
    else:
        data2 = wav2[0:length]

    if method == 1 or method == 'wav':
        # waveform level 
        data = data1 * para + data2 * (1.0 - para)

    elif method == 2 or method == 'specamp':
        # spectrum amplitude 
        
        _, _, Zxx1 = signal.stft(
            data1, nperseg=fl, noverlap=fl - fs, nfft=nfft)
        _, _, Zxx2 = signal.stft(
            data2, nperseg=fl, noverlap=fl - fs, nfft=nfft)
        
        amp1 = np.abs(Zxx1)
        amp2 = np.abs(Zxx2)
        pha1 = np.angle(Zxx1)
        pha2 = np.angle(Zxx2)
        
        # merge amplitude
        amp = np.power(amp1, para) * np.power(amp2, (1.0 - para))

        # 
        Zxx = amp * np.cos(pha1) + 1j * amp * np.sin(pha1)
        _, data = signal.istft(
            Zxx, nperseg = fl, noverlap = fl - fs, nfft = nfft)
        
    elif method == 3 or method == 'phase':
        # phase, 
        _, _, Zxx1 = signal.stft(
            data1, nperseg=fl, noverlap=fl - fs, nfft=nfft)
        _, _, Zxx2 = signal.stft(
            data2, nperseg=fl, noverlap=fl - fs, nfft=nfft)
        
        amp1 = np.abs(Zxx1)
        amp2 = np.abs(Zxx2)
        pha1 = np.unwrap(np.angle(Zxx1))
        pha2 = np.unwrap(np.angle(Zxx2))
        #amp = amp1 * para + amp2 * (1.0 - para)
        pha = pha1 * para + pha2 * (1.0 - para)
        Zxx = amp1 * np.cos(pha1) + 1j * amp1 * np.sin(pha)
        _, data = signal.istft(
            Zxx, nperseg=fl, noverlap=fl-fs, nfft=nfft)

    elif method == 4 or method == 'specamp-phase':
        # both
        _, _, Zxx1 = signal.stft(
            data1, nperseg=fl, noverlap=fl - fs, nfft=nfft)
        _, _, Zxx2 = signal.stft(
            data2, nperseg=fl, noverlap=fl - fs, nfft=nfft)
        
        amp1 = np.abs(Zxx1)
        amp2 = np.abs(Zxx2)
        pha1 = np.unwrap(np.angle(Zxx1))
        pha2 = np.unwrap(np.angle(Zxx2))
        amp = np.power(amp1, para) * np.power(amp2, (1.0 - para))
        pha = pha1 * para + pha2 * (1.0 - para)
        Zxx = amp * np.cos(pha1) + 1j * amp * np.sin(pha)
        _, data = signal.istft(
            Zxx, nperseg=fl, noverlap=fl-fs, nfft=nfft)

    # adjust length & shape
    data = unify_length_shape(data, wav1)
    return data


##################
# reverberation
##################

def wav_reverb(waveform, rir, use_fft=True, keep_alignment=False):
    """ output = wav_reverb(waveform, rir, use_fft=True, keep_alignment=False)

    input
    -----
      waveform: np.array, (length, 1), input waveform
      rir: np.array, (length, 1), room impulse response
      use_fft: bool, 
        True: use FFT to do convolution (default)
        False: use scipy.lfilter to do convolution (not implemented yet)
      keep_alignment: bool
        True: shift RIR so that max of RIR starts from 1st time step
        False: do nothing (default)
        
    output
    ------
      output_wav: np.array, (length, 1)
    """

    if use_fft:
        # handling different length        
        signal_length = max([waveform.shape[0], rir.shape[0]])
        
        # buffer 
        waveform_buf = np.zeros([signal_length])
        rir_buf = np.zeros([signal_length])
        waveform_buf[:waveform.shape[0]] = waveform[:, 0]
        rir_buf[:rir.shape[0]] = rir[:, 0]
        
        # alignment
        if keep_alignment:
            # get the max value of RIR
            max_index = np.argmax(rir, axis=0)[0]
            # circular shift the buffer
            rir_buf = np.roll(rir_buf, -max_index)        
        
        # fft
        convolved = np.fft.rfft(waveform_buf) * np.fft.rfft(rir_buf)

        # ifft
        output_wav = np.fft.irfft(convolved)
        
        # adjust volume
        orig_amp = nii_wav_tools.wav_get_amplitude(
            waveform, method='max')
        
        output_wav = nii_wav_tools.wav_scale_amplitude_to(
            output_wav, orig_amp, method='max')

    else:
        print("Not implemented")
        sys.exit(1)
    return np.expand_dims(output_wav, axis=1)



##################
# RawBoost
#  
# https://github.com/TakHemlata/RawBoost-antispoofing/blob/main/RawBoost.py
# 
# MIT license
# Copyright (c) 2021 Hemlata
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##################

def randRange(x1, x2, integer):
    y = np.random.uniform(low=x1, high=x2, size=(1,))
    if integer:
        y = int(y)
    return y

def normWav(x, always):
    if always:
        x = x/np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
            x = x/np.amax(abs(x))
    return x

def genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, 
                   minCoeff, maxCoeff, minG, maxG, fs):
    b = 1
    for i in range(0, nBands):
        fc = randRange(minF,maxF,0);
        bw = randRange(minBW,maxBW,0);
        c = randRange(minCoeff,maxCoeff,1);
          
        if c/2 == int(c/2):
            c = c + 1
        f1 = fc - bw/2
        f2 = fc + bw/2
        if f1 <= 0:
            f1 = 1/1000
        if f2 >= fs/2:
            f2 =  fs/2-1/1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),
            b)

    G = randRange(minG,maxG,0); 
    _, h = signal.freqz(b, 1, fs=fs)    
    b = pow(10, G/20)*b/np.amax(abs(h))   
    return b


def filterFIR(x,b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), 'constant')
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N/2):int(y.shape[0]-N/2)]
    return y


def LnL_convolutive_noise(x, 
                          fs=16000, N_f=5, nBands=5, 
                          minF=20, maxF=8000, minBW=100, maxBW=1000, 
                          minCoeff=10, maxCoeff=100, minG=0, maxG=0, 
                          minBiasLinNonLin=5, maxBiasLinNonLin=20):
    # Linear and non-linear convolutive noise
    y = [0] * x.shape[0]
    for i in range(0, N_f):
        if i == 1:
            minG = minG-minBiasLinNonLin;
            maxG = maxG-maxBiasLinNonLin;
        b = genNotchCoeffs(nBands, minF, maxF, minBW, 
                           maxBW, minCoeff, maxCoeff,minG,maxG,fs)
        y = y + filterFIR(np.power(x, (i+1)),  b)     
    y = y - np.mean(y)
    y = normWav(y,0)
    return y

def ISD_additive_noise(x, P=10, g_sd=2):
    # Impulsive signal dependent noise
    beta = randRange(0, P, 0)
    
    y = copy.deepcopy(x)
    x_len = x.shape[0]
    n = int(x_len*(beta/100))
    p = np.random.permutation(x_len)[:n]
    f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),
                     ((2*np.random.rand(p.shape[0]))-1))
    r = g_sd * x[p] * f_r
    y[p] = x[p] + r
    y = normWav(y,0)
    return y

def SSI_additive_noise(x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW, 
                       minCoeff, maxCoeff, minG, maxG, fs):
    # Stationary signal independent noise
    noise = np.random.normal(0, 1, x.shape[0])
    b = genNotchCoeffs(nBands, minF, maxF, minBW, maxBW, 
                       minCoeff, maxCoeff, minG, maxG, fs)
    noise = filterFIR(noise, b)
    noise = normWav(noise,1)
    SNR = randRange(SNRmin, SNRmax, 0)
    noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
    x = x + noise
    return x


def RawBoostWrapper12(x, fs=16000):
    """ RawBoost strategy 1+2
    """
    if x.ndim > 1:
        x_ = x[:, 0]
    else:
        x_ = x

    y = LnL_convolutive_noise(x_, fs)
    y = ISD_additive_noise(y)

    # make the length equal
    length = min(x.shape[0], y.shape[0])
    y_ = np.zeros_like(x_)
    y_[0:length] = y[0:length]

    if x.ndim > 1:
        y_ = np.expand_dims(y_, axis=1)
    return y_


if __name__ == "__main__":
    print("Waveform augmentation tools")

