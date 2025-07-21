#!/usr/bin/env python
"""Functions for data loading

These functions are written using Numpy.
They should be used before data are casted into torch.tensor.

For example, use them in config.py:input_trans_fns or output_trans_fns
"""
import os
import sys
import librosa
import scipy.signal
import numpy as np
from pathlib import Path

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


def default_spec_config(sr = 16000):

    spec_config = {'n_fft': 2048, 
                   'n_spec': 2048 // 2 + 1,
                   'frame_length': int(0.05 * sr), 
                   'frame_shift': int(0.0125 * sr),
                   'preemphasis': 0.97,
                   'n_mels': 80,
                   'ref_db': 20,
                   'max_db': 100,
                   'save_path': '../tmp_feat'}

    return spec_config

def get_spec(audio_data, sr, config, filename=''):
    """
    """
    # filename
    if len(filename) > 0:
        mel_path = config['save_path'] + '/' + filename + '.mel'
        mag_path = config['save_path'] + '/' + filename + '.mag'

        # check buffer
        if os.path.isfile(mel_path + '.npy') and os.path.isfile(mag_path + '.npy'):
            mel = np.load(mel_path + '.npy')
            mag = np.load(mag_path + '.npy')
        else:
            Path(mel_path).parent.mkdir(parents=True, exist_ok=True)
            #os.makedirs(config['save_path'], exist_ok=True)
            mel, mag = _get_spec(audio_data, sr, config)
            np.save(mel_path, mel)
            np.save(mag_path, mag)
    else:
        mel, mag = _get_spec(audio_data, sr, config)

    return mel, mag, mel.shape[0]

def _get_spec(audio_data, sr, config):
    """
    """
    #y, sr = librosa.load(fpath, sr=sr)
    if audio_data.ndim == 2:
        y = audio_data[:, 0]
    elif audio_data.ndim == 1:
        y = audio_data

    # Trimming
    y, _ = librosa.effects.trim(y)

    # Preemphasis 1 - 0.97z^-1
    y = scipy.signal.lfilter([1, -config['preemphasis']], [1], y)

    # STFT
    linear = librosa.stft(y = y,
                          n_fft = config['n_fft'],
                          hop_length = config['frame_shift'],
                          win_length = config['frame_length'])

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr=sr, n_fft=config['n_fft'], n_mels=config['n_mels'])
    mel = np.dot(mel_basis, mag)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - config['ref_db'] + config['max_db'])/config['max_db'], 
                  1e-8, 1)

    mag = np.clip((mag - config['ref_db'] + config['max_db'])/config['max_db'], 
                  1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    
    # Adjust the length in advance
    len_ad = y.shape[0]//config['frame_shift']
    if mel.shape[0] > len_ad:
        mel = mel[:len_ad]
        mag = mag[:len_ad]
    
    return [mel, mag]


def mag2wav_gl(gen_spec, sr, config):
    """
    """
    
    # transpose
    mag = gen_spec.T

    # 
    mag = (np.clip(mag, 0, 1) * config['max_db']) - config['max_db'] + config['ref_db']

    # dB to liear amplitude
    amp = np.power(10.0, mag / 20)

    # wav reconstruction
    wav = GriffinLim(amp ** 1.2, 
                     100, 
                     fl = config['frame_length'], 
                     fs = config['frame_shift'], 
                     fft_n = config['n_fft'])

    # de-preemphasis
    wav = scipy.signal.lfilter([1], [1, -config['preemphasis']], wav)
    wav = wav / np.max(np.abs(wav))
    return np.expand_dims(wav, axis=-1)


def GriffinLim(sp_amp, n_iter, fl, fs, fft_n, 
               window='hann', momentum=0.99, init='rand'):
    """
    wav = GriffinLim(sp_amp, n_iter, fl, fs, fft_n, 
         window='hann', momentum=0.99, init='rand')
    
    Code based on librosa API.
    
    input
    -----
      sp_amp: array, (frame, fft_n//2+1), spectrum amplitude (linear domain)
      n_iter: int, number of GL iterations
      fl: int, frame length
      fs: int, frame shift
      fft_n: int, number of FFT points,
      window: str, default hann window
      momentum: float, momentum for fast GL iteration default 0.99
      init: str, initialization method of initial phase, default rand
    
    output
    ------
      wav: array, (length, ), reconstructed waveform
      
    Example
    -------
      nfft = 512
      fl = 512
      fs = 256

      _, _, data_stft = scipy.signal.stft(data1, window='hann', nperseg=fl, 
                                        noverlap=fl - fs, nfft = nfft)
      data_stft = np.abs(data_stft)

      wav = GriffinLim(data_stft, 32, fl, fs, nfft)
    """
    def angle_to_complex(x):
        return np.cos(x) + 1j * np.sin(x)

    # check data shape
    if sp_amp.shape[0] != fft_n // 2 + 1:
        spec_amp = sp_amp.T
        if spec_amp.shape[0] != fft_n // 2 + 1:
            print("Input sp_amp has shape {:s}".format(str(sp_amp)))
            print("FFT bin number is {:d}, incompatible".format(fft_n))
    else:
        spec_amp = sp_amp
    
    # small value
    eps = 0.0000001
    
    # buffer for angles
    angles = np.zeros(spec_amp.shape, dtype=np.complex64)
    
    # initialize phase
    if init == "rand":
        angles[:] = angle_to_complex(2*np.pi * np.random.rand(*spec_amp.shape))
    else:
        angles[:] = 1.0
    
    # Place-holders for temporary data and reconstructed buffer
    rebuilt = None
    tprev = None
    inverse = None

    # Absorb magnitudes into angles
    angles *= spec_amp
    
    for _ in range(n_iter):
        # Invert 
        _, inverse = scipy.signal.istft(
            angles, window = window, 
            nperseg=fl, noverlap=fl - fs, nfft = fft_n)

        # rebuild
        _, _, rebuilt = scipy.signal.stft(
            inverse, window = window, 
            nperseg=fl, noverlap=fl - fs, nfft = fft_n)

        # update
        angles[:] = rebuilt
        if tprev is not None:
            angles -= (momentum / (1 + momentum)) * tprev
        angles /= np.abs(angles) + eps
        angles *= spec_amp
        
        # 
        rebuilt, tprev = tprev, rebuilt

    # reconstruct
    _, wav = scipy.signal.istft(angles, window = window, 
                                nperseg=fl, noverlap=fl - fs, nfft = fft_n)
    return wav

    

if __name__ == "__main__":
    print("Tools for data augmentation")
