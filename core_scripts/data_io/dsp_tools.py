#!/usr/bin/env python
"""
dsp_tools

Interface to process waveforms with DSP tools

Note that functions here are based on numpy, and they are intended to be used
before data are converted into torch tensors. 

data on disk -> DataSet.__getitem__()  -----> Collate  ---->  Pytorch model
                             numpy.tensor           torch.tensor

These functions don't work on pytorch tensors
"""
from __future__ import absolute_import

import os
import sys
import numpy as np

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

class Melspec(object):
    """Melspec
    A simple class to produce and invert mel-spectrogram
    Note that this not compatible with librosa.melspec
    
    Most of the API is written by Dr. Shinji Takaki 
    """
    def __init__(self, sf=16000, fl=400, fs=80, fftl=1024, mfbsize=80, 
                 melmin=0, melmax=None):
        """Melspec(sf, fl, fs, fftl, mfbsize, melmin, melmax)
        
        Args
        ----
          sf: int, sampling rate
          fl: int, frame length (number of waveform points)
          fs: int, frame shift
          fftl: int, FFT points
          mfbsize: int, mel-filter bank size
          melmin: int, lowest freq. covered by mel-filter bank, default 0
          melmax: int, highest freq. covered by mel-filter bank, default sf/2
        """
        # sampling rate
        self.sf = sf
        # frame length 
        self.fl = fl
        # frame shift
        self.fs = fs
        # fft length
        self.fftl = fftl
        # mfbsize
        self.mfbsize = mfbsize
        # mel.min frequency (in Hz)
        self.melmin = melmin
        # mel.max frequency (in Hz)
        if melmax is None:
            self.melmax = sf/2
        else:
            self.melmax = melmax
        
        # windows
        self.window = np.square(np.blackman(self.fl).astype(np.float32))
        winpower = np.sqrt(np.sum(self.window))
        self.window = self.window / winpower
        
        # create mel-filter bank
        self.melfb = self._melfbank(self.melmin, self.melmax)
        
        # eps = 1.0E-12
        self.eps = 1.0E-12
        return
    
    def _freq2mel(self, freq):
        return 1127.01048 * np.log(freq / 700.0 + 1.0)

    def _mel2freq(self, mel):
        return (np.exp(mel / 1127.01048) - 1.0) * 700.0
        
    def _melfbank(self, melmin, melmax):
        linear_freq = 1000.0
        mfbsize = self.mfbsize - 1

        bFreq = np.linspace(0, self.sf / 2.0, self.fftl//2 + 1, 
                            dtype=np.float32)
        
        minMel = self._freq2mel(melmin)
        maxMel = self._freq2mel(melmax)
        
        iFreq = self._mel2freq(np.linspace(minMel, maxMel, mfbsize + 2, 
                                           dtype=np.float32))
        
        linear_dim = np.where(iFreq<linear_freq)[0].size
        iFreq[:linear_dim+1] = np.linspace(iFreq[0], iFreq[linear_dim], 
                                           linear_dim+1)

        diff = np.diff(iFreq)
        so = np.subtract.outer(iFreq, bFreq)
        lower = -so[:mfbsize] / np.expand_dims(diff[:mfbsize], 1)
        upper = so[2:] / np.expand_dims(diff[1:], 1)
        fb = np.maximum(0, np.minimum(lower, upper))

        enorm = 2.0 / (iFreq[2:mfbsize+2] - iFreq[:mfbsize])
        fb *= enorm[:, np.newaxis]

        fb0 = np.hstack([np.array(2.0*(self.fftl//2)/self.sf, np.float32), 
                         np.zeros(self.fftl//2, np.float32)])
        fb = np.vstack([fb0, fb])

        return fb
    
    def _melfbank_pinv(self, melfb):
        """get the pseudo inverse of melfb
        """
        return
        
    
    def _frame(self, X):
        """framing
        """
        X = np.concatenate([np.zeros(self.fl//2, np.float32), X, 
                            np.zeros(self.fl//2, np.float32)])
        frame_num = (X.shape[0] - self.fl) // self.fs + 1
        F = np.zeros([frame_num, self.fl])
        for frame_idx in np.arange(frame_num):
            F[frame_idx, :] = X[frame_idx*self.fs : frame_idx*self.fs+self.fl]
        return F

    def _anawindow(self, F):
        W = F * self.window
        return W

    def _rfft(self, W):
        Y = np.fft.rfft(W, n=self.fftl).astype(np.complex64)
        return Y

    def _amplitude(self, Y):
        A = np.fmax(np.absolute(Y), self.eps)
        return A

    def _logmelfbspec(self, A):
        M = np.log(np.dot(A, self.melfb.T))
        return M

    def analyze(self, X):
        """Mel = analysze(X)
        input: X, np.array, waveform data, (length, )
        output: Mel, np.array, melspec., (frame_length, melfb_size)
        """
        M = self._amplitude(self._rfft(self._anawindow(self._frame(X))))
        M = self._logmelfbspec(M)
        return M


class LPClite(object):
    """ A lite LPC analyzer & synthesizr
    Note that this is based on numpy, not Pytorch
    It can be used for pre-processing in dataloader
    
    Example:
        sr, wav = wav_tools.waveReadAsFloat(wav_path)
        m_lpc = LPClite(320, 80)
        # LPC analysis 
        lpc_coef, _, rc, gain, err = m_lpc.analysis(wav)
        # LPC synthesis
        wav_re = m_lpc.synthesis(lpc_coef, err, gain)
        
    """
    def __init__(self, fl=320, fs=80, order=30, window='blackman'):
        """LPClite(fl=320, fs=80, order=30, window='blackman')
        Args
        ----
          fl: int, frame length
          fs: int, frame shift
          order: int, order of LPC, [1, a_2, a_3, ..., a_order]
          window: str, 'blackman' or 'hanning'
        """
        self.fl = fl
        self.fs = fs
        self.order = order
        
        if window == 'hanning':
            self.win = np.hanning(self.fl)
        else:
            self.win = np.blackman(self.fl)
            
        return
    
    def analysis(self, wav):
        """lpc_coef, lpc_err, gamma_array, gain, framed_err = analysis(wav)
        
        LPC analysis on each frame
        
        input
        -----
          wav: np.array, (length, )
          
        output
        ------
          lpc_coef: np.array, LPC coeff, (frame_num, lpc_order)
          lpc_err: np.array, LD analysis error, (frame_num, lpc_order)
          gamma: np.array, reflection coefficients, (frame_num, lpc_order)
          gain: np.array, gain, (frame_num, 1)
          framed_err: np.array, LPC error, (frame_num, frame_length)
        """
        # framing & windowing
        frame_wined = self._windowing(self._framing(wav))
        # auto-correlation
        auto = self._auto_correlation(frame_wined)
        # LD analysis
        lpc_coef, lpc_err, gamma_array, gain = self._levison_durbin(auto)
        # get LPC error
        framed_err = self._lpc_analysis_core(lpc_coef, frame_wined, gain)
        return lpc_coef, lpc_err, gamma_array, gain, framed_err
    
    def synthesis(self, lpc_coef, framed_err, gain):
        """wav = synthesis(lpc_coef, framed_err, gain):
        
        LPC synthesis (and overlap-add)
        
        input
        -----
          lpc_coef: np.array, LPC coeff, (frame_num, lpc_order)
          gain: np.array, gain, (frame_num, 1)
          framed_err: np.array, LPC error, (frame_num, frame_length)

        output
        ------
          wav: np.array, (length, )
        """
        framed_x = self._lpc_synthesis_core(lpc_coef, framed_err, gain)
        return self._overlapadd(framed_x)
    
    def _framing(self, wav):
        frame_num = (wav.shape[0] - self.fl) // self.fs + 1
        F = np.zeros([frame_num, self.fl], dtype=wav.dtype)
        for frame_idx in np.arange(frame_num):
            F[frame_idx, :] = wav[frame_idx*self.fs : frame_idx*self.fs+self.fl]
        return F
    
    def _windowing(self, framed_x):
        return framed_x * self.win
    
    
    def _overlapadd(self, framed_x):
        wavlen = (framed_x.shape[0] - 1) * self.fs + self.fl
        wavbuf = np.zeros([wavlen])
        
        protobuf = np.zeros([wavlen])
        win_prototype = self._windowing(self._framing(np.ones_like(protobuf)))
        
        # overlap and add
        for idx in range(framed_x.shape[0]):
            frame_s = idx * self.fs
            wavbuf[frame_s : frame_s + self.fl] += framed_x[idx]
            protobuf[frame_s : frame_s + self.fl] += win_prototype[idx]
        protobuf[protobuf<1e-05] = 1.0
        wavbuf = wavbuf / protobuf
        return wavbuf
    
    def _lpc_analysis_core(self, lpc_coef, framed_x, gain):
        """framed_err = _lpc_analysis_core(lpc_coef, framed_x, gain)
        
        LPC analysis on frame
        MA filtering: e[n] = \sum_k=0 a_k x[n-k] / gain
        
        input
        -----
          lpc_coef: np.array, (frame_num, order)
          framed_x: np.array, (frame_num, frame_length)
          gain: np.array, (frame_num, )
          
        output
        ------
          framed_err: np.array, (frame_num, frame_length)
        
        Note that lpc_coef[n, :] = (1, a_2, a_3, ..., a_order) for n-th frame
        framed_x[n, :] = (x[0], x[1], ..., x[frame_len]) for n-th frame
        """
        # 
        frame_num = framed_x.shape[0]
        frame_len = framed_x.shape[1]
        
        # including the 1st 1.0
        order = lpc_coef.shape[1]
        
        # pad with zero  (0, 0, ..., 0, x[0], x[1], ..., x[frame_len])
        tmp_framed = np.concatenate(
            [np.zeros([frame_num, order]), framed_x], axis=1)
        
        # reverse to (x[frame_len], ... x[1], x[0]. 0, 0)
        tmp_framed = tmp_framed[:, ::-1]
        
        framed_err = np.zeros_like(framed_x)
        for idx in range(self.order):
            # (idx+1)-order cofficients for each frame, (frame_num, 1)
            tmp_coef = lpc_coef[:, idx:idx+1]
            
            # x[n-idx] * coef[idx], do it for all time steps
            framed_err += tmp_framed[:, 0:frame_len] * tmp_coef
            
            # (x[frame_len-1], ... x[1], x[0]. 0, 0, x[frame_en])
            tmp_framed = np.roll(tmp_framed, -1, axis=1)

        # revese to (e[0], e[1], ..., e[frame_len])
        return framed_err[:, ::-1] / np.expand_dims(gain, axis=1)
    
    def _lpc_synthesis_core(self, lpc_coef, framed_err, gain):
        """framed_x = _lpc_synthesis_core(lpc_coef, framed_err, gain)
        AR filtering: x[n] = gain * e[n] - \sum_k=0 a_k x[n-k]
        LPC synthesis on frame
        
        input
        -----
          lpc_coef: np.array, (frame_num, order)
          framed_err: np.array, (frame_num, frame_length)
          gain: np.array, (frame_num, )
          
        output
        ------
          framed_x: np.array, (frame_num, frame_length)
        
        Note that lpc_coef[n, :] = (1, a_2, a_3, ..., a_order) for n-th frame
        framed_x[n, :] = (x[0], x[1], ..., x[frame_len]) for n-th frame
        """
        frame_num = framed_err.shape[0]
        frame_len = framed_err.shape[1]
        order = lpc_coef.shape[1]
        
        framed_x = np.concatenate(
            [np.zeros([frame_num, order-1]), np.zeros_like(framed_err)], axis=1)
        
        # it is convenient to do this so that [a_order, ..., a_1, 1]
        lpc_coef_tmp = lpc_coef[:, ::-1]
        
        for idx in range(frame_len):
            framed_x[:, idx+order-1] = framed_err[:, idx] * gain
            # [x[n-order], ..., x[n-1], err] * [a_order, a_1, 1]
            framed_x[:, idx+order-1] += np.sum(
                framed_x[:, idx:idx+order-1] * -lpc_coef_tmp[:, :-1], axis=1)
        return framed_x[:, order-1:] 
    
    def _auto_correlation(self, framed_x):
        """
        """
        # (frame_num, order)
        autocor = np.zeros([framed_x.shape[0], self.order])
        
        # loop and compute auto-corre
        for i in np.arange(self.order):
            autocor[:, i] = 0
            for idx in np.arange(self.fl):
                if (idx + i) < self.fl:
                    autocor[:, i] += framed_x[:, idx] * framed_x[:, idx + i]
                else:
                    break
        # (frame_num, order)
        return autocor
    
    def _levison_durbin(self, autocor):
        """lpc_coef_ou, lpc_err, gamma_array, gain = _levison_durbin(autocor)
        Levison durbin 
        
        input
        -----
          autocor: np.array, auto-correlation, (frame_num, lpc_order)
        
        output
        ------
          lpc_coef: np.array, LPC coefficients, (frame_num, lpc_order)
          lpc_err: np.array, LPC error, (frame_num, lpc_order)
          gamma: np.array, reflection coeff, (frame_num, lpc_order-1)
          gain: np.array, gain, (frame_num)
          
        Note that lpc_coef[n] = (1, a_2, ... a_order) for n-th frame
        """
        # (frame_num, order)
        frame_num, order = autocor.shape 
        order = order - 1
        polyOrder = order + 1
        
        # to log down the invalid frames
        tmp_order = np.zeros([frame_num], dtype=np.int32) + polyOrder

        lpc_coef = np.zeros([frame_num, 2, polyOrder])
        lpc_err = np.zeros([frame_num, polyOrder])
        gamma_array = np.zeros([frame_num, order])
        gain = np.zeros([frame_num])
        
        lpc_err[:, 0] = autocor[:, 0]
        lpc_coef[:, 0, 0] = 1.0

        for index in np.arange(1, polyOrder):
            lpc_coef[:, 1, index] = 1.0
            gamma = np.sum(
                lpc_coef[:, 0, 0:(index)] * autocor[:, 1:(index+1)], axis=1)\
                / lpc_err[:, index-1]
            gamma_array[:, index-1] = gamma

            tmp_order[np.abs(gamma) > 1.0] = index
    
            lpc_coef[:, 1, 0] = -1.0 * gamma
            if index > 1:
                lpc_coef[:, 1, 1:index] = lpc_coef[:, 0, 0:index-1] \
                + lpc_coef[:, 1, 0:1] * lpc_coef[:, 0, 0:index-1][:, ::-1]
                #for index2 in np.arange(1, index):
                #    lpc_coef[:, 1, index2] = lpc_coef[:, 0, index2-1] \
                #       - gamma * lpc_coef[:, 0, index - 1 - index2]
            lpc_err[:, index] = lpc_err[:, index-1] * (1 - gamma * gamma)
            lpc_coef[:, 0, :] = lpc_coef[:, 1,:]
        
        # flip to (1, a_2, ..., a_order)
        lpc_coef = lpc_coef[:, 0, ::-1]
        
        # output LPC coefficients
        lpc_coef_ou = np.zeros([frame_num, polyOrder])
        
        # if high-order LPC analysis is not working
        # each frame may require a different truncation length
        for idx in range(frame_num):
            lpc_coef_ou[idx, 0:tmp_order[idx]] = lpc_coef[idx, 0:tmp_order[idx]]
        gain = np.sqrt(lpc_err[np.arange(len(tmp_order)), tmp_order-2])
        
        # (frame_num, order)
        return lpc_coef_ou, lpc_err, gamma_array, gain
    
    def _rc2lpc(self, rc):
        """from reflection coefficients to LPC coefficients
        """
        # (frame_num, order)
        frame_num, order = rc.shape 
        polyOrder = order + 1
        
        lpc_coef = np.zeros([frame_num, 2, polyOrder])
        lpc_coef[:, 0, 0] = 1.0
        
        for index in np.arange(1, polyOrder):
            lpc_coef[:, 1, index] = 1.0
            gamma = rc[:, index-1]
            lpc_coef[:, 1, 0] = -1.0 * gamma
            if index > 1:
                lpc_coef[:, 1, 1:index] = lpc_coef[:, 0, 0:index-1] \
                + lpc_coef[:, 1, 0:1] * lpc_coef[:, 0, 0:index-1][:, ::-1]
                #for index2 in np.arange(1, index):
                #    lpc_coef[:, 1, index2] = lpc_coef[:, 0, index2-1] \
                #       - gamma * lpc_coef[:, 0, index - 1 - index2]
            lpc_err[:, index] = lpc_err[:, index-1] * (1 - gamma * gamma)
            lpc_coef[:, 0, :] = lpc_coef[:, 1,:]
        
        lpc_coef = lpc_coef[:, 0, ::-1]
        return lpc_coef
    

if __name__ == "__main__":
    print("DSP tools using numpy")
