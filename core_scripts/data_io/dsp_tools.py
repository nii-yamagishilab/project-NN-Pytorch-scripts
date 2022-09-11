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
import scipy 

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
                 melmin=0, melmax=None, ver=1):
        """Melspec(sf, fl, fs, fftl, mfbsize, melmin, melmax)
        
        Args
        ----
          sf: int, sampling rate
          fl: int, frame length (number of waveform points)
          fs: int, frame shift
          fftl: int, FFT points
          mfbsize: int, mel-filter bank size
          melmin: float, lowest freq. covered by mel-filter bank, default 0
          melmax: float, highest freq. covered by mel-filter bank, default sf/2

        Note
        ----
          configuration for Voiceprivacy challenge:
             dsp_tools.Melspec(fftl=1024, fl=400, fs=160, ver=2)
        """
        #
        self.ver = ver
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

        if self.ver == 2:
            self.window = np.blackman(self.fl).astype(np.float32) / winpower
        else:
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

    def _preprocess(self, X):
        if self.ver == 2:
            # in ver2, assume wave in 16 bits
            return X * np.power(2, 15)
        else:
            return X

    def analyze(self, X):
        """Mel = analysze(X)
        input: X, np.array, waveform data, (length, )
        output: Mel, np.array, melspec., (frame_length, melfb_size)
        """
        X = self._preprocess(X)
        M = self._amplitude(self._rfft(self._anawindow(self._frame(X))))
        M = self._logmelfbspec(M)
        return M

class LPClite(object):
    """ A lite LPC analyzer & synthesizr
    Note that this is based on numpy, not Pytorch
    It can be used for pre-processing when loading data, or use
    it as data transformation function
    (see message at top)
    
    Example:
        # load waveform
        sr, wav = wav_tools.waveReadAsFloat(wav_path)
        m_lpc = LPClite(320, 80)
        # LPC analysis 
        lpc_coef, _, rc, gain, err, err_overlapped = m_lpc.analysis(wav)
        # LPC synthesis
        wav_re = m_lpc.synthesis(lpc_coef, err, gain)
        # rc to LPC 
        lpc_coef_tmp = m_lpc._rc2lpc(lpc_coef)
        np.std(lpc_coef_tmp - lpc_coef)
    """
    def __init__(self, fl=320, fs=80, order=29, window='blackman', 
                 flag_emph=True, emph_coef=0.97):
        """LPClite(fl=320, fs=80, order=30, window='blackman')
        Args
        ----
          fl: int, frame length
          fs: int, frame shift
          order: int, order of LPC, [1, a_1, a_2, ..., a_order]
          window: str, 'blackman' or 'hanning'
          flag_emph: bool, whether use pre-emphasis (default True)
          emph_coef: float, coefficit for pre-emphasis filter (default 0.97)

        Note that LPC model is defined as:
           1                          Gain 
        -------- --------------------------------------------- 
        1- bz^-1 a_0 + a_1 z^-1 + ... + a_order z^-(order)
        
        b = emph_coef if flag_emph is True
        b = 0 otherwise
        """
        self.fl = fl
        self.fs = fs
        # 
        self.order = order
        self.flag_emph = flag_emph
        self.emph_coef = emph_coef

        if np.abs(emph_coef) >= 1.0:
            print("Warning: emphasis coef {:f} set to 0.97".format(emph_coef))
            self.emph_coef = 0.97

        if window == 'hanning':
            self.win = np.hanning(self.fl)
        else:
            self.win = np.blackman(self.fl)
            
        return
    
    def analysis(self, wav):
        """lpc_coef, ld_err, gamma, gain, framed_err, err_signal = analysis(wav)
        
        LPC analysis on each frame
        
        input
        -----
          wav: np.array, (length, 1)
          
        output
        ------
          lpc_coef:   np.array, LPC coeff, (frame_num, lpc_order + 1)
          ld_err:     np.array, LD analysis error, (frame_num, lpc_order + 1)
          gamma:      np.array, reflection coefficients, (frame_num,lpc_order)
          gain:       np.array, gain, (frame_num, 1)
          framed_err: np.array, LPC error per frame, (frame_num, frame_length)
          eer_signal: np.array, overlap-added excitation (length, 1)

        Note that framed_err is the excitation signal from LPC analysis on each
        frame. eer_signal is the overlap-added excitation signal.
        """
        if self.flag_emph:
            wav_tmp = self._preemphasis(wav)
        else:
            wav_tmp = wav

        # framing & windowing
        frame_wined = self._windowing(self._framing(wav_tmp[:, 0]))

        # auto-correlation
        auto = self._auto_correlation(frame_wined)

        # LD analysis
        lpc_coef, lpc_err, gamma_array, gain = self._levison_durbin(auto)

        # get LPC excitation signals in each frame
        framed_err = self._lpc_analysis_core(lpc_coef, frame_wined, gain)

        # overlap-add for excitation signal
        err_signal = self._overlapadd(framed_err)
        
        return lpc_coef, lpc_err, gamma_array, gain, framed_err, err_signal
    
    def synthesis(self, lpc_coef, framed_err, gain):
        """wav = synthesis(lpc_coef, framed_err, gain):
        
        LPC synthesis (and overlap-add)
        
        input
        -----
          lpc_coef:   np.array, LPC coeff, (frame_num, lpc_order + 1)
          framed_err: np.array, LPC excitations, (frame_num, frame_length)
          gain:       np.array, LPC gain, (frame_num, 1)
        
        output
        ------
          wav: np.array, (length, 1)
        
        This function does LPC synthesis in each frame and create
        the output waveform by overlap-adding
        """
        framed_x = self._lpc_synthesis_core(lpc_coef, framed_err, gain)
        wav_tmp = self._overlapadd(framed_x)
        if self.flag_emph:
            wav_tmp = self._deemphasis(wav_tmp)
        return wav_tmp

    def _preemphasis(self, wav):
        """ wav_out = _preemphasis(wav)

        input
        -----
          wav: np.array, (length)

        output
        ------
          wav: np.array, (length)
        """
        wav_out = np.zeros_like(wav) + wav
        wav_out[1:] = wav_out[1:] - wav_out[0:-1] * self.emph_coef
        return wav_out

    def _deemphasis(self, wav):
        """ wav_out = _deemphasis(wav)

        input
        -----
          wav: np.array, (length)

        output
        ------
          wav: np.array, (length)
        """
        wav_out = np.zeros_like(wav) + wav
        for idx in range(1, wav.shape[0]):
            wav_out[idx] = wav_out[idx] + wav_out[idx-1] * self.emph_coef
        return wav_out

    def _framing(self, wav):
        """F = _framed(wav)
        
        Framing the signal
        
        input
        -----
          wav: np.array, (length)

        output
        ------
          F: np.array, (frame_num, frame_length)
        """
        frame_num = (wav.shape[0] - self.fl) // self.fs + 1
        F = np.zeros([frame_num, self.fl], dtype=wav.dtype)
        for frame_idx in np.arange(frame_num):
            F[frame_idx, :] = wav[frame_idx*self.fs : frame_idx*self.fs+self.fl]
        return F
    
    def _windowing(self, framed_x):
        """windowing
        """
        return framed_x * self.win
    
    
    def _overlapadd(self, framed_x):
        """wav = _overlapadd(framed_x)
        
        Do overlap-add on framed (and windowed) signal
        
        input
        -----
          framed_x: np.array, (frame_num, frame_length)
          
        output
        ------
          wav: np.array, (length, 1)
        
        length = (frame_num - 1) * frame_shift + frame_length
        """
        # waveform length
        wavlen = (framed_x.shape[0] - 1) * self.fs + self.fl
        wavbuf = np.zeros([wavlen])
        
        # buf to save overlapped windows (to normalize the signal amplitude)
        protobuf = np.zeros([wavlen])
        win_prototype = self._windowing(self._framing(np.ones_like(protobuf)))
        
        # overlap and add
        for idx in range(framed_x.shape[0]):
            frame_s = idx * self.fs
            wavbuf[frame_s : frame_s + self.fl] += framed_x[idx]
            protobuf[frame_s : frame_s + self.fl] += win_prototype[idx]
        
        # remove the impact of overlapped windows
        #protobuf[protobuf<1e-05] = 1.0
        wavbuf = wavbuf / protobuf.mean()
        return np.expand_dims(wavbuf, axis=1)
    
    def _lpc_analysis_core(self, lpc_coef, framed_x, gain):
        """framed_err = _lpc_analysis_core(lpc_coef, framed_x, gain)
        
        LPC analysis on frame
        MA filtering: e[n] = \sum_k=0 a_k x[n-k] / gain
        
        input
        -----
          lpc_coef: np.array, (frame_num, order + 1)
          framed_x: np.array, (frame_num, frame_length)
          gain: np.array,     (frame_num, 1)
          
        output
        ------
          framed_err: np.array, (frame_num, frame_length)
        
        Note that lpc_coef[n, :] = (1, a_1, a_2, ..., a_order) for n-th frame
        framed_x[n, :] = (x[0], x[1], ..., x[frame_len]) for n-th frame
        """
        # 
        frame_num = framed_x.shape[0]
        frame_len = framed_x.shape[1]
        
        # lpc order (without the a_0 term)
        order = lpc_coef.shape[1] - 1
        
        # pad zero, every frame has [0, ..., 0, x[0], x[1], ..., x[frame_len]]
        tmp_framed = np.concatenate(
            [np.zeros([frame_num, order + 1]), framed_x], axis=1)
        
        # flip to (x[frame_len], ... x[1], x[0], 0, ..., 0)
        tmp_framed = tmp_framed[:, ::-1]

        # LPC excitation buffer
        framed_err = np.zeros_like(framed_x)
        
        # e[n] = \sum_k=0 a[k] x[n-k] 
        # do this for all frames and n simultaneously
        for k in range(self.order + 1):
            # a[k]
            tmp_coef = lpc_coef[:, k:k+1]
            
            # For each frame
            # RHS = [x[n-k], x[n-k-1], ..., ] * a[k]
            # 
            # By doing this for k in [0, order]
            # LHS = [e[n],   e[n-1],   ...]
            #       [x[n-0], x[n-0-1], ..., ] * a[0]
            #     + [x[n-1], x[n-1-1], ..., ] * a[1]
            #     + [x[n-2], x[n-2-1], ..., ] * a[2]
            #     + ...
            # We get the excitation for one frame
            # This process is conducted for all frames at the same time
            framed_err += tmp_framed[:, 0:frame_len] * tmp_coef
            
            # roll to [x[n-k-1], x[n-k-2], ..., ]
            tmp_framed = np.roll(tmp_framed, -1, axis=1)

        # revese to (e[0], e[1], ..., e[frame_len])
        return framed_err[:, ::-1] / gain
    
    def _lpc_synthesis_core(self, lpc_coef, framed_err, gain):
        """framed_x = _lpc_synthesis_core(lpc_coef, framed_err, gain)
        
        AR filtering: x[n] = gain * e[n] - \sum_k=0 a_k x[n-k]
        LPC synthesis on frame
        
        input
        -----
          lpc_coef:   np.array, (frame_num, order + 1)
          framed_err: np.array, (frame_num, frame_length)
          gain:       np.array, (frame_num, 1)
          
        output
        ------
          framed_x:   np.array, (frame_num, frame_length)
        
        Note that 
        lpc_coef[n, :] = (1, a_1, a_2, ..., a_order),  for n-th frame
        framed_x[n, :] = (x[0], x[1], ..., x[frame_len]), for n-th frame
        """
        frame_num = framed_err.shape[0]
        frame_len = framed_err.shape[1]
        order = lpc_coef.shape[1] - 1
        
        # pad zero 
        # the buffer looks like 
        #  [[0, 0, 0, 0, 0, ... x[0], x[1], x[frame_length -1]],  -> 1st frame
        #   [0, 0, 0, 0, 0, ... x[0], x[1], x[frame_length -1]],  -> 2nd frame
        #   ...] 
        framed_x = np.concatenate(
            [np.zeros([frame_num, order]), np.zeros_like(framed_err)], axis=1)
        
        # flip the cofficients of each frame as [a_order, ..., a_1, 1]
        lpc_coef_tmp = lpc_coef[:, ::-1]
        
        # synthesis (all frames are down at the same time)
        for idx in range(frame_len):
            # idx+order so that it points to the shifted time idx
            #                                    idx+order
            # [0, 0, 0, 0, 0, ... x[0], x[1], ... x[idx], ]
            # gain * e[n]
            framed_x[:, idx+order] = framed_err[:, idx] * gain[:, 0]
            
            # [x[idx-1-order], ..., x[idx-1]] * [a_order, a_1]
            pred = np.sum(framed_x[:, idx:idx+order] * lpc_coef_tmp[:, :-1], 
                          axis=1)
            
            # gain * e[n] - [x[idx-1-order], ..., x[idx-1]] * [a_order, a_1]
            framed_x[:, idx+order] = framed_x[:, idx+order] - pred
            
        # [0, 0, 0, 0, 0, ... x[0], x[1], ... ] -> [x[0], x[1], ...]
        return framed_x[:, order:] 
    
    def _auto_correlation(self, framed_x):
        """ autocorr = _auto_correlation(framed_x)

        input
        -----
          framed_x: np.array, (frame_num, frame_length), frame windowed signal
        
        output
        ------
          autocorr: np.array, auto-correlation coeff (frame_num, lpc_order+1)
        """
        # (frame_num, order)
        autocor = np.zeros([framed_x.shape[0], self.order+1])
        
        # loop and compute auto-corr (for all frames simultaneously)
        for i in np.arange(self.order+1):
            autocor[:, i] = np.sum(
                framed_x[:, 0:self.fl-i] * framed_x[:, i:],
                axis=1)
            #print(autocor[0, i])
            #autocor[:, i] = 0
            #for idx in np.arange(self.fl):
            #    if (idx + i) < self.fl:
            #        autocor[:, i] += framed_x[:, idx] * framed_x[:, idx + i]
            #    else:
            #        break
            #print(autocor[0, i])
        # (frame_num, order)
        return autocor
    
    def _levison_durbin(self, autocor):
        """lpc_coef_ou, lpc_err, gamma_array, gain = _levison_durbin(autocor)
        Levison durbin 
        
        input
        -----
          autocor: np.array, auto-correlation, (frame_num, lpc_order+1)
        
        output
        ------
          lpc_coef: np.array, LPC coefficients, (frame_num, lpc_order+1)
          lpc_err: np.array, LPC error, (frame_num, lpc_order+1)
          gamma: np.array, reflection coeff, (frame_num, lpc_order)
          gain: np.array, gain, (frame_num, 1)
          
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
            
            # compute gamma
            #   step1.
            gamma = np.sum(lpc_coef[:, 0, 0:(index)] * autocor[:, 1:(index+1)], 
                           axis=1)
            #   step2. check validity of lpc_err
            ill_idx = lpc_err[:,index-1] < 1e-07
            #      also frames that should have been stopped in previous iter
            ill_idx = np.bitwise_or(ill_idx, tmp_order < polyOrder)
            #   step3. make invalid frame gamma=0
            gamma[ill_idx] = 0
            gamma[~ill_idx] = gamma[~ill_idx] / lpc_err[~ill_idx,index-1]
            gamma_array[:, index-1] = gamma
            #   step4. log down the ill frames
            tmp_order[ill_idx] = index

            lpc_coef[:, 1, 0] = -1.0 * gamma
            if index > 1:
                lpc_coef[:, 1, 1:index] = lpc_coef[:, 0, 0:index-1] \
                + lpc_coef[:, 1, 0:1] * lpc_coef[:, 0, 0:index-1][:, ::-1]

            lpc_err[:, index] = lpc_err[:, index-1] * (1 - gamma * gamma)
            lpc_coef[:, 0, :] = lpc_coef[:, 1, :]
        
        # flip to (1, a_1, ..., a_order)
        lpc_coef = lpc_coef[:, 0, ::-1]
        
        # output LPC coefficients
        lpc_coef_ou = np.zeros([frame_num, polyOrder])
        
        # if high-order LPC analysis is not working
        # each frame may require a different truncation length
        for idx in range(frame_num):
            lpc_coef_ou[idx, 0:tmp_order[idx]] = lpc_coef[idx, 0:tmp_order[idx]]
            
        # get the gain, when tmp_order = polyOrder, tmp_order-2 -> order-1, 
        #  last element of the lpc_err buffer
        gain = np.sqrt(lpc_err[np.arange(len(tmp_order)), tmp_order-2])
        
        # if the gain is zero, it means analysis error is zero, 
        gain[gain < 1e-07] = 1.0

        # (frame_num, order)
        return lpc_coef_ou, lpc_err, gamma_array, np.expand_dims(gain, axis=1)
    
    def _rc2lpc(self, rc):
        """lpc_coef = _rc2lpc(rc)
        from reflection coefficients to LPC coefficients
        forward Levinson recursion
        
        input
        -----
          rc: np.array, (frame_num, lpc_order)
        
        output
        ------
          lpc_coef, np.array, (frame_num, lpc_order+1)

        Note that LPC model is defined as:
                            Gain 
        ---------------------------------------------
        a_0 + a_1 z^-1 + ... + a_order z^-(order)

        Thus, the reflection coefficitns [gamma_1, ... gamma_order]
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
            lpc_coef[:, 0, :] = lpc_coef[:, 1,:]
        
        lpc_coef = lpc_coef[:, 0, ::-1]
        return lpc_coef
    


def f0resize(input_f0, input_reso, output_reso):
    """output_f0 = f0size(input_f0, input_reso, output_reso)
    
    input
    -----
      input_f0: array, (length, )
      input_reso: int, frame_shift, ms
      output_reso: int, frame_shift, ms

    output
    ------
      output_f0: array, (length2, )
         where length2 ~ np.ceil(length * input_reso / output_reso)
    """
    # function to merge two f0 value
    #  average them unless there is u/v mismatch
    def merge_f0(val1, val2):
        if val1 < 1 and val2 < 1:
            return (val1 + val2)/2
        elif val1 < 1:
            return val2
        elif val2 < 1:
            return val1
        else:
            return (val1 + val2)/2
        
    def retrieve_f0(buf, idx):
        if idx > 0 and idx < buf.shape[0]:
            return buf[idx]
        else:
            return 0
        
    # input length
    input_len = input_f0.shape[0]
    # output length
    output_len = int(np.ceil(input_len * input_reso / output_reso))
    # output buffer
    output_f0 = np.zeros([output_len])
    
    for idx in np.arange(output_len):
        input_idx = idx * output_reso / input_reso
        input_idx_left = int(np.floor(input_idx))
        input_idx_right = int(np.ceil(input_idx))
        
        # get the nearest value from input f0
        val1 = retrieve_f0(input_f0, input_idx_left)
        val2 = retrieve_f0(input_f0, input_idx_right)
        output_f0[idx] = merge_f0(val1, val2)
    return output_f0


def spectra_substraction(input1, input2, ratio=0.1, 
                         frame_length = 512, frame_shift = 256, fft_n = 512):
    """
    output = spectra_substraction(input1, input2, ratio=0.1,
                     frame_length = 512, frame_shift = 256, fft_n = 512)
    
    input
    -----
      input1: array, (length1 ), waveform to be denoised
      input2: array, (length2 ), waveform background noise
      ratio: float, weight to average spectra of noise
   
      frame_length, frame_shift, fft_n

    output
    ------
      output: array, (length 1)
      
    """
    _, _, input_spec1 = scipy.signal.stft(
        input1, nperseg = frame_length, 
        noverlap = frame_length - frame_shift, nfft=fft_n)

    _, _, input_spec2 = scipy.signal.stft(        
        input1, nperseg = frame_length, 
        noverlap = frame_length - frame_shift, nfft=fft_n)
    
    # ampltiude and phase
    amp1 = np.abs(input_spec1)
    pha1 = np.angle(input_spec1)
    
    # nosie average spectrum
    amp2 = np.abs(input_spec2)
    amp2 = amp2.mean(axis=1, keepdims=1)
    
    #idx = np.bitwise_and(amp1 > 0.0000001, amp2 > 0.0000001)
    #amp_new = amp1
    #amp_new[idx] = np.exp(np.log(amp1[idx]) - np.log((amp2[idx] * ratio)))
    
    # spectra substraction
    amp_new = amp1 - amp2 * ratio
    
    # keep amplitude none-negative
    amp_new[amp_new<0] = 0.0
    
    # reconstruct
    spec_new = amp_new * np.cos(pha1) + 1j * amp_new * np.sin(pha1)
    
    _, output = scipy.signal.istft(
        spec_new, nperseg=frame_length, 
        noverlap=frame_length - frame_shift, nfft = fft_n)
    
    return output


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
        _, inverse = scipy.signal.istft(angles, window = window, 
                                    nperseg=fl, noverlap=fl - fs, nfft = fft_n)

        # rebuild
        _, _, rebuilt = scipy.signal.stft(inverse, window = window, 
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


def warp_function_bilinear(normed_freq, alpha=0):
    """ warped_freq = warp_function_quadratic(normed_freq)

    Frequency warping using bi-linear function
    
    input
    -----
      normed_freq: np.array, (N, ), normalized frequency values
                   between 0 and pi
      alpha: float, warping coefficient. alpha=0 means no warping
     
    output
    ------
      warped_freq: np.array, (N, ), warpped normalized frequency
    
    Example
    -------
      orig_rad = np.arange(0, 512)/512 * np.pi
      warp_rad = warp_function_bilinear(orig_rad, alpha=0.3)
    
    """
    if np.any(normed_freq < 0) or np.any(normed_freq > np.pi):
        print("Input freq. out of range")
        sys.exit(1)
    nom = (1 - alpha * alpha) * np.sin(normed_freq)
    den = (1 + alpha * alpha) * np.cos(normed_freq) - 2 * alpha
    output = np.arctan(nom / den)
    output[output < 0] = output[output < 0] + np.pi
    return output

def warp_interpolation(spec, alpha, warp_func=None):
    """output = wrap_interpolation(spec, spec)

    Do frequency Warping and linear interpolation of spectrum.
    This is used for Vocal-tract pertubation

    input
    -----
      spec: spectra evelope, (L, N), where L is the frame number
      alpha: float, coefficients for warping
      warp_func: a warp function, 
            if None, we will use warp_function_bilinear in dsp_tools.py
      
    output
    ------
      output: spectra evelope, (L, N), where L is the frame number


    Example
    -------
      # let us vocal-tract length perturbation
      # let's do warping on spectrum envelope
      # we use pyworld to extract spectrum envelope 

      import pyworld as pw      
      x, sf = some_waveread_function(audio_file)

      # World analysis
      _f0, t = pw.dio(x, sf)    # raw pitch extractor
      f0 = pw.stonemask(x, _f0, t, sf)  # pitch refinement
      sp = pw.cheaptrick(x, f0, t, sf)  # extract smoothed spectrogram
      ap = pw.d4c(x, f0, t, sf)         # extract aperiodicity

      # Synthesis without warpping
      y = pw.synthesize(f0, sp, ap, sf) 

      # Synthesis after warpping 
      alpha = 0.1
      sp_wrapped = warp_interpolation(sp, warp_function_bilinear, alpha)
      ap_wrapped = warp_interpolation(ap, warp_function_bilinear, alpha)

      y_wrapped = pw.synthesize(f0, sp_wrapped, ap_wrapped, sf) 

      # please listen and compare y and y_wrapped
    """
    nbins = spec.shape[1]
    
    orig_rad = np.arange(0, nbins) / nbins * np.pi
    warp_rad = warp_func(orig_rad, alpha=alpha)
    if np.mean(np.abs(warp_rad - orig_rad)) < 0.0001:
        return spec
    else:
        output = np.zeros_like(spec)

        for rad_idx in np.arange(nbins):
            warp = warp_rad[rad_idx]
            warp_idx = warp / np.pi * nbins
            idx_left = int(np.floor(warp_idx))
            idx_right = int(np.ceil(warp_idx))

            if idx_left < 0:
                idx_left = 0
            if idx_right >= nbins:
                idx_right = nbins - 1

            if idx_left == idx_right:
                w_l, w_r = 0.0, 1.0
            else:
                w_l = warp_idx - idx_left
                w_r = idx_right - warp_idx
                
            # weighted sum for interpolation
            output[:,rad_idx] = spec[:,idx_left] * w_l + spec[:,idx_right] * w_r
        return output


if __name__ == "__main__":
    print("DSP tools using numpy")
    
    # Example for downing LPC analysis
    sr, data1 = wav_tools.waveReadAsFloat('media/arctic_a0001.wav')
    m_lpc = LPClite(320, 80)

    # LPC analysis                                                      
    lpc_coef, _, rc, gain, err, err_overlapped = m_lpc.analysis(
        np.expand_dims(data1, axis=1))

    # LPC synthesis                                                     
    wav_re = m_lpc.synthesis(lpc_coef, err, gain)                       

    # excitation with Gain
    excitation_new = m_lpc._overlapadd(err * gain)

    # need to import
    # from tutorials.plot_tools import plot_API
    # from tutorials.plot_tools import plot_lib
    plot_API.plot_API([wav_re[:, 0], data1, 
                       err_overlapped[:, 0], 
                       excitation_new[:, 0]], 
                      plot_lib.plot_spec, 'v')

    plot_API.plot_API([wav_re[:, 0] - err_overlapped[:, 0]], 
                      plot_lib.plot_spec, 'single')

    # RC to LPC
    lpc_coef_tmp = m_lpc._rc2lpc(rc)
    print(np.std(lpc_coef_tmp - lpc_coef))

