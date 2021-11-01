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
          melmin: float, lowest freq. covered by mel-filter bank, default 0
          melmax: float, highest freq. covered by mel-filter bank, default sf/2
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

    plot_API.plot_API([wav_re[:, 0], data1, 
                       err_overlapped[:, 0], 
                       excitation_new[:, 0]], 
                      plot_lib.plot_spec, 'v')

    plot_API.plot_API([wav_re[:, 0] - err_overlapped[:, 0]], 
                      plot_lib.plot_spec, 'single')

    # RC to LPC
    lpc_coef_tmp = m_lpc._rc2lpc(rc)
    print(np.std(lpc_coef_tmp - lpc_coef))
