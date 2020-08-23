#!/usr/bin/python
""" This script extract Mel-spectrogram from input waveforms
Usage:
1. python sub_get_mel.py INPUT_WAV OUTPUT_MEL

Note: 
1. You may change the STFT and Mel configuration in the definition of class SpeechProcessing

"""
import os
import sys
import wave
import numpy as np

class SpeechProcessing(object):
    def __init__(self, sf=16000, fl=400, fs=80, fftl=1024, mfbsize=80, melmin=0, melmax=None):
        """
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
        
        winpower = np.sqrt(np.sum(np.square(np.blackman(self.fl).astype(np.float32))))
        self.window = np.blackman(self.fl).astype(np.float32) / winpower
        self.melfb = self._melfbank(self.melmin, self.melmax)

    def _freq2mel(self, freq):
        return 1127.01048 * np.log(freq / 700.0 + 1.0)

    def _mel2freq(self, mel):
        return (np.exp(mel / 1127.01048) - 1.0) * 700.0
        
    def _melfbank(self, melmin, melmax):
        linear_freq = 1000.0
        mfbsize = self.mfbsize - 1

        bFreq = np.linspace(0, self.sf / 2.0, self.fftl//2 + 1, dtype=np.float32)
        minMel = self._freq2mel(melmin)
        maxMel = self._freq2mel(melmax)
        iFreq = self._mel2freq(np.linspace(minMel, maxMel, mfbsize + 2, dtype=np.float32))
        linear_dim = np.where(iFreq<linear_freq)[0].size
        iFreq[:linear_dim+1] = np.linspace(iFreq[0], iFreq[linear_dim], linear_dim+1)

        diff = np.diff(iFreq)
        so = np.subtract.outer(iFreq, bFreq)
        lower = -so[:mfbsize] / np.expand_dims(diff[:mfbsize], 1)
        upper = so[2:] / np.expand_dims(diff[1:], 1)
        fb = np.maximum(0, np.minimum(lower, upper))

        enorm = 2.0 / (iFreq[2:mfbsize+2] - iFreq[:mfbsize])
        fb *= enorm[:, np.newaxis]

        fb0 = np.hstack([np.array(2.0*(self.fftl//2)/self.sf, np.float32), np.zeros(self.fftl//2, np.float32)])
        fb = np.vstack([fb0, fb])

        return fb

    def _frame(self, X):
        X = np.concatenate([np.zeros(self.fl//2, np.float32), X, np.zeros(self.fl//2, np.float32)])
        frame_num = (X.shape[0] - self.fl) // self.fs + 1
        F = np.zeros([frame_num, self.fl])
        for frame_idx in np.arange(frame_num):
            F[frame_idx, :] = X[frame_idx * self.fs : frame_idx * self.fs + self.fl]
        return F

    def _anawindow(self, F):
        W = F * self.window
        return W

    def _rfft(self, W):
        Y = np.fft.rfft(W, n=self.fftl).astype(np.complex64)
        return Y

    def _amplitude(self, Y):
        eps = 1.0E-12
        A = np.fmax(np.absolute(Y), eps)
        return A

    def _logmelfbspec(self, A):
        M = np.log(np.dot(A, self.melfb.T))
        return M

    def analyze(self, X):
        M = self._logmelfbspec(self._amplitude(self._rfft(self._anawindow(self._frame(X)))))
        return M



if __name__ == "__main__":
    
    # input waveform
    input_file = sys.argv[1]
    # output mel-spectrogram
    output_file = sys.argv[2]

    if os.path.isfile(input_file):

        # if necessary, configure the STFT and Mel-spectrogram configuration here
        sp = SpeechProcessing(fftl=1024, fl=400, fs=80)
        
        with wave.Wave_read(input_file) as f:
            T = np.frombuffer(f.readframes(f.getnframes()), np.int16).astype(np.float32)
            X = sp.analyze(T)
            
        f = open(output_file,'wb')
        datatype = np.dtype(('<f4',1))
        temp_data = X.astype(datatype)
        temp_data.tofile(f,'')
        f.close()
        print("processing %s and save to %s" % (input_file, output_file))
    else:
        print("cannot find %s" % (input_file))
        

