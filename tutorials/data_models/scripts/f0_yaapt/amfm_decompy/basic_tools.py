# -*- coding: utf-8 -*-
"""
Auxiliary classes and functions for used by the other AMFM_decompy modules.

Version 1.0.8.1
09/Jul/2018 Bernardo J.B. Schmitt - bernardo.jb.schmitt@gmail.com
"""

import numpy as np
from scipy.signal import lfilter


"""
Creates a signal object.
"""

class SignalObj(object):

    def __init__(self, *args):

        if len(args) == 1:
            try:
                from scipy.io import wavfile
            except:
                print("ERROR: Wav modules could not loaded!")
                raise KeyboardInterrupt
            self.fs, self.data = wavfile.read(args[0])
            self.name = args[0]
        elif len(args) == 2:
            self.data = args[0]
            self.fs = args[1]

        if self.data.dtype.kind == 'i':
            self.nbits = self.data.itemsize*8
            self.data = pcm2float(self.data, dtype='f')

        self.size = len(self.data)
        self.fs = float(self.fs)

        if self.size == self.data.size/2:
            print("Warning: stereo wav file. Converting it to mono for the analysis.")
            self.data = (self.data[:,0]+self.data[:,1])/2


    """
    Filters the signal data by a bandpass filter.
    """
    def filtered_version(self, bp_filter):

        tempData = lfilter(bp_filter.b, bp_filter.a, self.data)

        self.filtered = tempData[0:self.size:bp_filter.dec_factor]
        self.new_fs = self.fs/bp_filter.dec_factor

    """
    Method that uses the pitch values to estimate the number of modulated
    components in the signal.
    """

    def set_nharm(self, pitch_track, n_harm_max):

        n_harm = (self.fs/2)/np.amax(pitch_track) - 0.5
        self.n_harm = int(np.floor(min(n_harm, n_harm_max)))

    """
    Adds a zero-mean gaussian noise to the signal.
    """

    def noiser(self, pitch_track, SNR):

        self.clean = np.empty((self.size))
        self.clean[:] = self.data

        RMS = np.std(self.data[pitch_track > 0])
        noise = np.random.normal(0, RMS/(10**(SNR/20)), self.size)
        self.data += noise

"""
Transform a pcm raw signal into a float one, with values limited between -1 and
1.
"""

def pcm2float(sig, dtype=np.float64):

    sig = np.asarray(sig) # make sure it's a NumPy array
    assert sig.dtype.kind == 'i', "'sig' must be an array of signed integers!"
    dtype = np.dtype(dtype) # allow string input (e.g. 'f')

    # Note that 'min' has a greater (by 1) absolute value than 'max'!
    # Therefore, we use 'min' here to avoid clipping.
    return sig.astype(dtype) / dtype.type(-np.iinfo(sig.dtype).min)

