#!/usr/bin/python3
"""
Extract Mel-spectrogram using Librosa

Example:
$: python3 00_get.py ../sample-data/HAW-001.wav HAW-001.mel  

Usage:
1. Modify the configuration in hparams.py
2. run python 00_get.py input_waveform output_mel

"""
import os
import numpy as np
import sys

from hparams import Hparams_class
from audio import Audio

if __name__ == "__main__":

    try:
        input_wav = sys.argv[1]
        output_mel = sys.argv[2]
    except IndexError:
        input_wav = "./HAW-001.wav" 
        output_mel = "HAW-001.mel"
        pass
    
    hparams_ins = Hparams_class()
    audio = Audio(hparams_ins)
    wav = audio.load_wav(input_wav)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    mel_spectrogram.tofile(output_mel, format="<f4")
    
