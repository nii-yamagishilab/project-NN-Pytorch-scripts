#!/usr/bin/env python
# Tool to conduct copy-synthesis using DSP-based
#  algorithm or vocoder
# 
# python input_path output_path method
#   where method can be pyworld or griffinlim
# Require
#   librosa
#   soundfile with flac support

import os
import sys
import glob
import librosa
import numpy as np
import soundfile as sf
import pyworld as pw

def griffin_lim(input_wav, sr):
    """
    """
    S = np.abs(librosa.stft(input_wav))
    # Invert using Griffin-Lim
    y_inv = librosa.griffinlim(S)
    return y_inv

def pyworld(input_wav, sr):
    # analysis
    x = np.array(input_wav, dtype=np.float64)
    _f0, t = pw.dio(x, sr)    # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, sr)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, sr)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, sr)         # extract aperiodicity
    
    # synthesis the original waveform
    y = pw.synthesize(f0, sp, ap, sr) # synthesize an utterance using the parameters
    return y

if __name__ == "__main__":

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    option = sys.argv[3]

    print(input_path)
    if os.path.isfile(output_path):
        pass
    else:
        input_wav, sr = sf.read(input_path)
    
        if option == 'pyworld':
            output_wav = pyworld(input_wav, sr)
        elif option == 'griffinlim':
            output_wav = griffin_lim(input_wav, sr)
        else:
            assert 1==0, "unknown"
        
        sf.write(output_path, output_wav, sr)
        
