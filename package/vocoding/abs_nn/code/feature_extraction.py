#!/usr/bin/env python

import os
import sys
import pyworld as pw
import numpy as np
from core_scripts.data_io import io_tools
from core_scripts.data_io import wav_tools
from core_scripts.data_io import dsp_tools
from core_scripts.other_tools import list_tools
from core_scripts.other_tools import str_tools
from pathlib import Path
from multiprocessing import Pool

def extract_f0(wav_data, sr, frame_period):
    """f0 = extract_f0(wav_data, sr, frame_period)

    input: wav_data, np.array, in shape (length, 1), waveform data
    input: sr, int, sampling rate in Hz
    input: frame_period, float, frame length in ms
    output: f0, np.array, in shape (length, 1), f0 dat
    """
    x = np.array(wav_data, dtype=np.float64)
    _f0, t = pw.dio(x, sr, frame_period=frame_period)
    f0 = pw.stonemask(x, _f0, t, sr)  # pitch refinement                       
    return np.array(f0, dtype=np.float32)


def extract_mel_f0_core(f_mel_extractor, f_f0_extractor, input_wav):
    """mel, f0 = extract_mel_f0_core(f_mel_extractor, f_f0_extractor, input_wav)

    This function calls f_mel_extractor and f_f0_extractor to extract Mel-spectra
    and F0 from input wav. 

    input: f_mel_extractor, Mel-spectra extractor
    input: f_f0_extractor, F0 extractor,
    input: wav_data, np.array, in shape (length, 1), waveform data
    output: mel_spec, np.array, in shape (length, dim), mel-spectra data 
    output: f0, np.array, in shape (length, 1), f0 dat
    """
    # extract mel or other spectral featues                                    
    mel = f_mel_extractor(input_wav)
    # f0                                                                       
    f0 = f_f0_extractor(input_wav)

    # change shape of F0 data and assign 0 to unvoiced frames
    if f0.ndim > 1:
        f0 = f0[:, 0]
    f0[np.isnan(f0)] = 0

    # adjust length (number of frames) of two features
    mel_frame = mel.shape[0]
    if f0.shape[0] < mel_frame:
        f0 = np.concatenate(
            [f0, np.zeros([mel.shape[0] - f0.shape[0]])], axis=0)
    else:
        f0 = f0[:mel_frame]
    return mel, f0


def extract_mel_f0(input_dir, output_dir, filename, 
                   f_mel_extractor, f_f0_extractor, f_fileload,
                   mel_ext='.mel', f0_ext='.f0', wav_ext='.flac'):
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    #input_lst = list_tools.read_list_from_text(filelst)
    #
    #for filename in input_lst:
    if filename is not None:
        input_wav_path = (Path(input_dir) / filename).with_suffix(wav_ext)
        out_mel_name = (Path(output_dir) / filename).with_suffix(mel_ext)
        out_f0_name = (Path(output_dir) / filename).with_suffix(f0_ext)
        out_wav_name = (Path(output_dir) / filename).with_suffix(wav_ext)

        print(filename, flush=True)
        
        if os.path.isfile(out_mel_name) and os.path.isfile(out_f0_name):
            return
        
        sr, wav = f_fileload(input_wav_path)


        if wav_ext != '.wav':
            if not os.path.isfile(out_wav_name):
                wav_tools.waveFloatToPCMFile(wav, out_wav_name)
        
        mel, f0 = extract_mel_f0_core(f_mel_extractor, 
                                      f_f0_extractor,
                                      wav)
        
        parent_name = os.path.dirname(out_mel_name)
        if not os.path.isdir(parent_name):
            os.makedirs(parent_name)
        parent_name = os.path.dirname(out_f0_name)
        if not os.path.isdir(parent_name):
            os.makedirs(parent_name)
        
        io_tools.f_write_raw_mat(mel, out_mel_name)
        io_tools.f_write_raw_mat(f0, out_f0_name)

    return


if __name__ == "__main__":

    # sampling rate
    wav_sampling_rate = 16000
    # acoustic feature upsampling rate, 
    # this is equal to wav_sampling_rate * frame_shift 
    # here we use 10ms shift, thu 10ms * 16 kHz = 160
    feat_upsampling_rate = 160
    # FFT size for mel-spectra extraction
    feat_mel_fft_size = 1024
    # Frame length for Mel-spectra extraction
    feat_mel_frame_length = 400
    # Frame length for Pyworld (in ms), here is 10ms
    feat_f0_frame_period = feat_upsampling_rate * 1000 // wav_sampling_rate

    # arguments
    input_dir = sys.argv[2]
    output_dir = sys.argv[3]
    filelist = sys.argv[1]
    wav_ext = sys.argv[4]
    
    # extractors
    g_mel_extractor = dsp_tools.Melspec(sf=wav_sampling_rate, 
                                    fftl=feat_mel_fft_size, 
                                    fl=feat_mel_frame_length, 
                                    fs=feat_upsampling_rate, 
                                    ver=2)

    f_mel_extractor = lambda x: g_mel_extractor.analyze(x) 
    f_f0_extractor = lambda x: extract_f0(x, 
                                          sr=wav_sampling_rate, 
                                          frame_period=feat_f0_frame_period)
    if wav_ext == '.flac':
        wav_load = lambda x: wav_tools.flacReadAsFloat(x)
    elif wav_ext == '.wav':
        wav_load = lambda x: wav_tools.waveReadAsFloat(x)
    else:
        print("Unknown waveform format {:s}".format(wav_ext))
        sys.exit(1)
        
    # do extraction
    input_lst = list_tools.read_list_from_text(filelist)
    print("Processing {:d} files".format(len(input_lst)))

    if True:
        def _func(x):
            return extract_mel_f0(
                input_dir, output_dir, x,
                f_mel_extractor, f_f0_extractor, wav_load, wav_ext=wav_ext)
    
        with Pool(4) as executor:
            executor.map(_func, input_lst)
    else:
        for filename in input_lst:
            extract_mel_f0(
                input_dir, output_dir, filename,
                f_mel_extractor, f_f0_extractor, wav_load, wav_ext=wav_ext)
    
