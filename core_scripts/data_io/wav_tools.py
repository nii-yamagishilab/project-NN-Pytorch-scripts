#!/usr/bin/env python
"""
data_io

Interface to process waveforms.

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
import scipy.io.wavfile
import soundfile
import core_scripts.data_io.io_tools as nii_io_tk

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

def wavformRaw2MuLaw(wavdata, bit=16, signed=True, quanLevel = 256.0):
    """ 
    wavConverted = wavformRaw2MuLaw(wavdata, bit=16, signed=True, \
                                    quanLevel = 256.0)
    Assume wavData is int type:
        step1. convert int wav -> float wav
        step2. convert linear scale wav -> mu-law wav

    Args: 
      wavdata: np array of int-16 or int-32 waveform 
      bit: number of bits to encode waveform
      signed: input is signed or not
      quanLevel: level of quantization (default 2 ^ 8)
    Returned:
      wav: integer stored as float numbers
    """
    if wavdata.dtype != np.int16 and wavdata.dtype != np.int32:
        print("Input waveform data in not int16 or int32")
        sys.exit(1)

    # convert to float numbers
    if signed==True:
        wavdata = np.array(wavdata, dtype=np.float32) / \
                  np.power(2.0, bit-1)
    else:
        wavdata = np.array(wavdata, dtype=np.float32) / \
                  np.power(2.0, bit)
    
    tmp_quan_level = quanLevel - 1
    # mu-law compansion
    wavtrans = np.sign(wavdata) * \
               np.log(1.0 + tmp_quan_level * np.abs(wavdata)) / \
               np.log(1.0 + tmp_quan_level)
    wavtrans = np.round((wavtrans + 1.0) * tmp_quan_level / 2.0)
    return wavtrans


def wavformMuLaw2Raw(wavdata, quanLevel = 256.0):
    """ 
    waveformMuLaw2Raw(wavdata, quanLevel = 256.0)
    
    Convert Mu-law waveform  back to raw waveform
    
    Args:
      wavdata: np array
      quanLevel: level of quantization (default: 2 ^ 8)
    
    Return:
      raw waveform: np array, float
    """
    tmp_quan_level = quanLevel - 1
    wavdata = wavdata * 2.0 / tmp_quan_level - 1.0
    wavdata = np.sign(wavdata) * (1.0/ tmp_quan_level) * \
              (np.power(quanLevel, np.abs(wavdata)) - 1.0)
    return wavdata


def float2wav(rawData, wavFile, bit=16, samplingRate = 16000):
    """ 
    float2wav(rawFile, wavFile, bit=16, samplingRate = 16000)
    Convert float waveform into waveform in int

    This is identitcal to waveFloatToPCMFile
    To be removed

    Args: 
         rawdata: float waveform data in np-arrary
         wavFile: output file path
         bit: number of bits to encode waveform in output *.wav
         samplingrate: 
    """
    rawData = rawData * np.power(2.0, bit-1)
    rawData[rawData >= np.power(2.0, bit-1)] = np.power(2.0, bit-1)-1
    rawData[rawData < -1*np.power(2.0, bit-1)] = -1*np.power(2.0, bit-1)
    
    # write as signed 16bit PCM
    if bit == 16:
        rawData  = np.asarray(rawData, dtype=np.int16)
    elif bit == 32:
        rawData  = np.asarray(rawData, dtype=np.int32)
    else:
        print("Only be able to save wav in int16 and int32 type")
        print("Save to int16")
        rawData  = np.asarray(rawData, dtype=np.int16)
    scipy.io.wavfile.write(wavFile, samplingRate, rawData)
    return
    
def waveReadAsFloat(wavFileIn):
    """ sr, wavData = wavReadToFloat(wavFileIn)
    Wrapper over scipy.io.wavfile
    Return: 
        sr: sampling_rate
        wavData: waveform in np.float32 (-1, 1)
    """
    
    sr, wavdata = scipy.io.wavfile.read(wavFileIn)
    
    if wavdata.dtype is np.dtype(np.int16):
        wavdata = np.array(wavdata, dtype=np.float32) / \
                  np.power(2.0, 16-1)
    elif wavdata.dtype is np.dtype(np.int32):
        wavdata = np.array(wavdata, dtype=np.float32) / \
                  np.power(2.0, 32-1)
    elif wavdata.dtype is np.dtype(np.float32):
        pass
    else:
        print("Unknown waveform format %s" % (wavFileIn))
        sys.exit(1)
    return sr, wavdata

def waveFloatToPCMFile(waveData, wavFile, bit=16, sr=16000):
    """waveSaveFromFloat(waveData, wavFile, bit=16, sr=16000)
    Save waveData (np.float32) as PCM *.wav
    
    Args:
       waveData: waveform data as np.float32
       wavFile: output PCM waveform file
       bit: PCM bits
       sr: sampling rate
    """
    
    # recover to 16bit range [-32768, +32767]
    rawData  = waveData * np.power(2.0, bit-1)
    rawData[rawData >= np.power(2.0, bit-1)] = np.power(2.0, bit-1)-1
    rawData[rawData < -1*np.power(2.0, bit-1)] = -1*np.power(2.0, bit-1)
    
    # write as signed 16bit PCM
    if bit == 16:
        rawData  = np.asarray(rawData, dtype=np.int16)
    elif bit == 32:
        rawData  = np.asarray(rawData, dtype=np.int32)
    else:
        print("Only be able to save wav in int16 and int32 type")
        print("Save to int16")
        rawData  = np.asarray(rawData, dtype=np.int16)
    scipy.io.wavfile.write(wavFile, sr, rawData)
    return

def flacReadAsFloat(wavFileIn):
    """ sr, wavData = flacReadAsFloat(wavFileIn)
    Wrapper over soundfile.read
    Return: 
        sr: sampling_rate
        wavData: waveform in np.float32 (-1, 1)
    """
    x, sr = soundfile.read(wavFileIn)
    return sr, x


def buffering(x, n, p=0, opt=None):
    """buffering(x, n, p=0, opt=None)
    input
    -----
      x: np.array, input signal, (length, )
      n: int, window length
      p: int, overlap, not frame shift
    
    outpupt
    -------
      output: np.array, framed buffer, (frame_num, frame_length)
      
    Example
    -------
       framed = buffer(wav, 320, 80, 'nodelay')
       
    Code from https://stackoverflow.com/questions/38453249/
    """
    if opt not in ('nodelay', None):
        raise ValueError('{} not implemented'.format(opt))
    i = 0
    if opt == 'nodelay':
        # No zeros at array start
        result = x[:n]
        i = n
    else:
        # Start with `p` zeros
        result = np.hstack([np.zeros(p), x[:n-p]])
        i = n-p
        
    # Make 2D array, cast to list for .append()
    result = list(np.expand_dims(result, axis=0))

    while i < len(x):
        # Create next column, add `p` results from last col if given
        col = x[i:i+(n-p)]
        if p != 0:
            col = np.hstack([result[-1][-p:], col])

        # Append zeros if last row and not length `n`
        if len(col):
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result.append(np.array(col))
        i += (n - p)

    return np.vstack(result).astype(x.dtype)

def windowing(framed_buffer, window_type='hanning'):
    """windowing(framed_buffer, window_type='hanning')
    
    input
    -----
      framed_buffer: np.array, (frame_num, frame_length), output of buffering
      window_type: str, default 'hanning'
      
    """
    if window_type == 'hanning':
        window = np.hanning(framed_buffer.shape[1])
    else:
        assert False, "Unknown window type in windowing"
    return framed_buffer * window.astype(framed_buffer.dtype)



def silence_handler(wav, sr, fl=320, fs=80, 
                    max_thres_below=30, 
                    min_thres=-55, 
                    shortest_len_in_ms=50,
                    flag_output=0):
    """silence_handler(wav, sr, fs, fl)
    
    input
    -----
      wav: np.array, (wav_length, ), wavform data
      sr: int, sampling rate
      fl: int, frame length, default 320
      fs: int, frame shift, in number of waveform poings, default 80
      
      flag_output: int, flag to select output
          0: return wav_no_sil, sil_wav, time_tag
          1: return wav_no_sil
          2: return sil_wav
      
      max_thres_below: int, default 30, max_enenergy - max_thres_below 
          is the lower threshold for speech frame
      min_thres: int, default -55, the lower threshold for speech frame
      shortest_len_in_ms: int, ms, default 50 ms, 
          segment less than this length is treated as speech
      
    output
    ------
      wav_no_sil: np.array, (length_1, ), waveform after removing silence
      sil_wav: np.array, (length_2, ), waveform in silence regions
      time_tag: [[start, end], []], where 
      
      Note: output depends on flag_output
    """
    assert fs < fl, "Frame shift should be smaller than frame length"
    
    frames = buffering(wav, fl, fl - fs, 'nodelay')
    windowed_frames = windowing(frames)
    
    frame_energy = 20*np.log10(np.std(frames, axis=1)+np.finfo(np.float32).eps)
    frame_energy_max = np.max(frame_energy)
    
    frame_tag = np.bitwise_and(
        (frame_energy > (frame_energy_max - max_thres_below)),
        frame_energy > min_thres)
    frame_tag = np.asarray(frame_tag, dtype=np.int)
    
    seg_len_thres = shortest_len_in_ms * sr / 1000 / fs
    
    
    def ignore_short_seg(frame_tag, seg_len_thres):
        frame_tag_new = np.zeros_like(frame_tag) + frame_tag
        # boundary of each segment
        seg_bound = np.diff(np.concatenate(([0], frame_tag, [0])))
        # start of each segment
        seg_start = np.argwhere(seg_bound == 1)[:, 0]
        # end of each segment
        seg_end = np.argwhere(seg_bound == -1)[:, 0]
        assert seg_start.shape[0] == seg_end.shape[0], \
            "Fail to extract segment boundaries"
        
        # length of segment
        seg_len = seg_end - seg_start
        seg_short_ids = np.argwhere(seg_len < seg_len_thres)[:, 0]
        for idx in seg_short_ids:
            start_frame_idx = seg_start[idx]
            end_frame_idx = seg_end[idx]
            frame_tag_new[start_frame_idx:end_frame_idx] = 0
        return frame_tag_new
    
    # work on non-speech, 1-frame_tag indicates non-speech frames
    frame_process_sil = ignore_short_seg(1-frame_tag, seg_len_thres)
    # reverse the sign
    frame_process_sil = 1 - frame_process_sil
    
    # work on speech
    frame_process_all = ignore_short_seg(frame_process_sil, seg_len_thres)
    
    # separate non-speech and speech segments
    #  do overlap and add
    frame_tag = frame_process_all
    # buffer for speech segments
    spe_buf = np.zeros([np.sum(frame_tag) * fs + fl], dtype=wav.dtype)
    # buffer for non-speech segments
    sil_buf = np.zeros([np.sum(1-frame_tag) * fs + fl], dtype=wav.dtype)
    spe_fr_pt = 0
    non_fr_pt = 0
    for frame_idx, flag_speech in enumerate(frame_tag):
        if flag_speech:
            spe_buf[spe_fr_pt*fs:spe_fr_pt*fs+fl] += windowed_frames[frame_idx]
            spe_fr_pt += 1
        else:
            sil_buf[non_fr_pt*fs:non_fr_pt*fs+fl] += windowed_frames[frame_idx]
            non_fr_pt += 1
    
    if flag_output == 1: 
        return spe_buf
    elif flag_output == 2:
        return sil_buf
    else:
        return spe_buf, sil_buf, frame_tag

if __name__ == "__main__":
    print("Definition of tools for wav")
