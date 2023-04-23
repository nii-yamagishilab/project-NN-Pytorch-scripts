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
import wave
import scipy.io.wavfile
try:
    import soundfile
except ModuleNotFoundError:
    pass
import core_scripts.data_io.io_tools as nii_io_tk

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"



def mulaw_encode(x, quantization_channels, scale_to_int=True):
    """x_mu = mulaw_encode(x, quantization_channels, scale_to_int=True)
    
    mu-law companding

    input
    -----
       x: np.array, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law to int
         False: return mu-law in (-1, 1)

    output
    ------
       x_mu: np.array, mulaw companded wave
    """
    mu = quantization_channels - 1.0
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    if scale_to_int:
        x_mu = np.array((x_mu + 1) / 2 * mu + 0.5, dtype=np.int32)
    return x_mu

def mulaw_decode(x_mu, quantization_channels, input_int=True):
    """mulaw_decode(x_mu, quantization_channels, input_int=True)
    
    mu-law decoding

    input
    -----
      x_mu: np.array, mu-law waveform
      quantization_channels: int, Number of channels
      input_int: Bool
        True: convert x_mu (int) from int to float, before mu-law decode
        False: directly decode x_mu (float)

    output
    ------
        x: np.array, waveform from mulaw decoding
    """
    mu = quantization_channels - 1.0
    if input_int:
        x = x_mu / mu * 2 - 1.0
    else:
        x = x_mu
    x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.0) / mu
    return x
    
def alaw_encode(x, quantization_channels, scale_to_int=True, A=87.6):
    """x_a = alaw_encoder(x, quantization_channels, scale_to_int=True, A=87.6)

    input
    -----
       x: np.array, float-valued waveforms in (-1, 1)
       quantization_channels (int): Number of channels
       scale_to_int: Bool
         True: scale mu-law to int
         False: return mu-law in (-1, 1)
       A: float, parameter for a-law, default 87.6
    output
    ------
       x_a: np.array, a-law companded waveform
    """
    num = quantization_channels - 1.0
    x_abs = np.abs(x)
    flag = (x_abs * A) >= 1
    
    x_a = A * x_abs
    x_a[flag] = 1 + np.log(x_a[flag])
    x_a = np.sign(x) * x_a / (1 + np.log(A))
    
    if scale_to_int:
        x_a = np.array((x_a + 1) / 2 * num + 0.5, dtype=np.int32)
    return x_a
    
def alaw_decode(x_a, quantization_channels, input_int=True, A=87.6):
    """alaw_decode(x_a, quantization_channels, input_int=True)

    input
    -----
      x_a: np.array, mu-law waveform
      quantization_channels: int, Number of channels
      input_int: Bool
        True: convert x_mu (int) from int to float, before mu-law decode
        False: directly decode x_mu (float)
       A: float, parameter for a-law, default 87.6
    output
    ------
       x: np.array, waveform
    """
    num = quantization_channels - 1.0
    if input_int:
        x = x_a / num * 2 - 1.0
    else:
        x = x_a
        
    sign = np.sign(x)
    x_a_abs = np.abs(x)
    
    x = x_a_abs * (1 + np.log(A))
    flag = x >= 1
    
    x[flag] = np.exp(x[flag] - 1)
    x = sign * x / A
    return x

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
    if 'soundfile' in sys.modules:
        x, sr = soundfile.read(wavFileIn)
    else:
        print("soundfile is not installed.")
        print("Due to practical reason, soundfile is not included in env.yml")
        print("To install soundfile with support to flac, try:")
        print(" conda install libsndfile=1.0.31 -c conda-forge")
        print(" conda install pysoundfile -c conda-forge")
        exit(1)
    return sr, x


def readWaveLength(wavFileIn):
    """ length = readWaveLength(wavFileIn)
    Read the length of the waveform

    Input: 
             waveFile, str, path to the input waveform
    Return: 
             length, int, length of waveform
    """
    with wave.open(wavFileIn, 'rb') as file_ptr:
        wavlength = file_ptr.getnframes()
    return wavlength


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
                    flag_output=0, 
                    flag_norm_amp=True,
                    flag_only_startend_sil = False,
                    opt_silence_handler = -1):
    """silence_handler(wav, sr, fl=320, fs=80,
                    max_thres_below=30, 
                    min_thres=-55, 
                    shortest_len_in_ms=50,
                    flag_output=0, 
                    flag_norm_amp=True,
                    flag_only_startend_sil = False,
                    opt_silence_handler = 1)
    
    Based on the Speech activity detector mentioned in Sec5.1 of
    Tomi Kinnunen, and Haizhou Li. 
    An Overview of Text-Independent Speaker Recognition: From Features to 
    Supervectors. Speech Communication 52 (1). 
    Elsevier: 12–40. doi:10.1016/j.specom.2009.08.009. 2010.
    
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
      flag_norm_amp: bool, whether normalize the waveform amplitude
          based on window function (default True)
      flag_only_startend_sil (obsolete): bool, whether only consider silence in 
          the begining and end. If False, silence within the utterance
          will be marked / removed (default False)

      opt_silence_handler:  int, option to silence trim handler
          0: equivalent to flag_only_startend_sil = False
          1: equivalent to flag_only_startend_sil = True
          2: remove only silence between words
         -1: not use this option, but follow flag_only_startend_sil

    output
    ------
      wav_no_sil: np.array, (length_1, ), waveform after removing silence
      sil_wav: np.array, (length_2, ), waveform in silence regions
      frame_tag: np.array, [0, 0, 0, 1, 1, ..., 1, 0, ], where 0 indicates
              silence frame, and 1 indicates a non-silence frame
      
      Note: output depends on flag_output
    """
    assert fs < fl, "Frame shift should be smaller than frame length"
    
    # frame the singal
    frames = buffering(wav, fl, fl - fs, 'nodelay')
    # apply window to each frame
    windowed_frames = windowing(frames)
    # buffer to save window prototype, this is used to normalize the amplitude
    window_proto = windowing(np.ones_like(frames))
    
    # compute the frame energy and assign a sil/nonsil flag
    frame_energy = 20*np.log10(np.std(frames, axis=1)+np.finfo(np.float32).eps)
    frame_energy_max = np.max(frame_energy)
    
    frame_tag = np.bitwise_and(
        (frame_energy > (frame_energy_max - max_thres_below)),
        frame_energy > min_thres)
    frame_tag = np.asarray(frame_tag, dtype=np.int)
    
    # post filtering of the sil/nonsil flag sequence
    seg_len_thres = shortest_len_in_ms * sr / 1000 / fs
    #  function to ignore short segments
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
    
    # remove short sil segments
    #  1-frame_tag indicates non-speech frames
    frame_process_sil = ignore_short_seg(1-frame_tag, seg_len_thres)
    #  reverse the sign
    frame_process_sil = 1 - frame_process_sil
    # remove short nonsil segments
    frame_process_all = ignore_short_seg(frame_process_sil, seg_len_thres)    
    frame_tag = frame_process_all
    

    if opt_silence_handler < 0:
        # if only consder silence in the front and end
        if flag_only_startend_sil:
            tmp_nonzero = np.flatnonzero(frame_tag)
        
            # start of the first nonsil segment
            #start_nonsil = np.asarray(frame_tag == 1).nonzero()[0]
            if np.any(tmp_nonzero):
                start_nonsil = np.flatnonzero(frame_tag)[0]
                # end of the last nonsil segment
                end_nonsil = np.flatnonzero(frame_tag)[-1]
                # all segments between are switched to nonsil
                frame_tag[start_nonsil:end_nonsil] = 1
            else:
                # no non-silence data, just let it pass
                pass
    elif opt_silence_handler == 1:
        # if only consder silence in the front and end
        tmp_nonzero = np.flatnonzero(frame_tag)
        
        # start of the first nonsil segment
        #start_nonsil = np.asarray(frame_tag == 1).nonzero()[0]
        if np.any(tmp_nonzero):
            start_nonsil = np.flatnonzero(frame_tag)[0]
            # end of the last nonsil segment
            end_nonsil = np.flatnonzero(frame_tag)[-1]
            # all segments between are switched to nonsil
            frame_tag[start_nonsil:end_nonsil] = 1
        else:
            # no non-silence data, just let it pass
            pass
    elif opt_silence_handler == 2:
        # if only consder silence in the front and end
        tmp_nonzero = np.flatnonzero(frame_tag)
        
        # start of the first nonsil segment
        #start_nonsil = np.asarray(frame_tag == 1).nonzero()[0]
        if np.any(tmp_nonzero):
            start_nonsil = np.flatnonzero(frame_tag)[0]
            # end of the last nonsil segment
            end_nonsil = np.flatnonzero(frame_tag)[-1]
            # all segments between are switched to nonsil
            frame_tag[:start_nonsil] = 1
            frame_tag[end_nonsil:] = 1
        else:
            # no non-silence data, just let it pass
            pass
    else:
        pass
        

    # separate non-speech and speech segments
    #  do overlap and add
    # buffer for speech segments
    spe_buf = np.zeros([np.sum(frame_tag) * fs + fl], dtype=wav.dtype)
    spe_buf_win = np.zeros([np.sum(frame_tag) * fs + fl], dtype=wav.dtype)
    # buffer for non-speech segments
    sil_buf = np.zeros([np.sum(1-frame_tag) * fs + fl], dtype=wav.dtype)
    sil_buf_win = np.zeros([np.sum(1-frame_tag) * fs + fl], dtype=wav.dtype)
    
    spe_fr_pt = 0
    non_fr_pt = 0
    for frame_idx, flag_speech in enumerate(frame_tag):
        if flag_speech:
            spe_buf[spe_fr_pt*fs:spe_fr_pt*fs+fl] += windowed_frames[frame_idx]
            spe_buf_win[spe_fr_pt*fs:spe_fr_pt*fs+fl] += window_proto[frame_idx]
            spe_fr_pt += 1
        else:
            sil_buf[non_fr_pt*fs:non_fr_pt*fs+fl] += windowed_frames[frame_idx]
            sil_buf_win[non_fr_pt*fs:non_fr_pt*fs+fl] += window_proto[frame_idx]
            non_fr_pt += 1

    # normalize the amplitude if necessary
    if flag_norm_amp:
        spe_buf_win[spe_buf_win < 0.0001] = 1.0
        sil_buf_win[sil_buf_win < 0.0001] = 1.0
        spe_buf /= spe_buf_win
        sil_buf /= sil_buf_win
    
    if flag_output == 1: 
        return spe_buf
    elif flag_output == 2:
        return sil_buf
    else:
        return spe_buf, sil_buf, frame_tag

###################
# wrapper functions
###################

def silence_handler_wrapper(wav, sr, fl=320, fs=80, 
                            max_thres_below=30, 
                            min_thres=-55, 
                            shortest_len_in_ms=50,
                            flag_output=0, 
                            flag_norm_amp=True,
                            flag_only_startend_sil=False):
    """Wrapper over silence_handler

    Many APIs used in this project assume (length, 1) shape.
    Thus, this API is a wrapper to accept (length, 1) and output (length, 1)
    
    See more on silence_handler
    """
    output = silence_handler(
        wav[:, 0], sr, fl, fs, max_thres_below, 
        min_thres, shortest_len_in_ms,
        flag_output, flag_norm_amp, flag_only_startend_sil)
    
    if flag_output == 1:
        # from (length) to (length, 1)
        return np.expand_dims(output, axis=1)
    elif flag_output == 2:
        return np.expand_dims(output, axis=1)
    else:
        return np.expand_dims(output[0], axis=1), \
            np.expand_dims(output[1], axis=1), \
            output[2]



###################
# Other tools
###################

def wav_get_amplitude(waveform, method='max'):
    """
    input
    -----
      wavform: np.array, (length, 1)
      method: str, 
        'max': compute np.max(np.abs(waveform))
        'mean': compute np.mean(np.abs(waveform))
        
    output
    ------
      amp: np.array (1) 
    """
    if method == 'max':
        return np.max(np.abs(waveform))
    else:
        return np.mean(np.abs(waveform))
    
def wav_norm_amplitude(waveform, method='max', floor=1e-12):
    """
    input
    -----
      wavform: np.array, (length, 1)
      method: str, 
        'max': compute np.max(np.abs(waveform))
        'mean': compute np.mean(np.abs(waveform))
        
    output
    ------
      amp: np.array (1) 
    """
    amp = wav_get_amplitude(waveform, method=method)
    amp = amp + floor if amp < floor else amp
    return waveform / amp

def wav_scale_amplitude_to(waveform, amp, method = 'max'):
    """
    input
    -----
      wavform: np.array, (length, 1)
      get_amp_method: str, 
        'max': compute np.max(np.abs(wavform))
        'mean': compute np.mean(np.abs(wavform))
        
    output
    ------
      waveform: np.array, (length, 1)
    """
    
    return wav_norm_amplitude(waveform, method=method) * amp


###################
# legacy functions
###################

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


#################################
# Other utilities based on Numpy
#################################
def f_overlap_cat(data_list, overlap_length):
    """Wrapper for overlap and concatenate

    input:
    -----
      data_list: list of np.array, [(length1, dim), (length2, dim)]
    
    output
    ------
      data: np.array, (length1 + length2 ... - overlap_length * N, dim)
    """
    data_dtype = data_list[0].dtype
    if data_list[0].ndim == 1:
        dim = 1
    else:
        dim = data_list[0].shape[1]

    total_length = sum([x.shape[0] for x in data_list])
    data_gen = np.zeros([total_length, dim], dtype=data_dtype)
    
    prev_end = 0
    for idx, data_trunc in enumerate(data_list):
        tmp_len = data_trunc.shape[0]
        if data_trunc.ndim == 1:
            data_tmp = np.expand_dims(data_trunc, 1)
        else:
            data_tmp = data_trunc

        if idx == 0:
            data_gen[0:tmp_len] = data_tmp
            prev_end = tmp_len
        else:
            win_len = min([prev_end, overlap_length, tmp_len])
            win_cof = np.arange(0, win_len)/win_len
            win_cof = np.expand_dims(win_cof, 1)
            data_gen[prev_end - win_len:prev_end] *= 1.0 - win_cof
            data_tmp[:win_len] *= win_cof
            data_gen[prev_end-win_len:prev_end-win_len+tmp_len] += data_tmp
            prev_end = prev_end-win_len+tmp_len
    return data_gen[0:prev_end]


if __name__ == "__main__":
    print("Definition of tools for wav")
