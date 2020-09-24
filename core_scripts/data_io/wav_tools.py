#!/usr/bin/env python
"""
data_io

Interface to process waveforms

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import scipy.io.wavfile

import core_scripts.data_io.io_tools as nii_io_tk

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


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

if __name__ == "__main__":
    print("Definition of tools for wav")
