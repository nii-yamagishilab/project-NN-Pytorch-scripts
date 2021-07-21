from __future__ import absolute_import
from __future__ import print_function

import os
import scipy
import scipy.signal
import numpy as np
import scipy.io.wavfile


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

def read_raw_mat(filename,col,format='f4',end='l'):
    """read_raw_mat(filename,col,format='float',end='l')
       Read the binary data from filename
       Return data, which is a (N, col) array
       
       filename: the name of the file, take care about '\\'
       col:      the number of column of the data
       format:   please use the Python protocal to write format
                 default: 'f4', float32
                 see for more format:
       end:      little endian 'l' or big endian 'b'?
                 default: 'l'
       
       dependency: numpy
       Note: to read the raw binary data in python, the question
             is how to interprete the binary data. We can use
             struct.unpack('f',read_data) to interprete the data
             as float, however, it is slow.
    """
    f = open(filename,'rb')
    if end=='l':
        format = '<'+format
    elif end=='b':
        format = '>'+format
    else:
        format = '='+format
    datatype = np.dtype((format,(col,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:,0]
    else:
        return data

def write_raw_mat(data, filename, data_format='f4', end='l'):
    """write_raw_mat(data,filename,data_format='',end='l')                
       Write the binary data from filename.                               
       Return True                                                        
                                                                          
       data:     np.array                                                 
       filename: the name of the file, take care about '\\'               
       data_format:   please use the Python protocal to write data_format 
                 default: 'f4', float32                                   
       end:      little endian 'l' or big endian 'b'?                     
                 default: '', only when data_format is specified, end     
                 is effective                                             
                                                                          
       dependency: numpy                                                  
       Note: we can also write two for loop to write the data using       
             f.write(data[a][b]), but it is too slow                      
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input shoul be np.array")
        return False
    f = open(filename,'wb')
    if len(data_format)>0:
        if end=='l':
            data_format = '<'+data_format
        elif end=='b':
            data_format = '>'+data_format
        else:
            data_format = '='+data_format
        datatype = np.dtype(data_format)
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f,'')
    f.close()
    return True

    
def spec(data, fft_bins=4096, frame_shift=40, frame_length=240):
    """
    f, t, cfft = scipy.signal.stft(data, nfft=4096, noverlap=frame_length-frame_shift, nperseg=frame_length)
    """
    f, t, cfft = scipy.signal.stft(data, nfft=fft_bins, noverlap=frame_length-frame_shift, nperseg=frame_length)
    return f,t,cfft

def amplitude(cfft):
    """
    mag = amplitude(cfft)
    return spectral amplitude given FFT data (cfft) in complex numbers
    """
    mag = np.power(np.power(np.real(cfft),2) + np.power(np.imag(cfft),2), 0.5)
    return mag

def amplitude_re_im(data_re, data_im):
    """
    mag = amplitude_re_im(data_re, data_im)
    return spectral amplitude given real and imaginary part
    """
    mag = np.power(data_re * data_re + data_im * data_im, 0.5)
    return mag

def amplitude_to_db(mag):
    """
    20*np.log10(mag+ np.finfo(np.float32).eps)
    """
    return 20*np.log10(mag+ np.finfo(np.float32).eps)

def spec_amplitude(data,fft_bins=4096, frame_shift=40, frame_length=240):
    """
    return spectral amplitude in db, given speech waveform data
    mag_db = spec_amplitude(data,fft_bins=4096, frame_shift=40, frame_length=240):
    """
    _, _, cfft = spec(data, fft_bins, frame_shift, frame_length)
    mag  = amplitude(cfft)
    return amplitude_to_db(mag)

def fft_amplitude_db(data, nfft=1024):
    amp = np.fft.fft(data, nfft)[0:nfft/2+1]
    return amplitude_to_db(amplitude(amp))

def filter_res(data):
    w, h = scipy.signal.freqz(data, worN=4096/2)
    mag = amplitude(h)
    return amplitude_to_db(mag)


if __name__ == "__main__":
    pass