#!/usr/bin/python
"""
This script use pYAAPT to extract F0, which is robust to low-quality waveform
http://bingweb.binghamton.edu/~hhu1/pitch/YAPT.pdf
http://bjbschmitt.github.io/AMFM_decompy/pYAAPT.html

Usage:
1. specify configuration in __main__
2. $: python 00_get_f0.py input_wav output_f0

Note: 
1. the output will be binary, float32, litten-endian, which
   is compatible to HTS-scripts, CURRENNT-scripts

2. you can print it to string using SPTK x2x: 
   $: x2x +fa *.f0 > *.f0.txt

3. you can read it through Numpy
   >> f = open("PATH_TO_F0",'rb')
   >> datatype = np.dtype(("<f4",(col,)))
   >> f0 = np.fromfile(f,dtype=datatype)
   >> f.close()

4. you can also use pyTools by xin wang
   >> from ioTools import readwrite
   >> f0 = readwrite.read_raw_mat("PATH_TO_F0", 1)

"""
import os
import sys
import numpy

import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

def extractF0(input_wav, output_f0, min_f0 = 60, max_f0 = 400, frame_length = 35, frame_shift = 10):
    if os.path.isfile(input_wav):
        signal = basic.SignalObj(input_wav)
        pitch = pYAAPT.yaapt(signal, **{'f0_min': min_f0, 'f0_max': max_f0,
                                        'frame_length':frame_length,
                                        'frame_space':frame_shift})
        f0_value = pitch.samp_values
        datatype = numpy.dtype(('<f4',1))
        f0_value = numpy.asarray(f0_value, dtype=datatype)

        f = open(output_f0,'wb')
        f0_value.tofile(f,'')
        f.close()
        print("F0 processed: %s" % (output_f0))
    else:
        print("Cannot find %s" % (input_wav))
    return


def extractF0_wrapper(input_wav, min_f0 = 60, max_f0 = 600, 
                      frame_length = 25, frame_shift = 5):
    """extractF0_wrapper(input_wav, min_f0 = 60, max_f0 = 600, 
                      frame_length = 25, frame_shift = 5)
    
    input
    -----
      input_wav: string, path to the input waveform
      min_f0: int, minimum F0 (default 60 Hz)
      max_f0: int, maximum F0 (default 600 Hz)
      frame_length, int, analysis frame length in ms (default 25ms)
      frame_shift: int, analysis frame shift in ms (default 5ms)
    
    output
    ------
      f0: np.array, f0 sequence in shape [number_of_frame]
    """
    if os.path.isfile(input_wav):
        signal = basic.SignalObj(input_wav)
        pitch = pYAAPT.yaapt(signal, **{'f0_min': min_f0, 'f0_max': max_f0,
                                        'frame_length':frame_length,
                                        'frame_space':frame_shift})
        f0_value = pitch.samp_values
        f0_value = numpy.array(f0_value, dtype=numpy.float32)
        return f0_value
    else:
        print("Cannot find {:s}".format(input_wav))
        return None

if __name__ == "__main__":
    # configuration
    try:
        input_wave = sys.argv[1]
        output_f0 = sys.argv[2]
    except IndexError:
        print("Usage: python get_f0.py INPUT_WAV OUTPUT_F0")
        quit()
    
    # minimum F0 (Hz)
    min_f0 = 60
    # maximum F0 (Hz)    
    max_f0 = 600
    # frame length (ms)
    # frame length (ms)
    frame_length = 25
    # frame shift (ms)
    frame_shift = 5

    # no need to specify sampling rate
    
    extractF0(input_wave, output_f0, min_f0 = min_f0, max_f0 = max_f0,
              frame_length = frame_length, frame_shift = frame_shift)
