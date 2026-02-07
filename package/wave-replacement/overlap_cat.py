#!/usr/bin/env python

import math
import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf

def get_window(length):
    """
    Return a windowing function of a specified length
    Hanning window by default
    
    Paramaters
    ----------
    length : int, window size
    """
    return np.hanning(length)

# copied from scipy signal
# https://github.com/scipy/scipy/blob/v1.15.3/scipy/signal/_signaltools.py

def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_size : int
        First input size.
    in2_size : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags


def concatenate_by_overlapdd(wav_collections, sampling_rate = 24000, 
                             frame_shift = 12, frame_length = 50, 
                             buffer_frames = 5):
    """
    output = concatenate_by_overlapdd(wav_collections, frame_shift = 288, frame_length = 1200, buffer_frames = 5)
    
    input
    -----
      wav_collections:    list of waveveform, each is a 1-dim np.array 
      sampling_rate:      int, sampling rate (Hz)
      frame_shift:        int, frame shift when doing overlap-add (ms)
      frame_length:       int, frame length when doing overlap-add (ms)
      buffer_frames:      int, how many number of frames for overlap-add
      
    output
    ------
      output:             np.array, waveform
      lag_bag:            list of int, list of lag between overlap regions
    """
    assert buffer_frames > 1, "buffer_frames must be > 1"
    
    # set variables for overlap-add
    
    # convert to number of sampling points
    frame_shift = sampling_rate // 1000 * frame_shift
    frame_length = sampling_rate // 1000 * frame_length
    
    # buffer length
    buffer_length = (buffer_frames - 1) * frame_shift + frame_length
    # overlap length (frame_shift + frame_length)
    overlap_length = frame_shift + frame_length

    # start from the first wave segment
    wav_pre = wav_collections[0]
    
    # loop over the rest of the 
    lag_bag = []
    for wav_pos in wav_collections[1:]:
        
        # adjust buffer_length in case wav_pre is shorter than 
        buffer_length_ = min([buffer_length, wav_pre.size-1, wav_pos.size-1])
        overlap_length_ = min([overlap_length, buffer_length_, wav_pre.size-1, wav_pos.size-1])
        
        # part that will be not overlapped
        wav_pre_base = wav_pre[:-buffer_length_]
        wav_pos_base = wav_pos[buffer_length_:]
        
        # this part will be around the overlap-area, may be used in overlapped as well
        wav_pre_buffer = wav_pre[-buffer_length_:]
        wav_pos_buffer = wav_pos[:buffer_length_]
        
        # this part will be the overlapped-area
        wav_pre_overlap = wav_pre[-overlap_length_:]
        wav_pos_overlap = wav_pos[:overlap_length_]

        ### compute lag between overlap part
        correlation = signal.correlate(wav_pre_overlap, wav_pos_overlap, mode="same")
        
        # we don't allow shifting the buffer left
        correlation[:correlation.shape[0]//2] = min(correlation)
        lags = correlation_lags(wav_pre_overlap.size, wav_pos_overlap.size, mode="same")
        lag = lags[np.argmax(correlation)]
        lag_bag.append(lag)
        
        ### add overlap
        overlap_length_calibrated = min([overlap_length - lag, buffer_length_])
        
        add_buffer_length = 2 * buffer_length_ - (overlap_length_calibrated)
        add_buffer = np.zeros(add_buffer_length)
        

        window = get_window(overlap_length_calibrated * 2)

        # Just add the window and do overlap-add
        wav_pre_buffer[-overlap_length_calibrated:] = wav_pre_buffer[-overlap_length_calibrated:] * window[overlap_length_calibrated:]
        wav_pos_buffer[:overlap_length_calibrated] = wav_pos_buffer[:overlap_length_calibrated] * window[:overlap_length_calibrated]
        add_buffer[:buffer_length_] += wav_pre_buffer
        add_buffer[-buffer_length_:] += wav_pos_buffer

        wav_pre = np.concatenate([wav_pre_base, add_buffer, wav_pos_base])

    # concatenated and overlap-added waveform
    wave_data = wav_pre
    return wave_data, lag_bag
    
import math
import numpy as np
import scipy.signal as signal

def compute_active_speech_level_detailed(signal, fs, target_level_dBm0=-26):
    """
    Compute the Active Speech Level (ASL) based on ITU-T P.56, with optional normalization.
    
    input: signal, input signal, Numpy array, (N, )
    input: fs, int, sampling rate, 16000
    input: target_level_dBm0, int, target active speech level, -26 dB

    output: asl_msq_db, float, active speech level (dB)
    output: signal, same shape as x
    output: scale, float, scale factor to makr input signal to be at the target_level
    """
    asl_msq_db, actfact, c0 = asl_P56(signal, fs)
    scale = 10 ** ((target_level_dBm0 - asl_msq_db) / 20.0)
    return asl_msq_db, signal * scale, scale
    

def asl_P56(x, fs, nbits=16):
    """
    input: x, input signal, (N, )
    input: fs, int, sampling rate, 16000
    input: nbits, int, number of bits per sample, default 16
    """
    assert x.ndim == 1
    eps = np.finfo(float).eps
    # time constant of smoothing, in seconds
    T = 0.03 
    # hangover time in seconds
    H = 0.2  
    # margin in dB of the difference between threshold and ASL
    M = 15.9  
    # number of thresholds, for 16 bit, it's 15
    thres_no = nbits-1  
    # reference
    refdB = 0
    
    # hangover in samples
    I = np.floor(fs * H + 0.5) 
    # smoothing factor in envelop detection
    g = np.exp(-1 / (fs * T))  
    
    # 
    a = np.full(thres_no, -1)
    # threshold
    c = 2**np.arange(-15, thres_no-15, dtype=float)
    # hangover counter for each level threshold
    hang = np.full(thres_no, I)  

    # length of x
    x_len = x.shape[0] 

    # squared sum of samples since last reset 
    sq = np.sum(x * x)
    # sum 
    s_sum = np.sum(x)
    # maximum values
    max_abs = np.max(np.abs(x))
    max_wav = np.max(x)
    min_wav = np.min(x)
    
    
    # use a 2nd order IIR filter to detect the envelope q
    x_abs = np.abs(x)
    p = signal.lfilter([1-g, 0], [1, -g], np.squeeze(x_abs))
    # q is the envelope, obtained from moving average of abs(x) (with slight "hangover").
    q = signal.lfilter([1-g, 0], [1, -g], np.squeeze(p))  

    # count the number of active points
    for k in range(x_len):
        for j in range(thres_no):
            if q[k] >= c[j]:
                a[j] = a[j]+1
                hang[j] = 0
            elif hang[j] < I:
                a[j] = a[j]+1
                hang[j] = hang[j]+1
            else:
                break

    asl_ms_log = -100.0
    actfact = 0
    asl_msq = 0
    c0 = 0
    
    if a[0] == -1:
        return asl_ms_log, actfact, c0
    else:
        a+=2
        AdB1=10*np.log10(sq/a[0]+eps)

    CdB1 = 20*np.log10(c[0]+eps)
    if (AdB1-CdB1 < M):
        return asl_ms_log, actfact, c0

    AdB = np.zeros(thres_no)
    CdB = np.zeros(thres_no)
    Delta = np.zeros(thres_no)
    
    AdB[0] = AdB1
    CdB[0] = CdB1
    Delta[0] = AdB1-CdB1

    for j in range(1, thres_no):
        AdB[j] = 10 * np.log10(sq/(a[j]+eps)+eps)
        CdB[j] = 20 * np.log10(c[j]+eps)

    for j in range(1, thres_no):
        if a[j] != 0:
            Delta[j]= AdB[j]- CdB[j]
            if Delta[j] <= M:
                # interpolate to find the actfact
                asl_ms_log, cl0 = bin_interp(AdB[j],
                    AdB[j-1], CdB[j], CdB[j-1], M, 0.5)
                # this is the mean square value NOT the rms
                asl_msq = 10**(asl_ms_log/10)
                # this is the proportion of the time that the speech is deemed "active"
                actfact = (sq/x_len)/asl_msq 
                # this is the threshold above which the speech is deemed "active".
                c0= 10**(cl0/20) 
                break
    
    return asl_ms_log - refdB, actfact, c0


# --------------------------------------------------------------------------

def bin_interp(upcount, lwcount, upthr, lwthr, Margin, tol):

    if tol < 0:
        tol = -tol

    # Check if extreme counts are not already the true active value
    iterno = 1
    if np.abs(upcount - upthr - Margin) < tol:
        asl_ms_log = upcount
        cc = upthr
        return asl_ms_log, cc

    if np.abs(lwcount - lwthr - Margin) < tol:
        asl_ms_log= lwcount
        cc= lwthr
        return asl_ms_log, cc

    # Initialize first middle for given (initial) bounds
    midcount = (upcount + lwcount) / 2.0
    midthr = (upthr + lwthr) / 2.0

    # Repeats loop until `diff' falls inside the tolerance (-tol<=diff<=tol)
    while 1:
        diff = midcount-midthr-Margin
        if abs(diff) <= tol:
            break

        # if tolerance is not met up to 20 iteractions, then relax the
        # tolerance by 10#
        iterno += 1

        if iterno>20:
            tol = tol*1.1

        if diff > tol:   # then new bounds are...
            midcount = (upcount+midcount)/2.0
            # upper and middle activities
            midthr = (upthr+midthr)/2.0
            # ... and thresholds
        elif diff < -tol:  # then new bounds are...
            midcount = (midcount+lwcount)/2.0
            # middle and lower activities
            midthr = (midthr+lwthr)/2.0
            # ... and thresholds

    #   Since the tolerance has been satisfied, midcount is selected
    #   as the interpolated value with a tol [dB] tolerance.

    asl_ms_log = midcount
    cc = midthr
    return asl_ms_log, cc


def replace_word_segment(wav_carrier: np.ndarray,
                         word_start_carrier: float,
                         word_end_carrier: float,
                         wav_fake: np.ndarray,
                         word_start_fake: float,
                         word_end_fake: float,
                         sr: int,
                         frame_shift: int = 6,
                         frame_length: int = 12,
                         buffer_frames: int = 2):
    """Replace a word segment with overlap handling
    
    input
    -----
      wav_carrier: input waveform (a bona fide one)
      word_start_carrier: starting time (s) of the segment to be replaced in carrier
      word_end_carrier: ending time (s) of the segment to be replaced in carrier
      wav_fake: input fake waveform 
      word_start_fake: starting time (s) of the segment to be inserted into wav_carrier
      word_end_fake: ending time (s) of the segment to be inserted into wav_carrier
      sr: sampling rate (Hz)
    """

    # overlap buffer length
    overlap_buffer = int((frame_shift + frame_length) / 1000 * sr)


    # separate original wave into segments
    start_sample = int(word_start_carrier * sr)
    end_sample = int(word_end_carrier * sr)
    
    # ensure we stay within bounds
    buffer_start = max(0, start_sample - overlap_buffer)
    buffer_end = min(len(wav_carrier), end_sample + overlap_buffer)
    before_seg = wav_carrier[:buffer_start].copy()
    orig_seg = wav_carrier[buffer_start:buffer_end].copy()
    after_seg = wav_carrier[buffer_end:].copy()
    
    # Fake segment from wav_fake
    start_sample_fake = int(word_start_fake * sr)
    end_sample_fake = int(word_end_fake * sr)
    buffer_start = max(0, start_sample_fake - overlap_buffer)
    buffer_end = min(len(wav_fake), end_sample_fake + overlap_buffer)
    fake_seg = wav_fake[buffer_start:buffer_end].copy()
    
    if before_seg.size < 1 or fake_seg.size < 1 or after_seg.size < 1:
        return None, None, None, None
    
    # Normalize speech activity level of the fake data
    target_level, _, _ = compute_active_speech_level_detailed(orig_seg, sr)
    _, fake_seg_normed, _ = compute_active_speech_level_detailed(fake_seg, sr, target_level)
     
    # Concatenate with overlap
    wav_collections = [before_seg, fake_seg_normed, after_seg]
    wav_replaced, lags = concatenate_by_overlapdd(
        wav_collections, sr,
        frame_shift=frame_shift,
        frame_length=frame_length,
        buffer_frames=buffer_frames
    )
    return wav_replaced, before_seg, fake_seg_normed, after_seg


if __name__ == "__main__":
    # the original waveform
    # any API to load file is OK
    sample_rate = 16000
    wav_orig, sr = librosa.load('sample/1089_134686_000012_000000.wav', sr=sample_rate)

    # time to start the replacement
    wav_start = 2.617
    # time to end the replacement
    wav_end = 3.067

    # the fake waveform
    # any API to load file is OK
    wav_fake, sr = librosa.load('sample_griffin_lim/1089_134686_000012_000000.wav', sr=sample_rate)
    # time to start the replacement
    wav_start_fake = 2.635
    # time to end the replacement
    wav_end_fake = 3.062

    wav_partial, _, _, _ = replace_word_segment(wav_orig, wav_start, wav_end, wav_fake, wav_start_fake, wav_end_fake, sr)
