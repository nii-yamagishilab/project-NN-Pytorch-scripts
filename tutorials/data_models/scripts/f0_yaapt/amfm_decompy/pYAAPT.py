# -*- coding: utf-8 -*-
"""
This a ported version for Python from the YAAPT algorithm. The original MATLAB
program was written by Hongbing Hu and Stephen A.Zahorian.

The YAAPT program, designed for fundamental frequency tracking,
is extremely robust for both high quality and telephone speech.

The YAAPT program was created by the Speech Communication Laboratory of
the state university of New York at Binghamton. The original program is
available at http://www.ws.binghamton.edu/zahorian as free software. Further
information about the program could be found at Stephen A. Zahorian, and
Hongbing Hu, "A spectral/temporal method for robust fundamental frequency
tracking," J. Acoust. Soc. Am. 123(6), June 2008.

It must be noticed that, although this ported version is almost equal to the
original, some few changes were made in order to make the program more "pythonic"
and improve its performance. Nevertheless, the results obtained with both
algorithms were similar.

USAGE:
    pitch = yaapt(signal, <options>)

INPUTS:
    signal: signal object created by amfm_decompy.basic_tools. For more
    information about its properties, please consult the documentation file.

    <options>: must be formated as follows:
               **{'option_name1' : value1, 'option_name2' : value2, ...}
               The default configuration values for all of them are the same as
               in the original version. The main yaapt function in this file
               provides a short description about each option.
               For more information, please refer to the original bibliography.

OUTPUTS:
    pitch: pitch object. For more information about its properties, please
           consult the documentation file.

Version 1.0.8.1
09/Jul/2018 Bernardo J.B. Schmitt - bernardo.jb.schmitt@gmail.com
"""

import numpy as np
import numpy.lib.stride_tricks as stride_tricks
from scipy.signal import firwin, medfilt, lfilter
from scipy.signal.windows import hann, kaiser
import scipy.interpolate as scipy_interp
import amfm_decompy.basic_tools as basic


"""
--------------------------------------------
                Classes.
--------------------------------------------
"""
"""
Auxiliary class to handle the class properties.
"""
class ClassProperty(object):

    def __init__(self, initval=None):
        self.val = initval

    def __get__(self, obj, objtype):
        return self.val

    def __set__(self, obj, val):
        self.val = val


"""
Creates a pitch object.
"""
class PitchObj(object):

    PITCH_HALF = ClassProperty(0)
    PITCH_HALF_SENS = ClassProperty(2.9)
    PITCH_DOUBLE = ClassProperty(0)
    PITCH_DOUBLE_SENS = ClassProperty(2.9)
    SMOOTH_FACTOR = ClassProperty(5)
    SMOOTH = ClassProperty(5)
    PTCH_TYP = ClassProperty(100.0)

    def __init__(self, frame_size, frame_jump, nfft=8192):

        self.nfft = nfft
        self.frame_size = frame_size
        self.frame_jump = frame_jump
        self.noverlap = self.frame_size-self.frame_jump

    def set_energy(self, energy, threshold):
        self.mean_energy = np.mean(energy)
        self.energy = energy/self.mean_energy
        self.vuv = (self.energy > threshold)

    def set_frames_pos(self, frames_pos):
        self.frames_pos = frames_pos
        self.nframes = len(self.frames_pos)

    def set_values(self, samp_values, file_size, interp_tech='pchip'):
        self.samp_values = samp_values
        self.fix()
        self.values = self.upsample(self.samp_values, file_size, 0, 0,
                                    interp_tech)
        self.edges = self.edges_finder(self.values)
        self.interpolate()
        self.values_interp = self.upsample(self.samp_interp, file_size,
                                           self.samp_interp[0],
                                           self.samp_interp[-1], interp_tech)

    """
    For the voiced/unvoiced version of the pitch data, finds the n samples where
    the transitions between these two states occur.
    """
    def edges_finder(self, values):
        vec1 = (np.abs(values[1:]+values[:-1]) > 0)
        vec2 = (np.abs(values[1:]*values[:-1]) == 0)
        edges = np.logical_and(vec1, vec2)
        # The previous logical operation detects where voiced/unvoiced transitions
        # occur. Thus, a 'True' in the edges[n] sample indicates that the sample
        # value[n+1] has a different state than value[n](i.e. if values[n] is
        # voiced, then values[n+1] is unvoiced - and vice-versa). Consequently,
        # the last sample from edges array will always be 'False' and is not
        # calculated (because "there is no n+1 sample" for it. That's why
        # len(edges) = len(values)-1). However, just for sake of comprehension
        # (and also to avoid python warnings about array length mismatchs), I
        # add a 'False' to edges the array. But in pratice, this 'False' is
        # useless.
        edges = np.append(edges,[False])
        index = np.arange(len(values))
        index = index[edges > 0]
        return index.tolist()

    """
    This method corresponds to the first half of the ptch_fix.m file. It tries
    to fix half pitch and double pitch errors.
    """
    def fix(self):
        if self.PITCH_HALF > 0:
            nz_pitch = self.samp_values[self.samp_values > 0]
            idx = self.samp_values < (np.mean(nz_pitch)-self.PITCH_HALF_SENS *
                                      np.std(nz_pitch))
            if self.PITCH_HALF == 1:
                self.samp_values[idx] = 0
            elif self.PITCH_HALF == 2:
                self.samp_values[idx] = 2*self.samp_values[idx]

        if self.PITCH_DOUBLE > 0:
            nz_pitch = self.samp_values[self.samp_values > 0]
            idx = self.samp_values > (np.mean(nz_pitch)+self.PITCH_DOUBLE_SENS *
                                      np.std(nz_pitch))
            if self.PITCH_DOUBLE == 1:
                self.samp_values[idx] = 0
            elif self.PITCH_DOUBLE == 2:
                self.samp_values[idx] = 0.5*self.samp_values[idx]

    """
    Corresponds to the second half of the ptch_fix.m file. Creates the
    interpolated pitch data.
    """
    def interpolate(self):
        pitch = np.zeros((self.nframes))
        pitch[:] = self.samp_values
        pitch2 = medfilt(self.samp_values, self.SMOOTH_FACTOR)

        # This part in the original code is kind of confused and caused
        # some problems with the extrapolated points before the first
        # voiced frame and after the last voiced frame. So, I made some
        # small modifications in order to make it work better.
        edges = self.edges_finder(pitch)
        first_sample = pitch[0]
        last_sample = pitch[-1]

        if len(np.nonzero(pitch2)[0]) < 2:
            pitch[pitch == 0] = self.PTCH_TYP
        else:
            nz_pitch = pitch2[pitch2 > 0]
            pitch2 = scipy_interp.pchip(np.nonzero(pitch2)[0],
                                        nz_pitch)(range(self.nframes))
            pitch[pitch == 0] = pitch2[pitch == 0]
        if self.SMOOTH > 0:
            pitch = medfilt(pitch, self.SMOOTH_FACTOR)
        try:
            if first_sample == 0:
                pitch[:edges[0]-1] = pitch[edges[0]]
            if last_sample == 0:
                pitch[edges[-1]+1:] = pitch[edges[-1]]
        except:
            pass
        self.samp_interp = pitch

    """
    Upsample the pitch data so that it length becomes the same as the speech
    value.
    """
    def upsample(self, samp_values, file_size, first_samp=0, last_samp=0,
                 interp_tech='pchip'):
        if interp_tech is 'step':
            beg_pad = int((self.noverlap)/2)
            up_version = np.zeros((file_size))
            up_version[:beg_pad] = first_samp
            up_version[beg_pad:beg_pad+self.frame_jump*self.nframes] = \
                                    np.repeat(samp_values, self.frame_jump)
            up_version[beg_pad+self.frame_jump*self.nframes:] = last_samp

        elif interp_tech is 'pchip' or 'spline':
            if np.amin(samp_values) > 0:
                if interp_tech is 'pchip':
                    up_version = scipy_interp.pchip(self.frames_pos,
                                                    samp_values)(range(file_size))

                elif interp_tech is 'spline':
                    tck, u_original = scipy_interp.splprep(
                                                [self.frames_pos, samp_values],
                                                u=self.frames_pos)
                    up_version = scipy_interp.splev(range(file_size), tck)[1]
            else:
                beg_pad = int((self.noverlap)/2)
                up_version = np.zeros((file_size))
                up_version[:beg_pad] = first_samp
                voiced_frames = np.nonzero(samp_values)[0]
                edges = np.nonzero((voiced_frames[1:]-voiced_frames[:-1]) > 1)[0]
                edges = np.insert(edges, len(edges), len(voiced_frames)-1)
                voiced_frames = np.split(voiced_frames, edges+1)[:-1]

                for frame in voiced_frames:
                    up_interval = self.frames_pos[frame]
                    tot_interval = np.arange(int(up_interval[0]-(self.frame_jump/2)),
                                          int(up_interval[-1]+(self.frame_jump/2)))

                    if interp_tech is 'pchip' and len(frame) > 2:
                        up_version[tot_interval] = scipy_interp.pchip(
                                                    up_interval,
                                                    samp_values[frame])(tot_interval)

                    elif interp_tech is 'spline' and len(frame) > 2:
                        tck, u_original = scipy_interp.splprep(
                                            [up_interval, samp_values[frame]],
                                             u=up_interval)
                        up_version[tot_interval] = scipy_interp.splev(tot_interval, tck)[1]

                    # MD: In case len(frame)==2, above methods fail.
                    #Use linear interpolation instead.
                    elif len(frame) == 2:
                        up_version[tot_interval] = scipy_interp.interp1d(
                                                    up_interval,
                                                    samp_values[frame],
                                        fill_value='extrapolate')(tot_interval)

                    elif len(frame) == 1:
                        up_version[tot_interval] = samp_values[frame]


                up_version[beg_pad+self.frame_jump*self.nframes:] = last_samp

        return up_version

"""
Creates a bandpass filter object.
"""
class BandpassFilter(object):

    def __init__(self, fs, parameters):

        fs_min = 1000.0
        if (fs > fs_min):
            dec_factor = parameters['dec_factor']
        else:
            dec_factor = 1

        filter_order = parameters['bp_forder']
        f_hp = parameters['bp_low']
        f_lp = parameters['bp_high']

        f1 = f_hp/(fs/2)
        f2 = f_lp/(fs/2)

        self.b = firwin(filter_order+1, [f1, f2], pass_zero=False)
        self.a = 1
        self.dec_factor = dec_factor


"""
--------------------------------------------
                Main function.
--------------------------------------------
"""
def yaapt(signal, **kwargs):

    # Rename the YAAPT v4.0 parameter "frame_lengtht" to "tda_frame_length"
    # (if provided).
    if 'frame_lengtht' in kwargs:
        if 'tda_frame_length' in kwargs:
            warning_str = 'WARNING: Both "tda_frame_length" and "frame_lengtht" '
            warning_str += 'refer to the same parameter. Therefore, the value '
            warning_str += 'of "frame_lengtht" is going to be discarded.'
            print(warning_str)
        else:
            kwargs['tda_frame_length'] = kwargs.pop('frame_lengtht')

    #---------------------------------------------------------------
    # Set the default values for the parameters.
    #---------------------------------------------------------------
    parameters = {}
    parameters['frame_length'] = kwargs.get('frame_length', 35.0)   #Length of each analysis frame (ms)
    # WARNING: In the original MATLAB YAAPT 4.0 code the next parameter is called
    # "frame_lengtht" which is quite similar to the previous one "frame_length".
    # Therefore, I've decided to rename it to "tda_frame_length" in order to
    # avoid confusion between them. Nevertheless, both inputs ("frame_lengtht"
    # and "tda_frame_length") are accepted when the function is called.
    parameters['tda_frame_length'] = \
                              kwargs.get('tda_frame_length', 35.0)  #Frame length employed in the time domain analysis (ms)
    parameters['frame_space'] = kwargs.get('frame_space', 10.0)     #Spacing between analysis frames (ms)
    parameters['f0_min'] = kwargs.get('f0_min', 60.0)               #Minimum F0 searched (Hz)
    parameters['f0_max'] = kwargs.get('f0_max', 400.0)              #Maximum F0 searched (Hz)
    parameters['fft_length'] = kwargs.get('fft_length', 8192)       #FFT length
    parameters['bp_forder'] = kwargs.get('bp_forder', 150)          #Order of band-pass filter
    parameters['bp_low'] = kwargs.get('bp_low', 50.0)               #Low frequency of filter passband (Hz)
    parameters['bp_high'] = kwargs.get('bp_high', 1500.0)           #High frequency of filter passband (Hz)
    parameters['nlfer_thresh1'] = kwargs.get('nlfer_thresh1', 0.75) #NLFER boundary for voiced/unvoiced decisions
    parameters['nlfer_thresh2'] = kwargs.get('nlfer_thresh2', 0.1)  #Threshold for NLFER definitely unvoiced
    parameters['shc_numharms'] = kwargs.get('shc_numharms', 3)      #Number of harmonics in SHC calculation
    parameters['shc_window'] = kwargs.get('shc_window', 40.0)       #SHC window length (Hz)
    parameters['shc_maxpeaks'] = kwargs.get('shc_maxpeaks', 4)      #Maximum number of SHC peaks to be found
    parameters['shc_pwidth'] = kwargs.get('shc_pwidth', 50.0)       #Window width in SHC peak picking (Hz)
    parameters['shc_thresh1'] = kwargs.get('shc_thresh1', 5.0)      #Threshold 1 for SHC peak picking
    parameters['shc_thresh2'] = kwargs.get('shc_thresh2', 1.25)     #Threshold 2 for SHC peak picking
    parameters['f0_double'] = kwargs.get('f0_double', 150.0)        #F0 doubling decision threshold (Hz)
    parameters['f0_half'] = kwargs.get('f0_half', 150.0)            #F0 halving decision threshold (Hz)
    parameters['dp5_k1'] = kwargs.get('dp5_k1', 11.0)               #Weight used in dynamic program
    parameters['dec_factor'] = kwargs.get('dec_factor', 1)          #Factor for signal resampling
    parameters['nccf_thresh1'] = kwargs.get('nccf_thresh1', 0.3)    #Threshold for considering a peak in NCCF
    parameters['nccf_thresh2'] = kwargs.get('nccf_thresh2', 0.9)    #Threshold for terminating serach in NCCF
    parameters['nccf_maxcands'] = kwargs.get('nccf_maxcands', 3)    #Maximum number of candidates found
    parameters['nccf_pwidth'] = kwargs.get('nccf_pwidth', 5)        #Window width in NCCF peak picking
    parameters['merit_boost'] = kwargs.get('merit_boost', 0.20)     #Boost merit
    parameters['merit_pivot'] = kwargs.get('merit_pivot', 0.99)     #Merit assigned to unvoiced candidates in
                                                                    #defintely unvoiced frames
    parameters['merit_extra'] = kwargs.get('merit_extra', 0.4)      #Merit assigned to extra candidates
                                                                    #in reducing F0 doubling/halving errors
    parameters['median_value'] = kwargs.get('median_value', 7)      #Order of medial filter
    parameters['dp_w1'] = kwargs.get('dp_w1', 0.15)                 #DP weight factor for V-V transitions
    parameters['dp_w2'] = kwargs.get('dp_w2', 0.5)                  #DP weight factor for V-UV or UV-V transitions
    parameters['dp_w3'] = kwargs.get('dp_w3', 0.1)                  #DP weight factor of UV-UV transitions
    parameters['dp_w4'] = kwargs.get('dp_w4', 0.9)                  #Weight factor for local costs

    #---------------------------------------------------------------
    # Create the signal objects and filter them.
    #---------------------------------------------------------------
    fir_filter = BandpassFilter(signal.fs, parameters)
    nonlinear_sign = basic.SignalObj(signal.data**2, signal.fs)

    signal.filtered_version(fir_filter)
    nonlinear_sign.filtered_version(fir_filter)

    #---------------------------------------------------------------
    # Create the pitch object.
    #---------------------------------------------------------------
    nfft = parameters['fft_length']
    frame_size = int(np.fix(parameters['frame_length']*signal.fs/1000))
    frame_jump = int(np.fix(parameters['frame_space']*signal.fs/1000))
    pitch = PitchObj(frame_size, frame_jump, nfft)

    assert pitch.frame_size > 15, 'Frame length value {} is too short.'.format(pitch.frame_size)
    assert pitch.frame_size < 2048, 'Frame length value {} exceeds the limit.'.format(pitch.frame_size)


    #---------------------------------------------------------------
    # Calculate NLFER and determine voiced/unvoiced frames.
    #---------------------------------------------------------------
    nlfer(signal, pitch, parameters)

    #---------------------------------------------------------------
    # Calculate an approximate pitch track from the spectrum.
    #---------------------------------------------------------------
    spec_pitch, pitch_std = spec_track(nonlinear_sign, pitch, parameters)

    #---------------------------------------------------------------
    # Temporal pitch tracking based on NCCF.
    #---------------------------------------------------------------
    time_pitch1, time_merit1 = time_track(signal, spec_pitch, pitch_std, pitch,
                                          parameters)

    time_pitch2, time_merit2 = time_track(nonlinear_sign, spec_pitch, pitch_std,
                                          pitch, parameters)

    # Added in YAAPT 4.0
    if time_pitch1.shape[1] < len(spec_pitch):
        len_time = time_pitch1.shape[1]
        len_spec = len(spec_pitch)
        time_pitch1 = np.concatenate((time_pitch1, np.zeros((3,len_spec-len_time),
                                      dtype=time_pitch1.dtype)),axis=1)
        time_pitch2 = np.concatenate((time_pitch2, np.zeros((3,len_spec-len_time),
                                      dtype=time_pitch2.dtype)),axis=1)
        time_merit1 = np.concatenate((time_merit1, np.zeros((3,len_spec-len_time),
                                      dtype=time_merit1.dtype)),axis=1)
        time_merit2 = np.concatenate((time_merit2, np.zeros((3,len_spec-len_time),
                                      dtype=time_merit2.dtype)),axis=1)

    #---------------------------------------------------------------
    # Refine pitch candidates.
    #---------------------------------------------------------------
    ref_pitch, ref_merit = refine(time_pitch1, time_merit1, time_pitch2,
                                  time_merit2, spec_pitch, pitch, parameters)

    #---------------------------------------------------------------
    # Use dyanamic programming to determine the final pitch.
    #---------------------------------------------------------------
    final_pitch = dynamic(ref_pitch, ref_merit, pitch, parameters)

    pitch.set_values(final_pitch, signal.size)

    return pitch


"""
--------------------------------------------
                Side functions.
--------------------------------------------
"""

"""
Normalized Low Frequency Energy Ratio function. Corresponds to the nlfer.m file,
but instead of returning the results to them function, encapsulates them in the
pitch object.
"""
def nlfer(signal, pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    N_f0_min = np.around((parameters['f0_min']*2/float(signal.new_fs))*pitch.nfft)
    N_f0_max = np.around((parameters['f0_max']/float(signal.new_fs))*pitch.nfft)

    window = hann(pitch.frame_size+2)[1:-1]
    data = np.zeros((signal.size))  #Needs other array, otherwise stride and
    data[:] = signal.filtered     #windowing will modify signal.filtered

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    samples = np.arange(int(np.fix(float(pitch.frame_size)/2)),
                        signal.size-int(np.fix(float(pitch.frame_size)/2)),
                        pitch.frame_jump)

    data_matrix = np.empty((len(samples), pitch.frame_size))
    data_matrix[:, :] = stride_matrix(data, len(samples),
                                    pitch.frame_size, pitch.frame_jump)
    data_matrix *= window

    specData = np.fft.rfft(data_matrix, pitch.nfft)

    frame_energy = np.abs(specData[:, int(N_f0_min-1):int(N_f0_max)]).sum(axis=1)
    pitch.set_energy(frame_energy, parameters['nlfer_thresh1'])
    pitch.set_frames_pos(samples)

"""
Spectral pitch tracking. Computes estimates of pitch using nonlinearly processed
speech (typically square or absolute value) and frequency domain processing.
Search for frequencies which have energy at multiplies of that frequency.
Corresponds to the spec_trk.m file.
"""
def spec_track(signal, pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    nframe_size = pitch.frame_size*2
    maxpeaks = parameters['shc_maxpeaks']
    delta = signal.new_fs/pitch.nfft

    window_length = int(np.fix(parameters['shc_window']/delta))
    half_window_length = int(np.fix(float(window_length)/2))
    if not(window_length % 2):
        window_length += 1

    max_SHC = int(np.fix((parameters['f0_max']+parameters['shc_pwidth']*2)/delta))
    min_SHC = int(np.ceil(parameters['f0_min']/delta))
    num_harmonics = parameters['shc_numharms']

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    cand_pitch = np.zeros((maxpeaks, pitch.nframes))
    cand_merit = np.ones((maxpeaks, pitch.nframes))

    data = np.append(signal.filtered,
                  np.zeros((1, nframe_size +
                         ((pitch.nframes-1)*pitch.frame_jump-signal.size))))

    #Compute SHC for voiced frame
    window = kaiser(nframe_size, 0.5)
    SHC = np.zeros((max_SHC))
    row_mat_list = np.array([np.empty((max_SHC-min_SHC+1, window_length))
                            for x in range(num_harmonics+1)])

    magnitude = np.zeros(int((half_window_length+(pitch.nfft/2)+1)))

    for frame in np.where(pitch.vuv)[0].tolist():
        fir_step = frame*pitch.frame_jump

        data_slice = data[fir_step:fir_step+nframe_size]*window
        data_slice -= np.mean(data_slice)

        magnitude[half_window_length:] = np.abs(np.fft.rfft(data_slice,
                                                pitch.nfft))

        for idx,row_mat in enumerate(row_mat_list):
            row_mat[:, :] = stride_matrix(magnitude[min_SHC*(idx+1):],
                                          max_SHC-min_SHC+1,
                                          window_length, idx+1)
        SHC[min_SHC-1:max_SHC] = np.sum(np.prod(row_mat_list,axis=0),axis=1)

        cand_pitch[:, frame], cand_merit[:, frame] = \
            peaks(SHC, delta, maxpeaks, parameters)

    #Extract the pitch candidates of voiced frames for the future pitch selection.
    spec_pitch = cand_pitch[0, :]
    voiced_cand_pitch = cand_pitch[:, cand_pitch[0, :] > 0]
    voiced_cand_merit = cand_merit[:, cand_pitch[0, :] > 0]
    num_voiced_cand = len(voiced_cand_pitch[0, :])
    avg_voiced = np.mean(voiced_cand_pitch[0, :])
    std_voiced = np.std(voiced_cand_pitch[0, :])

    #Interpolation of the weigthed candidates.
    delta1 = abs((voiced_cand_pitch - 0.8*avg_voiced))*(3-voiced_cand_merit)
    index = delta1.argmin(0)

    voiced_peak_minmrt = voiced_cand_pitch[index, range(num_voiced_cand)]
    voiced_merit_minmrt = voiced_cand_merit[index, range(num_voiced_cand)]

    voiced_peak_minmrt = medfilt(voiced_peak_minmrt,
                                 max(1, parameters['median_value']-2))

    #Replace the lowest merit candidates by the median smoothed ones
    #computed from highest merit peaks above.
    voiced_cand_pitch[index, range(num_voiced_cand)] = voiced_peak_minmrt
    voiced_cand_merit[index, range(num_voiced_cand)] = voiced_merit_minmrt

    #Use dynamic programming to find best overal path among pitch candidates.
    #Dynamic weight for transition costs balance between local and
    #transition costs.
    weight_trans = parameters['dp5_k1']*std_voiced/avg_voiced

    if num_voiced_cand > 2:
        voiced_pitch = dynamic5(voiced_cand_pitch, voiced_cand_merit,
                                weight_trans, parameters['f0_min'])
        voiced_pitch = medfilt(voiced_pitch, max(1, parameters['median_value']-2))

    else:
        if num_voiced_cand > 0:
            voiced_pitch = (np.ones((num_voiced_cand)))*150.0
        else:
            voiced_pitch = np.array([150.0])
            cand_pitch[0, 0] = 0

    pitch_avg = np.mean(voiced_pitch)
    pitch_std = np.std(voiced_pitch)
    spec_pitch[cand_pitch[0, :] > 0] = voiced_pitch[:]

    if (spec_pitch[0] < pitch_avg/2):
        spec_pitch[0] = pitch_avg

    if (spec_pitch[-1] < pitch_avg/2):
        spec_pitch[-1] = pitch_avg

    spec_voiced = np.array(np.nonzero(spec_pitch)[0])
    spec_pitch = scipy_interp.pchip(spec_voiced,
                                    spec_pitch[spec_voiced])(range(pitch.nframes))

    spec_pitch = lfilter(np.ones((3))/3, 1.0, spec_pitch)

    spec_pitch[0] = spec_pitch[2]
    spec_pitch[1] = spec_pitch[3]

    return spec_pitch, pitch_std

"""
Temporal pitch tracking.
Corresponds to the tm_trk.m file.
"""
def time_track(signal, spec_pitch, pitch_std, pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    tda_frame_length = int(parameters['tda_frame_length']*signal.fs/1000)
    tda_noverlap = tda_frame_length-pitch.frame_jump
    tda_nframes = int((len(signal.data)-tda_noverlap)/pitch.frame_jump)

    len_spectral = len(spec_pitch)
    if tda_nframes < len_spectral:
        spec_pitch = spec_pitch[:tda_nframes]
    elif tda_nframes > len_spectral:
        tda_nframes = len_spectral

    merit_boost = parameters['merit_boost']
    maxcands = parameters['nccf_maxcands']
    freq_thresh = 5.0*pitch_std

    spec_range = np.maximum(spec_pitch-2.0*pitch_std, parameters['f0_min'])
    spec_range = np.vstack((spec_range,
                         np.minimum(spec_pitch+2.0*pitch_std, parameters['f0_max'])))

    time_pitch = np.zeros((maxcands, tda_nframes))
    time_merit = np.zeros((maxcands, tda_nframes))

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    data = np.zeros((signal.size))  #Needs other array, otherwise stride and
    data[:] = signal.filtered       #windowing will modify signal.filtered
    signal_frames = stride_matrix(data, tda_nframes,tda_frame_length,
                                  pitch.frame_jump)
    for frame in range(tda_nframes):
        lag_min0 = (int(np.fix(signal.new_fs/spec_range[1, frame])) -
                                    int(np.fix(parameters['nccf_pwidth']/2.0)))
        lag_max0 = (int(np.fix(signal.new_fs/spec_range[0, frame])) +
                                    int(np.fix(parameters['nccf_pwidth']/2.0)))

        phi = crs_corr(signal_frames[frame, :], lag_min0, lag_max0)
        time_pitch[:, frame], time_merit[:, frame] = \
            cmp_rate(phi, signal.new_fs, maxcands, lag_min0, lag_max0, parameters)

    diff = np.abs(time_pitch - spec_pitch)
    match1 = (diff < freq_thresh)
    match = ((1 - diff/freq_thresh) * match1)
    time_merit = (((1+merit_boost)*time_merit) * match)

    return time_pitch, time_merit

"""
Refines pitch candidates obtained from NCCF using spectral pitch track and
NLFER energy information.
Corresponds to the refine.m file.
"""
def refine(time_pitch1, time_merit1, time_pitch2, time_merit2, spec_pitch,
           pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    nlfer_thresh2 = parameters['nlfer_thresh2']
    merit_pivot = parameters['merit_pivot']

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    time_pitch = np.append(time_pitch1, time_pitch2, 0)
    time_merit = np.append(time_merit1, time_merit2, 0)
    maxcands = time_pitch.shape[0]

    idx = np.argsort(-time_merit, axis=0)
    time_merit.sort(axis=0)
    time_merit[:, :] = time_merit[::-1,:]

    time_pitch = time_pitch[idx, range(pitch.nframes)]

    best_pitch = medfilt(time_pitch[0, :], parameters['median_value'])*pitch.vuv

    idx1 = pitch.energy <= nlfer_thresh2
    idx2 = (pitch.energy > nlfer_thresh2) & (time_pitch[0, :] > 0)
    idx3 = (pitch.energy > nlfer_thresh2) & (time_pitch[0, :] <= 0)
    merit_mat = (time_pitch[1:maxcands-1, :] == 0) & idx2
    merit_mat = np.insert(merit_mat, [0, maxcands-2],
                          np.zeros((1, pitch.nframes), dtype=bool), 0)

    time_pitch[:, idx1] = 0
    time_merit[:, idx1] = merit_pivot

    time_pitch[maxcands-1, idx2] = 0.0
    time_merit[maxcands-1, idx2] = 1.0-time_merit[0, idx2]
    time_merit[merit_mat] = 0.0

    time_pitch[0, idx3] = spec_pitch[idx3]
    time_merit[0, idx3] = np.minimum(1, pitch.energy[idx3]/2.0)
    time_pitch[1:maxcands, idx3] = 0.0
    time_merit[1:maxcands, idx3] = 1.0-time_merit[0, idx3]

    time_pitch[maxcands-2, :] = best_pitch
    non_zero_frames = best_pitch > 0.0
    time_merit[maxcands-2, non_zero_frames] = time_merit[0, non_zero_frames]
    time_merit[maxcands-2, ~(non_zero_frames)] = 1.0-np.minimum(1,
                                       pitch.energy[~(non_zero_frames)]/2.0)

    time_pitch[maxcands-3, :] = spec_pitch
    time_merit[maxcands-3, :] = pitch.energy/5.0

    return time_pitch, time_merit


"""
Dynamic programming used to compute local and transition cost matrices,
enabling the lowest cost tracking of pitch candidates.
It uses NFLER from the spectrogram and the highly robust spectral F0 track,
plus the merits, for computation of the cost matrices.
Corresponds to the dynamic.m file.
"""
def dynamic(ref_pitch, ref_merit, pitch, parameters):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    num_cands = ref_pitch.shape[0]
    best_pitch = ref_pitch[num_cands-2, :]
    mean_pitch = np.mean(best_pitch[best_pitch > 0])

    dp_w1 = parameters['dp_w1']
    dp_w2 = parameters['dp_w2']
    dp_w3 = parameters['dp_w3']
    dp_w4 = parameters['dp_w4']

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    local_cost = 1 - ref_merit
    trans_cmatrix = np.ones((num_cands, num_cands, pitch.nframes))

    ref_mat1 = np.zeros((num_cands, num_cands, pitch.nframes))
    ref_mat2 = np.zeros((num_cands, num_cands, pitch.nframes))
    idx_mat1 = np.zeros((num_cands, num_cands, pitch.nframes), dtype=bool)
    idx_mat2 = np.zeros((num_cands, num_cands, pitch.nframes), dtype=bool)
    idx_mat3 = np.zeros((num_cands, num_cands, pitch.nframes), dtype=bool)

    ref_mat1[:, :, 1:] = np.tile(ref_pitch[:, 1:].reshape(1, num_cands,
                        pitch.nframes-1), (num_cands, 1, 1))
    ref_mat2[:, :, 1:] = np.tile(ref_pitch[:, :-1].reshape(num_cands, 1,
                        pitch.nframes-1), (1, num_cands, 1))

    idx_mat1[:, :, 1:] = (ref_mat1[:, :, 1:] > 0) & (ref_mat2[:, :, 1:] > 0)
    idx_mat2[:, :, 1:] = (((ref_mat1[:, :, 1:] == 0) & (ref_mat2[:, :, 1:] > 0)) |
                       ((ref_mat1[:, :, 1:] > 0) & (ref_mat2[:, :, 1:] == 0)))
    idx_mat3[:, :, 1:] = (ref_mat1[:, :, 1:] == 0) & (ref_mat2[:, :, 1:] == 0)

    mat1_values = np.abs(ref_mat1-ref_mat2)/mean_pitch
    benefit2 = np.insert(np.minimum(1, abs(pitch.energy[:-1]-pitch.energy[1:])),
                         0, 0)
    benefit2 = np.tile(benefit2, (num_cands, num_cands, 1))

    trans_cmatrix[idx_mat1] = dp_w1*mat1_values[idx_mat1]
    trans_cmatrix[idx_mat2] = dp_w2*(1-benefit2[idx_mat2])
    trans_cmatrix[idx_mat3] = dp_w3

    trans_cmatrix = trans_cmatrix/dp_w4
    path = path1(local_cost, trans_cmatrix, num_cands, pitch.nframes)
    final_pitch = ref_pitch[path, range(pitch.nframes)]

    return final_pitch

"""
--------------------------------------------
                Auxiliary functions.
--------------------------------------------
"""

"""
Computes peaks in a frequency domain function associated with the peaks found
in each frame based on the correlation sequence.
Corresponds to the peaks.m file.
"""
def peaks(data, delta, maxpeaks, parameters):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    PEAK_THRESH1 = parameters['shc_thresh1']
    PEAK_THRESH2 = parameters['shc_thresh2']

    epsilon = .00000000000001

    width = int(np.fix(parameters['shc_pwidth']/delta))
    if not(float(width) % 2):
        width = width + 1

    center = int(np.ceil(width/2))

    min_lag = int(np.fix(parameters['f0_min']/delta - center))
    max_lag = int(np.fix(parameters['f0_max']/delta + center))

    if (min_lag < 1):
        min_lag = 1
        print('Min_lag is too low and adjusted ({}).'.format(min_lag))

    if max_lag > (len(data) - width):
        max_lag = len(data) - width
        print('Max_lag is too high and adjusted ({}).'.format(max_lag))

    pitch = np.zeros((maxpeaks))
    merit = np.zeros((maxpeaks))

    #---------------------------------------------------------------
    # Main routine.
    #---------------------------------------------------------------
    max_data = max(data[min_lag:max_lag+1])

    if (max_data > epsilon):
        data = data/max_data

    avg_data = np.mean(data[min_lag:max_lag+1])

    if (avg_data > 1/PEAK_THRESH1):
        pitch = np.zeros((maxpeaks))
        merit = np.ones((maxpeaks))
        return pitch, merit

    #---------------------------------------------------------------
    #Step1 (this step was implemented differently than in original version)
    #---------------------------------------------------------------
    numpeaks = 0
    vec_back = (data[min_lag+center+1:max_lag-center+1] >
                                            data[min_lag+center:max_lag-center])
    vec_forw = (data[min_lag+center+1:max_lag-center+1] >
                                        data[min_lag+center+2:max_lag-center+2])
    above_thresh = (data[min_lag+center+1:max_lag-center+1] >
                                        PEAK_THRESH2*avg_data)
    peaks = np.logical_and(np.logical_and(vec_back, vec_forw), above_thresh)

    for n in (peaks.ravel().nonzero()[0]+min_lag+center+1).tolist():
        if np.argmax(data[n-center:n+center+1]) == center:
            if numpeaks >= maxpeaks:
                pitch = np.append(pitch, np.zeros((1)))
                merit = np.append(merit, np.zeros((1)))

            pitch[numpeaks] = float(n)*delta
            merit[numpeaks] = data[n]
            numpeaks += 1

    #---------------------------------------------------------------
    #Step2
    #---------------------------------------------------------------
    if (max(merit)/avg_data < PEAK_THRESH1):
        pitch = np.zeros((maxpeaks))
        merit = np.ones((maxpeaks))
        return pitch, merit

    #---------------------------------------------------------------
    #Step3
    #---------------------------------------------------------------
    idx = (-merit).ravel().argsort().tolist()
    merit = merit[idx]
    pitch = pitch[idx]

    numpeaks = min(numpeaks, maxpeaks)
    pitch = np.append(pitch[:numpeaks], np.zeros((maxpeaks-numpeaks)))
    merit = np.append(merit[:numpeaks], np.zeros((maxpeaks-numpeaks)))

    #---------------------------------------------------------------
    #Step4
    #---------------------------------------------------------------

    if (numpeaks > 0):
        # The first two "if pitch[0]" statements seem to had been deprecated in
        # the original YAAPT Matlab code, so they may be removed here as well.
        if (pitch[0] > parameters['f0_double']):
            numpeaks = min(numpeaks+1, maxpeaks)
            pitch[numpeaks-1] = pitch[0]/2.0
            merit[numpeaks-1] = parameters['merit_extra']

        if (pitch[0] < parameters['f0_half']):
            numpeaks = min(numpeaks+1, maxpeaks)
            pitch[numpeaks-1] = pitch[0]*2.0
            merit[numpeaks-1] = parameters['merit_extra']

        if (numpeaks < maxpeaks):
            pitch[numpeaks:maxpeaks] = pitch[0]
            merit[numpeaks:maxpeaks] = merit[0]

    else:
        pitch = np.zeros((maxpeaks))
        merit = np.ones((maxpeaks))

    return np.transpose(pitch), np.transpose(merit)

"""
Dynamic programming used to compute local and transition cost matrices,
enabling the lowest cost tracking of pitch candidates.
It uses NFLER from the spectrogram and the highly robust spectral F0 track,
plus the merits, for computation of the cost matrices.
Corresponds to the dynamic5.m file.
"""
def dynamic5(pitch_array, merit_array, k1, f0_min):

    num_cand = pitch_array.shape[0]
    num_frames = pitch_array.shape[1]

    local = 1-merit_array
    trans = np.zeros((num_cand, num_cand, num_frames))

    trans[:, :, 1:] = abs(pitch_array[:, 1:].reshape(1, num_cand, num_frames-1) -
                    pitch_array[:, :-1].reshape(num_cand, 1, num_frames-1))/f0_min
    trans[:, :, 1:] = 0.05*trans[:, :, 1:] + trans[:, :, 1:]**2

    trans = k1*trans
    path = path1(local, trans, num_cand, num_frames)

    final_pitch = pitch_array[path, range(num_frames)]

    return final_pitch

"""
Finds the optimal path with the lowest cost if two matrice(Local cost matrix
and Transition cost) are given.
Corresponds to the path1.m file.
"""
def path1(local, trans, n_lin, n_col):

# Apparently the following lines are somehow kind of useless.
# Therefore, I removed them in the version 1.0.3.

#    if n_lin >= 100:
#        print 'Stop in Dynamic due to M>100'
#        raise KeyboardInterrupt
#
#    if n_col >= 1000:
#        print 'Stop in Dynamic due to N>1000'
#        raise KeyboardInterrupt

    PRED = np.zeros((n_lin, n_col), dtype=int)
    P = np.ones((n_col), dtype=int)
    p_small = np.zeros((n_col), dtype=int)

    PCOST = np.zeros((n_lin))
    CCOST = np.zeros((n_lin))
    PCOST = local[:, 0]

    for I in range(1, n_col):

        aux_matrix = PCOST+np.transpose(trans[:, :, I])
        K = n_lin-np.argmin(aux_matrix[:, ::-1], axis=1)-1
        PRED[:, I] = K
        CCOST = PCOST[K]+trans[K, range(n_lin), I]

        assert CCOST.any() < 1.0E+30, 'CCOST>1.0E+30, Stop in Dynamic'
        CCOST = CCOST+local[:, I]

        PCOST[:] = CCOST
        J = n_lin - np.argmin(CCOST[::-1])-1
        p_small[I] = J

    P[-1] = p_small[-1]

    for I in range(n_col-2, -1, -1):
        P[I] = PRED[P[I+1], I+1]

    return P

"""
Computes the NCCF (Normalized cross correlation Function) sequence based on
the RAPT algorithm discussed by DAVID TALKIN.
Corresponds to the crs_corr.m file.
"""
def crs_corr(data, lag_min, lag_max):

    eps1 = 0.0
    data_len = len(data)
    N = data_len-lag_max

    error_str = 'ERROR: Negative index in the cross correlation calculation of '
    error_str += 'the pYAAPT time domain analysis. Please try to increase the '
    error_str += 'value of the "tda_frame_length" parameter.'
    assert N>0, error_str

    phi = np.zeros((data_len))
    data -= np.mean(data)
    x_j = data[0:N]
    x_jr = data[lag_min:lag_max+N]
    p = np.dot(x_j, x_j)

    x_jr_matrix = stride_matrix(x_jr, lag_max-lag_min, N, 1)

    formula_nume = np.dot(x_jr_matrix, x_j)
    formula_denom = np.sum(x_jr_matrix*x_jr_matrix, axis=1)*p + eps1

    phi[lag_min:lag_max] = formula_nume/np.sqrt(formula_denom)

    return phi

"""
Computes pitch estimates and the corresponding merit values associated with the
peaks found in each frame based on the correlation sequence.
Corresponds to the cmp_rate.m file.
"""
def cmp_rate(phi, fs, maxcands, lag_min, lag_max, parameters):

    #---------------------------------------------------------------
    # Set parameters.
    #---------------------------------------------------------------
    width = parameters['nccf_pwidth']
    center = int(np.fix(width/2.0))
    merit_thresh1 = parameters['nccf_thresh1']
    merit_thresh2 = parameters['nccf_thresh2']

    numpeaks = 0
    pitch = np.zeros((maxcands))
    merit = np.zeros((maxcands))

    #---------------------------------------------------------------
    # Main routine.
    #(this step was implemented differently than in original version)
    #---------------------------------------------------------------
    vec_back = (phi[lag_min+center:lag_max-center+1] >
                                            phi[lag_min+center-1:lag_max-center])
    vec_forw = (phi[lag_min+center:lag_max-center+1] >
                                        phi[lag_min+center+1:lag_max-center+2])
    above_thresh = phi[lag_min+center:lag_max-center+1] > merit_thresh1
    peaks = np.logical_and(np.logical_and(vec_back, vec_forw), above_thresh)

    peaks = (peaks.ravel().nonzero()[0]+lag_min+center).tolist()

    if np.amax(phi) > merit_thresh2 and len(peaks) > 0:
        max_point = peaks[np.argmax(phi[peaks])]
        pitch[numpeaks] = fs/float(max_point+1)
        merit[numpeaks] = np.amax(phi[peaks])
        numpeaks += 1
    else:
        for n in peaks:
            if np.argmax(phi[n-center:n+center+1]) == center:
                try:
                    pitch[numpeaks] = fs/float(n+1)
                    merit[numpeaks] = phi[n]
                except:
                    pitch = np.hstack((pitch, fs/float(n+1)))
                    merit = np.hstack((merit, phi[n]))
                numpeaks += 1

    #---------------------------------------------------------------
    # Sort the results.
    #---------------------------------------------------------------
    idx = (-merit).ravel().argsort().tolist()
    merit = merit[idx[:maxcands]]
    pitch = pitch[idx[:maxcands]]

    if (np.amax(merit) > 1.0):
        merit = merit/np.amax(merit)

    return pitch, merit

"""
--------------------------------------------
                Extra functions.
--------------------------------------------
"""

def stride_matrix(vector, n_lin, n_col, hop):

    data_matrix = stride_tricks.as_strided(vector, shape=(n_lin, n_col),
                        strides=(vector.strides[0]*hop, vector.strides[0]))

    return data_matrix
