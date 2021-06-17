# -*- coding: utf-8 -*-
"""
This package provides the tools necessary to decompose the voiced part of a
speech signal into its modulated components, aka AM-FM decomposition. This
designation is used due the fact that, in this method, the signal is modeled as
a sum of amplitude- and frequency-modulated components. The goal is to overcome
the drawbacks from Fourier-alike techniques, e.g. SFFT, wavelets, etc, which
are limited in the time-frequency analysis by the so-called Heisenberg-Gabor
inequality.

The algorithms here implemented were the QHM (Quasi-Harmonic Model), and its
upgrades, aQHM (adaptive Quasi-Harmonic Model) and eaQHM (extended adaptive
Quasi-Harmonic Model). Their formulation can be found at references [1-3].

USAGE:
    Please refer to the documentation for examples.

References:
    [1] Y. Pantazis, “Decomposition of AM-FM signals with applications in
        speech processing”, PhD Thesis, University of Creta, 2010.

    [2] Y. Pantazis, O. Rosec and Y. Stylianou, “Adaptive AM-FM signal
        decomposition with application to speech analysis”, IEEE Transactions
        on Audio, Speech and Language Processing, vol. 19, n 2, 2011.

    [3] G. P. Kafentzis, Y. Pantazis, O. Rosec and Y. Stylianou, “An extension
        of the adaptive quasi-harmonic model”, em IEEE International Conference
        on Acoustics, Speech and Signal Processing (ICASSP), 2012.

Version 1.0.8.1
09/Jul/2018 Bernardo J.B. Schmitt - bernardo.jb.schmitt@gmail.com
"""

import numpy as np
import scipy

"""
--------------------------------------------
                Classes.
--------------------------------------------
"""

"""
Creates a single component object.
"""

class ComponentObj(object):

    def __init__(self, H, harm):
        self.mag = H[harm, 0, :]
        self.phase = H[harm, 1, :]
        self.freq = H[harm, 2, :]

    """
    Synthsize the modulated component by using the extracted magnitude and
    phase.
    """

    def synthesize(self):
        self.signal = 2*self.mag*np.cos(self.phase)


"""
Creates the output signal object (which, in its turn, is formed by n_harm
modulated components).
"""

class ModulatedSign(object):

    def __init__(self, n_harm, file_size, fs, phase_tech='phase'):
        self.n_harm = n_harm
        self.size = file_size
        self.fs = fs
        self.H = np.zeros((self.n_harm, 3, self.size))
        self.harmonics = [ComponentObj(self.H, i) for i in range(self.n_harm)]
        self.error = np.zeros(self.size)
        self.phase_tech = phase_tech

    """
    Updates the 3-dimension array H, which stores the magnitude, phase and
    frequency values from all components. Its first dimension refers to the
    n_harm components, the second to the three composing parameters (where 0
    stands for the magnitude, 1 for the phase and 2 for the frequency) and the
    third dimension to the temporal axis.
    """

    def update_values(self, a, freq, frame):
        self.H[:, 0, frame] = np.abs(a)
        self.H[:, 1, frame] = np.angle(a)
        self.H[:, 2, frame] = freq

    """
    Interpolate the parameters values when the extraction is not performed
    sample-by-sample. While the interpolation from magnitude and frequency
    is pretty straightforward, the phase one is not. Therefore, references
    [1,2] present a solution for this problem.
    """

    def interpolate_samp(self, samp_frames, pitch_track):

        # Interpolation from magnitude and frequency.

        for idx, func in [(0, 'linear'), (2, 'cubic')]:
            f = scipy.interpolate.interp1d(samp_frames,
                                       self.H[:, idx, samp_frames], kind=func)
            self.H[:, idx, np.nonzero(pitch_track)[0]] = f(
                                                    np.nonzero(pitch_track)[0])

        # Interpolation from phase.

        step = samp_frames[1]-samp_frames[0]
        sin_f = np.cumsum(np.sin(np.pi*np.arange(1, step)/step)).reshape(
                                                                    1, step-1)
        for idx, frame in np.ndenumerate(samp_frames[1:]):
            if frame-samp_frames[idx] <= step:
                cum_phase = np.cumsum(self.H[:, 2, samp_frames[idx]+1:frame+1],
                                    axis=1)*2*np.pi
                bad_phase = cum_phase[:, -1]+self.H[:, 1, samp_frames[idx]]
                M = np.around(np.abs(self.H[:, 1, frame]-bad_phase)/(2*np.pi))
                if frame-samp_frames[idx] < step:
                    end_step = frame-samp_frames[idx]
                    func = np.cumsum(np.sin(np.pi*np.arange(1, end_step) /
                                            end_step)).reshape(1, end_step-1)
                else:
                    func = sin_f

                r_vec = (np.pi*(self.H[:, 1, frame]+2*np.pi*M-bad_phase) /
                        (2*(frame-samp_frames[idx]))).reshape(self.n_harm, 1)

                new_phase = cum_phase[:, :-1]+r_vec*func + \
                        self.H[:, 1, samp_frames[idx]].reshape(self.n_harm, 1)
                self.H[:, 1, samp_frames[idx]+1:frame] = ((new_phase + np.pi) %
                                                            (2*np.pi)-np.pi)

    """
    Synthesize the final signal by initially creating each modulated component
    and then summing all of them.
    """

    def synthesize(self, N=None):
        if N is None:
            N = self.n_harm
        [self.harmonics[i].synthesize()
                            for i in range(N)]
        self.signal = sum([self.harmonics[i].signal
                            for i in range(self.n_harm)])

    """
    Calculates the SRER (Signal-to-Reconstruction Error Ratio) for the
    synthesized signal.
    """

    def srer(self, orig_signal, pitch_track):
        self.SRER = 20*np.log10(np.std(orig_signal[np.nonzero(pitch_track)[0]]) /
                    np.std(orig_signal[np.nonzero(pitch_track)[0]] -
                    self.signal[np.nonzero(pitch_track)[0]]))

    """
    Extrapolates the phase at the border of the voiced frames by integrating
    the edge frequency value. This procedure is necessary for posterior aQHM
    calculations. Additionally, the method allows the replacement of the
    extracted phase by the cumulative frequency. The objective is to provide
    smoother bases for further aQHM and eaQHM calculations. Normally this is
    not necessary, since that the interpolation process already smooths the
    phase vector. But in a sample-by-sample extraction case, this substitution
    is very helpful to avoid the degradation of aQHM and eaQHM performance
    due the phase wild behaviour.
    """

    def phase_edges(self, edges, window):

        # Selects whether the phase itself or the cummulative frequency will be
        # used.
        if self.phase_tech is 'phase':
            self.extrap_phase = np.unwrap(self.H[:, 1, :])

        elif self.phase_tech is 'freq':
            delta_phase = self.H[:, 1, edges[0]+1] - \
                                            self.H[:, 2, edges[0]+1]*2*np.pi
            self.extrap_phase = np.cumsum(self.H[:, 2, :], axis=1)*2*np.pi + \
                                            delta_phase.reshape(self.n_harm, 1)

        # Extrapolate the phase edges.
        n_beg = -window.half_len_vec[::-1][:-1].reshape(1, window.N)
        n_end = window.half_len_vec[1:].reshape(1, window.N)

        for beg, end in zip(edges[::2], edges[1::2]):

            old_phase = self.extrap_phase[:, beg+1].reshape(self.n_harm, 1)
            freq = self.H[:, 2, beg+1].reshape(self.n_harm, 1)
            self.extrap_phase[:, beg-window.N+1:beg+1] = \
                                                2*np.pi*freq*n_beg+old_phase

            old_phase = self.extrap_phase[:, end].reshape(self.n_harm, 1)
            freq = self.H[:, 2, end].reshape(self.n_harm, 1)
            self.extrap_phase[:, end+1:end+window.N+1] = \
                                                2*np.pi*freq*n_end+old_phase


"""
Creates the sample window object.
"""

class SampleWindow(object):

    def __init__(self, window_duration, fs):
        self.dur = window_duration         # in seconds
        self.length = int(self.dur*fs+1)
        if not self.length %2:
            self.length -= 1
        self.data = np.hamming(self.length)
        self.data2 = self.data**2
        self.N = int(self.dur*fs/2)
        self.half_len_vec = np.arange(self.N+1)
        self.len_vec = np.arange(-self.N, self.N+1)

        self.a0 = 0.54**2 + (0.46**2)/2
        self.a1 = 0.54*0.46
        self.a2 = (0.46**2)/4

        self.R0_diag = R_eq(0, g0, self)
        self.R2_diag = sum(self.data2*(self.len_vec**2))


"""
--------------------------------------------
               Main Functions.
--------------------------------------------
"""

"""
Main QHM function.
"""

def qhm(signal, pitch, window, samp_jump=None, N_iter=1, phase_tech='phase'):

    return HM_run(qhm_iteration, signal, pitch, window, samp_jump, N_iter,
                  phase_tech)

"""
Main aQHM function.
"""

def aqhm(signal, previous_HM, pitch, window, samp_jump=None, N_iter=1,
         N_runs=float('Inf'), phase_tech='phase', eaQHM_flag=False):

    count = 1
    outflag = False

    while outflag is False:
        func_options = [previous_HM, eaQHM_flag, 0]
        HM = HM_run(aqhm_iteration, signal, pitch, window, samp_jump, N_iter,
                    phase_tech, func_options)
        if count == 1:
            previous_HM = HM
        elif (count > 1 and HM.SRER > previous_HM.SRER):
            previous_HM = HM
        else:
            outflag = True

        count += 1

        if count > N_runs:
            outflag = True

    return previous_HM

"""
Main eaQHM function (which in fact varies very few from the aQHM).
"""

def eaqhm(signal, previous_HM, pitch, window, samp_jump=None, N_iter=1,
          N_runs=float('Inf'), phase_tech='phase'):

    return aqhm(signal, previous_HM, pitch, window, samp_jump, N_iter, N_runs,
                phase_tech, eaQHM_flag=True)

"""
Parser for the three algorithms.
"""

def HM_run(func, signal, pitch, window, samp_jump=None, N_iter=1,
           phase_tech='phase', func_options=None):

    # Creates the output signal object and the dummy frequency vector.
    HM = ModulatedSign(signal.n_harm, signal.size, signal.fs, phase_tech)
    freq = np.zeros(signal.n_harm)

    # Selects whether the extration will be performed with temporal jumps or
    # not.
    if samp_jump is None:
        voiced_frames = np.nonzero(pitch.values)[0]
    else:
        jump = int(np.fix(max(samp_jump*signal.fs, 1.0)))
        voiced_frames = np.array([], dtype=int)
        for beg, end in zip(pitch.edges[::2], pitch.edges[1::2]):
            voiced_frames = np.append(voiced_frames, np.arange(
                                                        beg+1, end-1, jump))
            voiced_frames = np.append(voiced_frames, end)

    # Run the algorithm in the selected voiced frames.
    for frame in voiced_frames:
        # Uses the pitch value and the harmonic definition f_k = k*f0 to create
        # a frequency reference vector, which is employed to keep each component
        # within a frquency band and thus, avoiding least-squares instability.
        f0_ref = pitch.values[frame]*np.arange(1, signal.n_harm+1)/signal.fs

        # Set some algorithm options.
        if func is qhm_iteration:
            if frame-1 in pitch.edges[::2]:
                freq[:] = f0_ref
            func_options = freq
        elif func is aqhm_iteration:
            func_options[2] = frame

        # Core algorithm function.
        coef, freq, HM.error[frame] = func(
                                signal.data[frame-window.N:frame+window.N+1],
                                f0_ref, window, signal.fs, 20.0, func_options,
                                N_iter)

        # Updates frame parameter values in the 3-dimension storage array H.
        HM.update_values(coef[:signal.n_harm], freq, frame)

    # If the extraction was performed with temporal jumps, interpolate the
    # results.
    if samp_jump is not None:
        HM.interpolate_samp(voiced_frames, pitch.values)
    HM.synthesize()
    HM.srer(signal.data, pitch.values)
    HM.phase_edges(pitch.edges, window)

    return HM

"""
Core QHM function.
"""

def qhm_iteration(data, f0_ref, window, fs, max_step, freq, N_iter=1):

    # Initialize and allocate variables.
    K = len(freq)
    coef = np.zeros((2*K))

    E = np.ones((window.length, 2*K), dtype=complex)
    E = exp_matrix(E, freq, window, K)
    E_windowed = np.ones((window.length, 2*K), dtype=complex)

    windowed_data = (window.data*data).reshape(window.length, 1)

    # Run the QHM algorithm N_iter times.
    for k in range(N_iter):
        # Calculate the a and b coeficients via least-squares.
        coef = least_squares(E, E_windowed, windowed_data, window, K)

        # Set a magnitude reference, which is used to detect and supress
        # erroneous magnitude spikes.
        mag_ref = np.abs(coef[0])

        # Updates the frequency values.
        freq, ro = freq_correction(coef[:K], coef[K:], freq, f0_ref, mag_ref, K,
                                   max_step, fs)

        # Updates the complex exponentials matrix.
        E = exp_matrix(E, freq, window, K)

    # Compute the final coefficients values.
    coef = least_squares(E, E_windowed, windowed_data, window, K)

    # This part is a workaround not present in the original references [1-3].
    # It was created to detect and supress erroneous magnitude spikes, which
    # degradate the final synthsized signal and consequently, its SRER.
    # Alternatively, the magnitude signals could be smoothed after extraction.
    # For more details, check the README file.
    cond = (np.abs(coef[:K]) < 5.5*np.abs(coef[0]))
    if not cond.all():
        freq[~cond] = f0_ref[~cond]

        # Updates the complex exponentials matrix with the modified frequencies.
        E = exp_matrix(E, freq, window, K)

        # Recalculate the final coefficients.
        coef = least_squares(E, E_windowed, windowed_data, window, K)

    # Calculate the mean squared error between the original frame and the
    # synthesized one.
    err = error_calc(windowed_data, E, coef, window)
    return coef, freq, err

"""
Core aQHM and eaQHM function.
"""

def aqhm_iteration(data, f0_ref, window, fs, max_step, func_options,
                   N_iter=1):

    # Initialize and allocate variables.
    previous_HM = func_options[0]
    eaQHM_flag = func_options[1]
    frame = func_options[2]

    freq = previous_HM.H[:, 2, frame]
    windowed_data = (window.data*data).reshape(window.length, 1)

    # Set a magnitude reference, which is used to detect and supress
    # erroneous magnitude spikes.
    mag_ref = np.abs(previous_HM.H[0, 0, frame])

    # Ajust the phase frame.
    extrap_phase_center = previous_HM.extrap_phase[:, frame].reshape(
                                                        previous_HM.n_harm, 1)
    phase_frame = previous_HM.extrap_phase[:, frame-window.N:frame+window.N+1] - \
                    extrap_phase_center

    # Initialize the coefficients.
    coef = np.vstack((previous_HM.H[:, 0, frame].reshape(previous_HM.n_harm, 1) *
        np.exp(1j*extrap_phase_center), np.zeros((previous_HM.n_harm, 1))))[:, 0]

    # Initialize the matrices.
    E = np.ones((window.length, 2*previous_HM.n_harm), dtype=complex)
    E_ro = np.ones((window.length, 2*previous_HM.n_harm), dtype=complex)
    E_windowed = np.ones((window.length, 2*previous_HM.n_harm), dtype=complex)

    E[:, :previous_HM.n_harm] = np.exp(1j*phase_frame.T)

    # If the eaQHM algorithm was selected, ajust the exponential matrix with
    # the normalized magnitude.
    if eaQHM_flag:
        mag_center = previous_HM.H[:, 0, frame].reshape(previous_HM.n_harm, 1)
        mag_frame = previous_HM.H[:, 0, frame-window.N:frame+window.N+1] / \
                    mag_center
        E[:, :previous_HM.n_harm] = mag_frame.T*E[:, :previous_HM.n_harm]

    E[:, previous_HM.n_harm:] = E[:, :previous_HM.n_harm] * \
                                window.len_vec.reshape(window.length, 1)

    # Run the aQHM/eaQHM algorithm N_iter times.
    for k in range(N_iter):

        # Calculate the a and b coeficients via least-squares.
        coef = least_squares(E, E_windowed, windowed_data, window,
                             previous_HM.n_harm)

        # Updates the frequency values.
        freq, ro = freq_correction(coef[:previous_HM.n_harm],
                                   coef[previous_HM.n_harm:], freq, f0_ref,
                                   mag_ref, previous_HM.n_harm, max_step, fs)

        # Updates the complex exponentials matrix.
        E = E*exp_matrix(E_ro, ro/(2*np.pi), window, previous_HM.n_harm)

    # Compute the final coefficients values.
    coef = least_squares(E, E_windowed, windowed_data, window,
                         previous_HM.n_harm)

    # This part is a workaround not present in the original references [1-3].
    # It was created to detect and supress erroneous magnitude spikes, which
    # degradate the final synthsized signal and consequently, its SRER.
    # Alternatively, the magnitude signals could be smoothed after extraction.
    # For more details, check the README file.
    cond = (np.abs(coef[:previous_HM.n_harm]) < 5.5*mag_ref)
    if not cond.all():
        freq[~cond] = f0_ref[~cond]

        # Since that the troubling aQHM/eaQHM exponentials are degradating the
        # results, they are replaced by the QHM version, which is more stable.
        E[:, ~np.append(cond, cond)] = exp_matrix(E_ro, freq, window,
                              previous_HM.n_harm)[:, ~np.append(cond, cond)]

        # Recalculate the final coefficients.
        coef = least_squares(E, E_windowed, windowed_data, window,
                             previous_HM.n_harm)

    # Calculate the mean squared error between the original frame and the
    # synthsized one.
    err = error_calc(windowed_data, E, coef, window)

    return coef, freq, err

"""
--------------------------------------------
            Auxiliary Functions.
--------------------------------------------
"""

"""
Calculate the a and b coeficients via least-squares method.
"""

def least_squares(E, E_windowed, windowed_data, window, K):

    R = np.zeros((2*K, 2*K), dtype=complex)
    B = np.zeros((window.length, 1), dtype=complex)

    E_windowed[:, :] = E*window.data.reshape(window.length, 1)
    R = E_windowed.conj().T.dot(E_windowed)
    B = E_windowed.conj().T.dot(windowed_data)

    coef = np.linalg.solve(R, B)[:, 0]

    return coef

"""
Calculates the frequency mismatch and updates the frequency values.
"""

def freq_correction(a, b, freq, f0_ref, mag_ref, n_harm, max_step, fs):

    old_freq = np.zeros(n_harm)
    old_freq[:] = freq[:]

    ro = (a.real*b.imag-a.imag*b.real)/(np.abs(a)*np.abs(a))

    # If the mismatch is too high (>20Hz), the frequency update is satured to
    # this value. This avoids big fluctuations, which can spoil the algorithms
    # convergence as whole.
    over_ro = np.abs(ro) > max_step*2*np.pi/fs
    ro[over_ro] = np.sign(ro[over_ro])*(max_step*2*np.pi/fs)
    freq = freq+ro/(2*np.pi)

    # Checks whether each component frequency lies within its spectral band and
    # also checks whether there are magnitude spikes.
    cond = ((np.round(freq/f0_ref[0]) != np.arange(n_harm)+1) |
            (freq > 0.5) | (freq < 0) | (np.abs(a) > 5.5*mag_ref))

    freq[cond] = f0_ref[cond]

    return freq, (freq-old_freq)*(2*np.pi)

"""
Calculate the mean squared error between the original frame and the
synthsized one.
"""

def error_calc(windowed_data, E, coef, window):
    h = E.dot(coef)

    err = np.sum((windowed_data-2*h.real*window.data)**2)

    return err

"""
Mounts the complex exponentials matrix.
"""

def exp_matrix(E, freq, window, K):

    E[window.N+1:, :K] = np.exp(1j*np.pi*2*freq)
    E[window.N+1:, :K] = np.cumprod(E[window.N+1:, :K], axis=0)
    E[:window.N, :K] = np.conj(E[window.N+1:, :K][::-1, :])

    E[:, K:] = E[:, :K]*window.len_vec.reshape(window.length, 1)

    return E

"""
Some side functions found in reference [2].
"""

def g0(x, N):
    if x != 0:
        return np.sin((2*N+1)*x/2)/np.sin(x/2)
    else:
        return 2*N+1

def g1(x, N):
    if x != 0:
        return 1j*((np.sin(N*x)/(2*np.sin(x/2)**2)) -
                   N*(np.cos((2*N+1)*x/2)/np.sin(x/2)))
    else:
        return 0

def R_eq(delta_f, func, window):
    return (window.a0*func(2*np.pi*delta_f, window.N) +
            func(2*np.pi*(delta_f+1./(2*window.N)), window.N)*window.a1 +
            func(2*np.pi*(delta_f-1./(2*window.N)), window.N)*window.a1 +
            func(2*np.pi*(delta_f+1./window.N), window.N)*window.a2 +
            func(2*np.pi*(delta_f-1./window.N), window.N)*window.a2)
