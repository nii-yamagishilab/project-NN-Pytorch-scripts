#!/usr/bin/env python
"""
Functions for evaluation - music data

This is just wrappers around MIR_eval http://craffel.github.io/mir_eval/

"""
from __future__ import print_function
import os
import sys
import numpy as np

try:
    import mir_eval
except ModuleNotFoundError:
    print("Please install mir_eval: http://craffel.github.io/mir_eval/")
    print("Anaconda: https://anaconda.org/conda-forge/mir_eval")
    print("Pip: https://pypi.org/project/mir_eval/")
    sys.exit(1)

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


###############
# Functions to evaluation melody accuracy

def eva_music(est_f0, ref_f0, resolution=0.012, tolerence=50):
    """ Evaluating estimated F0 using mir_eval metrics.

    Parameters
    ----------
    est_f0 : np.array, in shape (N)
        estimated F0 (Hz)
    ref_f0 : np.array, in shape (N)
        reference F0 (Hz)
    resolution : float (s)
        corresponding to frame-shift of the F0
    tolerence : int
        tolerence for cent evaluation
    
    Returns
    -------
    rpa : float
        raw pitch accuracy
    rca : float
        raw chroma accuracy
    """ 

    # generate time sequences
    def _time_seq(f0_seq, reso):
        time_seq = np.arange(f0_seq.shape[0]) * reso
        return time_seq
    
    est_time = _time_seq(est_f0, resolution)
    ref_time = _time_seq(ref_f0, resolution)
    
    # format change
    (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(
        ref_time, ref_f0, est_time, est_f0)
    
    # evaluation
    rpa = mir_eval.melody.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
    rca = mir_eval.melody.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)

    return rpa, rca
