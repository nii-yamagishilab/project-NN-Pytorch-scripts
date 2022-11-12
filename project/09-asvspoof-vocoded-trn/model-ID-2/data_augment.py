#!/usr/bin/env python
"""Functions for data augmentation

These functions are written using Numpy.
They should be used before data are casted into torch.tensor.

For example, use them in config.py:input_trans_fns or output_trans_fns


"""
import os
import sys
import numpy as np

from core_scripts.other_tools import debug as nii_debug
from core_scripts.data_io import wav_tools as nii_wav_tools
from core_scripts.data_io import wav_augmentation as nii_wav_aug

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

def wav_aug_wrapper(input_data, 
                    name_info, 
                    wav_samp_rate, 
                    length):
    # use this API to randomly trim the input length
    input_data = nii_wav_aug.batch_pad_for_multiview(
        [input_data], wav_samp_rate, length, random_trim_nosil=True)[0]
    return input_data

def wav_aug_wrapper_val(input_data, 
                        name_info, 
                        wav_samp_rate, 
                        length):
    # use this API to randomly trim the input length
    input_data = nii_wav_aug.batch_pad_for_multiview(
        [input_data], wav_samp_rate, length, random_trim_nosil=False)[0]
    return input_data

if __name__ == "__main__":
    print("Tools for data augmentation")
