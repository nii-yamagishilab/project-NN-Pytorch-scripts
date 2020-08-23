#!/usr/bin/env python
"""
Simple script to test speed on CPU

"""

# At the begining, let's load packages 
from __future__ import absolute_import
from __future__ import print_function
import sys
import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import time

import tool_lib

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

if __name__ == "__main__":
    # input feature dim (80 dimension Mel-spec + 1 dimension F0)
    mel_dim = 80
    f0_dim = 1
    input_dim = mel_dim + f0_dim

    # output dimension = 1 for waveform
    output_dim = 1
    # sampling rate of waveform (Hz)
    sampling_rate = 16000
    # up-sampling rate of acoustic features (sampling_rate * frame_shift)
    feat_upsamp_rate = int(16000 * 0.005)

    # load the basic function blocks
    import data_models.pre_trained_hn_nsf.model as nii_nn_blocks
    
    # sampling rate and up-sampling rate have been written
    # in data_models/pre_trained_hn_nsf/model.py for this tutorial.
    # no need to provide them as arguments
    
    # declare the model
    hn_nsf_model = nii_nn_blocks.Model(input_dim, output_dim, None)

    # load pre-trained model
    device=torch.device("cpu")
    hn_nsf_model.to(device, dtype=torch.float32)
    checkpoint = torch.load("data_models/pre_trained_hn_nsf/trained_network.pt", map_location="cpu")
    hn_nsf_model.load_state_dict(checkpoint)


    # load mel and F0
    input_mel = tool_lib.read_raw_mat("data_models/acoustic_features/hn_nsf/slt_arctic_b0474.mfbsp", mel_dim)
    input_f0 = tool_lib.read_raw_mat("data_models/acoustic_features/hn_nsf/slt_arctic_b0474.f0", f0_dim)

    print("Input Mel shape:" + str(input_mel.shape))
    print("Input F0 shape:" + str(input_f0.shape))

    # compose the input tensor
    input_length = min([input_mel.shape[0], input_f0.shape[0]])
    input_tensor = torch.zeros(1, input_length, mel_dim + f0_dim, dtype=torch.float32)
    input_tensor[0, :, 0:mel_dim] = torch.tensor(input_mel[0:input_length, :])
    input_tensor[0, :, mel_dim:] = torch.tensor(input_f0[0:input_length]).unsqueeze(-1)
    print("Input data tensor shape:" + str(input_tensor.shape))


    #
    num_iter = 5

    print("Generate a waveform for %d times:" % (num_iter))
    time_start = time.time()
    hn_nsf_model.eval()
    with torch.no_grad():
        for idx in range(num_iter):
            output_waveform = hn_nsf_model(input_tensor)
            print("%d" % (idx), end=', ')
    time_end = time.time()

    print("\nGeneration done")
    output_waveform_array = output_waveform[0].numpy()
    output_duration = output_waveform_array.shape[0] / sampling_rate

    time_average = (time_end - time_start) / num_iter
    speed_per_s = output_waveform_array.shape[0] / time_average
    real_time_factor = time_average / output_duration
    print("Speed (waveform sampling points per second): %f" % (speed_per_s))
    print("Real time factor: %f" % (real_time_factor))


