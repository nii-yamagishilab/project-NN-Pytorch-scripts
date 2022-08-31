#!/usr/bin/env python
"""
model.py 

waveglow is defined in sandbox.block_waveglow.
This file wrapps waveglow inside model() so that it can be used by the script.
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import time
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

import core_scripts.other_tools.debug as nii_debug
import core_scripts.other_tools.display as nii_warn
import sandbox.block_nn as nii_nn

import sandbox.block_waveglow as nii_waveglow
import sandbox.util_frontend as nii_nn_frontend

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        #################
        ## must-have
        #################
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        self.input_dim = in_dim
        self.output_dim = out_dim
        
        # a flag for debugging (by default False)    
        self.model_debug = False
        
        #################
        ## model config
        #################        
        # waveform sampling rate
        self.sample_rate = prj_conf.wav_samp_rate
        # up-sample rate
        self.up_sample = prj_conf.input_reso[0]
        
        # configuration for WaveGlow
        # number of waveglow blocks
        self.num_waveglow_blocks = 3
        # number of flow steps in each block
        self.num_flow_steps_perblock = 4
        # number of wavenet layers in one flow step
        self.num_wn_blocks_perflow = 8
        # channel size of the dilated conv in wavenet layer
        self.num_wn_channel_size = 256
        # kernel size of the dilated conv in wavenet layer
        self.num_wn_conv_kernel = 3
        # use affine transformation? (if False, use a + b)
        self.flag_affine = True
        # dimension of the early extracted latent z
        self.early_z_feature_dim = 2
        # use legacy implementation of affine (default False)
        # the difference is the wavenet core to compute the affine
        # parameter. For more details, check sandbox.block_waveglow
        self.flag_affine_legacy_implementation = True

        self.m_waveglow = nii_waveglow.WaveGlow(
            in_dim, self.up_sample, 
            self.num_waveglow_blocks,
            self.num_flow_steps_perblock,
            self.num_wn_blocks_perflow, 
            self.num_wn_channel_size,
            self.num_wn_conv_kernel,
            self.flag_affine,
            self.early_z_feature_dim,
            self.flag_affine_legacy_implementation)

        # buffer for noise in inference
        self.bg_noise = None

        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
        """
        """
        if data_mean_std is not None:
            in_m = torch.from_numpy(data_mean_std[0])
            in_s = torch.from_numpy(data_mean_std[1])
            out_m = torch.from_numpy(data_mean_std[2])
            out_s = torch.from_numpy(data_mean_std[3])
            if in_m.shape[0] != in_dim or in_s.shape[0] != in_dim:
                print("Input dim: {:d}".format(in_dim))
                print("Mean dim: {:d}".format(in_m.shape[0]))
                print("Std dim: {:d}".format(in_s.shape[0]))
                print("Input dimension incompatible")
                sys.exit(1)
            if out_m.shape[0] != out_dim or out_s.shape[0] != out_dim:
                print("Output dim: {:d}".format(out_dim))
                print("Mean dim: {:d}".format(out_m.shape[0]))
                print("Std dim: {:d}".format(out_s.shape[0]))
                print("Output dimension incompatible")
                sys.exit(1)
        else:
            in_m = torch.zeros([in_dim])
            in_s = torch.ones([in_dim])
            out_m = torch.zeros([out_dim])
            out_s = torch.ones([out_dim])
            
        return in_m, in_s, out_m, out_s
        
    def normalize_input(self, x):
        """ normalizing the input data
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        """
        return y * self.output_std + self.output_mean

    def forward(self, input_feat, wav):
        """loss = forward(self, input_feat, wav)

        input
        -----
          input_feat: tensor, input features (batchsize, length1, input_dim)
          wav: tensor, target waveform (batchsize, length2, 1)
               it should be raw waveform, flot valued, between (-1, 1)
               the code will do mu-law conversion
        output
        ------
          loss: tensor / scalar
        
        Note: returned loss can be directly used as the loss value
        no need to write Loss()
        """
        # normalize conditiona feature
        #input_feat = self.normalize_input(input_feat)
        # compute 
        z_bags, neg_logp, logp_z, log_detjac = self.m_waveglow(wav, input_feat)
        return [[-logp_z, -log_detjac], [True, True]]

    def inference(self, input_feat):
        """wav = inference(mels)

        input
        -----
          input_feat: tensor, input features (batchsize, length1, input_dim)

        output
        ------
          wav: tensor, target waveform (batchsize, length2, 1)

        Note: length2 will be = length1 * self.up_sample
        """ 
        #normalize input
        #input_feat = self.normalize_input(input_feat)
        
        length = input_feat.shape[1] * self.up_sample
        noise = self.m_waveglow.get_z_noises(length, noise_std=0.6, 
                                             batchsize=input_feat.shape[0])
        output = self.m_waveglow.reverse(noise, input_feat)

        # use a randomized 
        if self.bg_noise is None:
            self.bg_noise = self.m_waveglow.reverse(noise, input_feat * 0)

        # do post filtering
        output = nii_nn_frontend.spectral_substraction(output,self.bg_noise,1.0)
        return output


# Loss is returned by model.forward(), no need to specify 
# just a place holder so that the output of model.forward() can be 
# sent to the optimizer
class Loss():
    def __init__(self, args):
        return

    def compute(self, output, target):
        return output

    
if __name__ == "__main__":
    print("Definition of model")

    
