#!/usr/bin/env python
"""
model.py for iLPCNet
version: 1
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

import core_scripts.other_tools.debug as nii_debug
import core_scripts.other_tools.display as nii_warn
import sandbox.block_nn as nii_nn

import block_lpcnet as nii_lpcnet

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


########
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
        
        self.fl = prj_conf.options['framelength']
        self.fs = prj_conf.options['frameshift']
        self.lpc_order = prj_conf.options['lpc_order']


        self.feat_reso = prj_conf.input_reso[0]
        if self.fs % self.feat_reso == 0:
            if self.fs >= self.feat_reso:
                self.feat_upsamp = self.fs // self.feat_reso
            else:
                print("Adjust condition feature resolution not supported ")
                sys.exit(1)

        self.flag_fix_cond = args.temp_flag == 'stage2'

        self.m_lpcnet = nii_lpcnet.LPCNetV1(
            self.fl, self.fs, self.lpc_order, in_dim, self.flag_fix_cond)

        
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
        #return (y - self.output_mean) / self.output_std
        # now the target features will be a list of features
        return y

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        """
        return y * self.output_std + self.output_mean

    def forward(self, cond_feat, target):
        """

        input
        -----

        output
        ------
          loss: tensor / scalar
        
        Note: returned loss can be directly used as the loss value
        no need to write Loss()
        """
        # 
        wav = target[0]
        # (batch, frame_num, lpc_order)
        lpc_coef = target[1]
        # (batch, framge_num, lpc_order - 1)
        rc = target[2]
        # (batch, framge_num, 1)
        gain = target[3]
        
        # condition feature adjust
        cond_feat_tmp = cond_feat[:, ::self.feat_upsamp]

        loss_cond, loss_lpcnet = self.m_lpcnet(
            cond_feat_tmp, self.normalize_input(cond_feat_tmp), 
            lpc_coef, rc, gain, wav)
        return [[loss_cond, loss_lpcnet], [True, True]]

    def inference(self, cond_feat):
        """wav = inference(mels)

        input
        -----

        output
        ------
          wav_new: tensor, same shape
        """ 

        # condition feature adjust
        cond_feat_tmp = cond_feat[:, ::self.feat_upsamp]

        return self.m_lpcnet.inference(
            cond_feat_tmp, self.normalize_input(cond_feat_tmp))


# Loss is returned by Model.forward(), no need to specify 
# just a place holder
class Loss():
    def __init__(self, args):
        return

    def compute(self, output, target):
        return output

    
if __name__ == "__main__":
    print("Definition of model")

    
