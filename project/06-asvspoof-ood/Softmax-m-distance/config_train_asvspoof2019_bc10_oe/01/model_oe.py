#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import core_modules.p2sgrad as nii_p2sgrad
import sandbox.eval_asvspoof as nii_asvspoof 
import importlib
import core_scripts.math_tools.stats as nii_stats_tk

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##############
## FOR MODEL
##############

class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        ##### required part, no need to change #####

        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(
            in_dim,out_dim, args, prj_conf, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)        
        #####
        
        ####
        # on input waveform and output target
        ####
        # Load protocol and prepare the target data for network training

        # Backend-model
        # The model should be define
        # model.v_emd_dim: dimension of the embedding
        # model.get_embedding(x): compute the embedding given the input x
        model = importlib.import_module("model")
        self.m_model = model.Model(in_dim, out_dim, args, prj_conf, mean_std)
        self.m_loss_in = model.Loss(args)
        # load pre-trained softmax model to initialize
        self.g_pretrained_model_path = 'trained_network.pt'
        self.g_pretrained_model_prefix = 'm_model'
        
        # training criterion for OE
        # here we define a few OE loss
        if type(self.m_loss_in.m_loss) == nii_p2sgrad.P2SGradLoss:
            # p2sgrad
            self.m_loss_oe = self.__loss_oe_p2s

        elif type(self.m_loss_in.m_loss) == torch_nn.BCELoss:
            # BCE for sigmoid
            self.m_loss_oe = self.__loss_oe_sig
        else:
            print("Loss of outlier exposure is not implemented")
            sys.exit(1)

        
        # batch size of each data sub set
        self.m_orig_batchsize = args.batch_size

        # done
        return
    
    def __loss_oe_p2s(self, input_score, target):
        # (cost\theta - 0.5)^2
        return ((input_score - 0.5)**2).mean()
    
    def __loss_oe_sig(self, input_score, target):
        # -0.5(logp + log(1-p))
        return -0.5 * ( torch.log(input_score * (1 - input_score))).mean()


    def prepare_mean_std(self, in_dim, out_dim, args, prj_conf, 
                         data_mean_std=None):
        """ prepare mean and std for data processing
        This is required for the Pytorch project, but not relevant to this code
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
        This is required for the Pytorch project, but not relevant to this code
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but not relevant to this code
        """
        return y * self.output_std + self.output_mean


    def forward(self, input, fileinfo):
        """
        """
        batchsize = input.shape[0]
        # data in D_in
        input_in = input[0 : self.m_orig_batchsize]
        fileinfo_in = fileinfo[0 : self.m_orig_batchsize]

        # data in D_oe
        input_oe = input[self.m_orig_batchsize : ]
        fileinfo_oe = fileinfo[self.m_orig_batchsize : ]

        # all the classication model takes [scores, target_vec, -] as output
        out_in = self.m_model.forward(input_in, fileinfo_in)
        # just send the output of model.forward to the loss function
        loss_in = self.m_loss_in.compute(out_in, None)
        
        # OE loss depends on the model
        out_oe = self.m_model.forward(input_oe, fileinfo_oe)
        # out_oe[0] is the score, output_oe[1] is the target
        loss_oe = self.m_loss_oe(out_oe[0], out_oe[1])
        
        # return the loss
        return loss_in + loss_oe

    def inference(self, input, fileinfo):
        """
        """
        return self.m_model.forward(input, fileinfo)


class Loss():
    """ Wrapper to define loss function 
    Here we do noting since loss has been computed by forward
    """
    def __init__(self, args):
        return


    def compute(self, outputs, target):
        return outputs

    
if __name__ == "__main__":
    print("Definition of model")

    
