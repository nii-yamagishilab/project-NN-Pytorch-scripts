#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torchaudio
import torch.nn.functional as torch_nn_func

import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import sandbox.eval_asvspoof as nii_asvspoof 

import sandbox.block_rawnet as nii_rawnet

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
            in_dim,out_dim,args, prj_conf, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        
        # a flag for debugging (by default False)
        #self.model_debug = False
        #self.validation = False
        #####
        
        ####
        # on input waveform and output target
        ####
        # Load protocol and prepare the target data for network training
        protocol_pth = prj_conf.optional_argument[0]
        self.protocol_parser = nii_asvspoof.protocol_parse_general(protocol_pth)
        
        # Model
        self.m_filter_num = 20
        self.m_filter_len = 1025
        self.m_res_ch_1 = 128
        self.m_res_ch_2 = 512
        self.m_gru_node = 256
        self.m_gru_layer = 3
        self.m_emb_dim = 64
        self.m_num_class = 2
        self.m_rawnet = nii_rawnet.RawNet(
            self.m_filter_num, self.m_filter_len,
            in_dim, prj_conf.wav_samp_rate,
            self.m_res_ch_1, self.m_res_ch_2,
            self.m_gru_node, self.m_gru_layer,
            self.m_emb_dim,  self.m_num_class
        )
        
        # segment length = 
        self.m_seg_len = prj_conf.truncate_seq
        if self.m_seg_len is None:
            # use default segment length
            self.m_seg_len = 64600
            print("RawNet uses a default segment length {:d}".format(
                self.m_seg_len))
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, 
                         prj_conf, data_mean_std=None):
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


    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def forward(self, x, fileinfo):
        
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        
        if self.training:    
            # log prob
            logprob = self.m_rawnet.forward(x)
            
            # target
            target = self._get_target(filenames)
            target_vec = torch.tensor(
                target, device=x.device, dtype=torch.long)
            
            return [logprob, target_vec, True]

        else:
            
            if x.shape[1] < self.m_seg_len:
                # too short, no need to split
                scores = self.m_rawnet.inference(x)
            else:
                # split input into segments
                num_seq = x.shape[1] // self.m_seg_len
            
                scores = []
                for idx in range(num_seq):
                    stime = idx * self.m_seg_len
                    etime = idx * self.m_seg_len + self.m_seg_len
                    scores.append(self.m_rawnet.inference(x[:, stime:etime]))

                # average scores
                # (batch, num_classes, seg_num) -> (batch, num_classes)
                scores = torch.stack(scores, dim=2).mean(dim=-1)
            
            targets = self._get_target(filenames)
            for filename, target, score in zip(filenames, targets, scores):
                print("Output, %s, %d, %f" % (filename, target, score[-1]))
            return None


class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        self.m_loss = torch_nn.CrossEntropyLoss()


    def compute(self, outputs, target):
        """ 
        """
        loss = self.m_loss(outputs[0], outputs[1])
        return loss

    
if __name__ == "__main__":
    print("Definition of model")

    
