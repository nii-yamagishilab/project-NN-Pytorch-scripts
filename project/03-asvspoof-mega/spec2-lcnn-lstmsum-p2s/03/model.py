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

import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import core_modules.p2sgrad as nii_p2sgrad
import config as prj_conf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##############
## util
##############

def protocol_parse(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    
    input:
    -----
      protocol_filepath: string, path to the protocol file
        for convenience, I put train/dev/eval trials into a single protocol file
    
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """ 
    data_buffer = {}
    temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    for row in temp_buffer:
        if row[-1] == 'bonafide':
            data_buffer[row[1]] = 1
        else:
            data_buffer[row[1]] = 0
    return data_buffer

##############
## FOR MODEL
##############


class TrainableLinearFb(nii_nn.LinearInitialized):
    """Linear layer initialized with linear filter bank
    """
    def __init__(self, fn, sr, filter_num):
        super(TrainableLinearFb, self).__init__(
            nii_front_end.linear_fb(fn, sr, filter_num))
        return
    def forward(self, x):
        return torch.log10(
            torch.pow(super(TrainableLinearFb, self).forward(x), 2) + 
            torch.finfo(torch.float32).eps)


class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, mean_std=None):
        super(Model, self).__init__()

        ##### required part, no need to change #####

        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
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
        protocol_file = prj_conf.optional_argument[0]
        self.protocol_parser = protocol_parse(protocol_file)
        
        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        self.m_target_sr = 16000

        ####
        # optional configs (not used)
        ####                
        # re-sampling (optional)
        #self.m_resampler = torchaudio.transforms.Resample(
        #    prj_conf.wav_samp_rate, self.m_target_sr)

        # vad (optional)
        #self.m_vad = torchaudio.transforms.Vad(sample_rate = self.m_target_sr)
        
        # flag for balanced class (temporary use)
        #self.v_flag = 1

        ####
        # front-end configuration
        #  multiple front-end configurations may be used
        #  by default, use a single front-end
        ####    
        # frame shift (number of waveform points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [512]

        self.spec_with_delta = False
        self.spec_fb_dim = 60

        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating (not used)
        self.amp_floor = 0.00001
        
        # number of frames to be kept for each trial
        # no truncation
        self.v_truncate_lens = [None for x in self.frame_hops]


        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.frame_lens)        

        # dimension of embedding vectors
        self.v_emd_dim = 64

        # output classes
        self.v_out_class = 2

        ####
        # create network
        ####
        # 1st part of the classifier
        self.m_transform = []
        # 
        self.m_before_pooling = []
        # 2nd part of the classifier
        self.m_output_act = []
        # front-end
        self.m_frontend = []
        # final part on training 
        self.m_angle = []
        

        # it can handle models with multiple front-end configuration
        # by default, only a single front-end
        for idx, (trunc_len, fft_n) in enumerate(zip(
                self.v_truncate_lens, self.fft_n)):
            
            fft_n_bins = fft_n // 2 + 1
            
            self.m_transform.append(
                torch_nn.Sequential(
                    TrainableLinearFb(fft_n,self.m_target_sr,self.spec_fb_dim),

                    torch_nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
                    nii_nn.MaxFeatureMap2D(),
                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),
                    torch_nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),
                    torch_nn.BatchNorm2d(48, affine=False),

                    torch_nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(48, affine=False),
                    torch_nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    torch_nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(64, affine=False),
                    torch_nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),

                    torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),
                    torch_nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.MaxPool2d([2, 2], [2, 2]),
                    
                    torch_nn.Dropout(0.7)
                )
            )

            self.m_before_pooling.append(
                torch_nn.Sequential(
                    nii_nn.BLSTMLayer((self.spec_fb_dim//16) * 32, 
                                      (self.spec_fb_dim//16) * 32),
                    nii_nn.BLSTMLayer((self.spec_fb_dim//16) * 32, 
                                      (self.spec_fb_dim//16) * 32)
                )
            )

            self.m_output_act.append(
                torch_nn.Linear((self.spec_fb_dim // 16) * 32, self.v_emd_dim)
            )

            self.m_angle.append(
                nii_p2sgrad.P2SActivationLayer(self.v_emd_dim, self.v_out_class)
            )

            self.m_frontend.append(
                nii_front_end.Spectrogram(self.frame_lens[idx],
                                          self.frame_hops[idx],
                                          self.fft_n[idx],
                                          self.m_target_sr)
            )

        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)
        self.m_angle = torch_nn.ModuleList(self.m_angle)
        self.m_before_pooling = torch_nn.ModuleList(self.m_before_pooling)
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, data_mean_std=None):
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


    def _front_end(self, wav, idx, trunc_len, datalength):
        """ simple fixed front-end to extract features
        
        input:
        ------
          wav: waveform
          idx: idx of the trial in mini-batch
          trunc_len: number of frames to be kept after truncation
          datalength: list of data length in mini-batch

        output:
        -------
          x_sp_amp: front-end featues, (batch, frame_num, frame_feat_dim)
        """
        
        with torch.no_grad():
            x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))

        # return
        return x_sp_amp

    def _compute_embedding(self, x, datalength):
        """ definition of forward method 
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        #x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)
        
        # number of sub models
        batch_size = x.shape[0]

        # buffer to store output scores from sub-models
        output_emb = torch.zeros([batch_size * self.v_submodels, 
                                  self.v_emd_dim], 
                                  device=x.device, dtype=x.dtype)
        
        # compute scores for each sub-models
        for idx, (fs, fl, fn, trunc_len, m_trans, m_be_pool, m_output) in \
            enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, 
                    self.v_truncate_lens, self.m_transform, 
                    self.m_before_pooling, self.m_output_act)):
            
            # extract front-end feature
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)

            # compute scores
            #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
            #  2. compute hidden features
            hidden_features = m_trans(x_sp_amp.unsqueeze(1))

            #  3. (batch, channel, frame//N, feat_dim//N) ->
            #     (batch, frame//N, channel * feat_dim//N)
            #     where N is caused by conv with stride
            hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
            frame_num = hidden_features.shape[1]
            hidden_features = hidden_features.view(batch_size, frame_num, -1)
            
            #  4. pass through LSTM then summing
            hidden_features_lstm = m_be_pool(hidden_features)

            #  5. pass through the output layer
            tmp_emb = m_output((hidden_features_lstm + hidden_features).sum(1))
            
            output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_emb

        return output_emb

    def _compute_score(self, x, inference=False):
        """
        """
        # number of sub models
        batch_size = x.shape[0]

        # compute score through p2sgrad layer
        out_score = torch.zeros(
            [batch_size * self.v_submodels, self.v_out_class], 
            device=x.device, dtype=x.dtype)

        # compute scores for each sub-models
        for idx, m_score in enumerate(self.m_angle):
            tmp_score = m_score(x[idx * batch_size : (idx+1) * batch_size])
            out_score[idx * batch_size : (idx+1) * batch_size] = tmp_score

        if inference:
            # output_score [:, 1] corresponds to the positive class
            return out_score[:, 1]
        else:
            return out_score


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
            
            feature_vec = self._compute_embedding(x, datalength)
            scores = self._compute_score(feature_vec)
            
            # target
            target = self._get_target(filenames)
            target_vec = torch.tensor(target, 
                                      device=x.device, dtype=scores.dtype)
            target_vec = target_vec.repeat(self.v_submodels)
            
            return [scores, target_vec, True]

        else:
            feature_vec = self._compute_embedding(x, datalength)
            scores = self._compute_score(feature_vec, True)
            
            target = self._get_target(filenames)
            print("Output, %s, %d, %f" % (filenames[0], 
                                          target[0], scores.mean()))
            # don't write output score as a single file
            return None


class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        self.m_loss = nii_p2sgrad.P2SGradLoss()


    def compute(self, outputs, target):
        """ 
        """
        loss = self.m_loss(outputs[0], outputs[1])
        return loss

    
if __name__ == "__main__":
    print("Definition of model")

    
