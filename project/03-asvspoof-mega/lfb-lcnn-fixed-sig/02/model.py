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
import core_modules.a_softmax as nii_a_softmax
import core_scripts.data_io.seq_info as nii_seq_tk
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
        
        ## a flag for debugging (by default False)
        #self.model_debug = False
        #self.validation = False
        #####

        ####
        # on input waveform and output target
        ####
        # Load protocol and prepare the target data for network training
        # target data
        protocol_file = prj_conf.optional_argument[0]
        self.protocol_parser = protocol_parse(protocol_file)
 
        # working sampling rate, torchaudio is used to change sampling rate
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
        # frame shift (number of points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [512]

        # LFB dim (base component)
        self.lfb_dim = [60]
        self.lfb_with_delta = False

        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating
        self.amp_floor = 0.00001
        
        # manual choose the first 600 frames in the data
        self.v_truncate_lens = [10 * 16 * 750 // x for x in self.frame_hops]

        # number of sub-models
        self.v_submodels = len(self.frame_lens)        

        # dimension of embedding vectors
        self.v_emd_dim = 1

        # output class
        self.v_out_class = 1

        self.m_transform = []
        self.m_output_act = []
        self.m_frontend = []
        #self.m_a_softmax = []

        for idx, (trunc_len, fft_n, lfb_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfb_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfb_with_delta:
                lfb_dim = lfb_dim * 3
            
            self.m_transform.append(
                torch_nn.Sequential(
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
                    torch_nn.MaxPool2d([2, 2], [2, 2])
                )
            )
            self.m_output_act.append(
                torch_nn.Sequential(
                    torch_nn.Dropout(0.7),
                    torch_nn.Linear((trunc_len // 16) * 
                                    (lfb_dim // 16) * 32, 160),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.Linear(80, self.v_emd_dim)
                )
            )
            
            self.m_frontend.append(
                nii_front_end.LFB(self.frame_lens[idx],
                                  self.frame_hops[idx],
                                  self.fft_n[idx],
                                  self.m_target_sr,
                                  self.lfb_dim[idx],
                                  with_energy=False,
                                  with_emphasis=True,
                                  with_delta=self.lfb_with_delta)
            )

            #self.m_a_softmax.append(
            #    nii_a_softmax.AngleLayer(self.v_emd_dim, self.v_out_class)
            #)

        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)
        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        #self.m_a_softmax = torch_nn.ModuleList(self.m_a_softmax)
        
        # output 

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
            in_s = torch.zeros([in_dim])
            out_m = torch.ones([out_dim])
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


    def _front_end(self, wav, idx, trunc_len, datalength):
        """ simple fixed front-end to extract features
        fs: frame shift
        fl: frame length
        fn: fft points
        trunc_len: number of frames per file (by truncating)
        datalength: original length of data
        """
        
        with torch.no_grad():
            x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))
            
            #  permute to (batch, fft_bin, frame_length)
            x_sp_amp = x_sp_amp.permute(0, 2, 1)
            
            # make sure the buffer is long enough
            x_sp_amp_buff = torch.zeros(
                [x_sp_amp.shape[0], x_sp_amp.shape[1], trunc_len], 
                dtype=x_sp_amp.dtype, device=x_sp_amp.device)
            
            # for batch of data, handle the padding and trim independently
            fs = self.frame_hops[idx]
            for fileidx in range(x_sp_amp.shape[0]):
                # roughtly this is the number of frames
                true_frame_num = datalength[fileidx] // fs
                
                if true_frame_num > trunc_len:
                    # trim randomly
                    pos = torch.rand([1]) * (true_frame_num-trunc_len)
                    pos = torch.floor(pos[0]).long()
                    tmp = x_sp_amp[fileidx, :, pos:trunc_len+pos]
                    x_sp_amp_buff[fileidx] = tmp
                else:
                    rep = int(np.ceil(trunc_len / true_frame_num)) 
                    tmp = x_sp_amp[fileidx, :, 0:true_frame_num].repeat(1, rep)
                    x_sp_amp_buff[fileidx] = tmp[:, 0:trunc_len]

            #  permute to (batch, frame_length, fft_bin)
            x_sp_amp = x_sp_amp_buff.permute(0, 2, 1)
            
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
        output_score = torch.zeros([batch_size * self.v_submodels, 
                                   self.v_emd_dim], 
                                  device=x.device, dtype=x.dtype)
        
        # compute scores for each sub-models
        for idx, (fs, fl, fn, trunc_len, m_trans, m_output) in enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, 
                    self.v_truncate_lens,
                    self.m_transform, self.m_output_act)):
            
            # extract feature (stft spectrogram)
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)

            # compute scores
            #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
            #  2. compute hidden features
            hidden_features = m_trans(x_sp_amp.unsqueeze(1))
            #  3. flatten and transform through output function
            tmp_score = m_output(torch.flatten(hidden_features, 1))
            
            output_score[idx * batch_size : (idx+1) * batch_size] = tmp_score

        return output_score

    def _compute_score(self, feature_vec, inference=False):
        """
        """
        # feature_vec is [batch * submodel, 1]
        if inference:
            return feature_vec.squeeze(1)
        else:
            return torch.sigmoid(feature_vec).squeeze(1)


    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def forward(self, x, fileinfo):
        
        #with torch.no_grad():
        #    vad_waveform = self.m_vad(x.squeeze(-1))
        #    vad_waveform = self.m_vad(torch.flip(vad_waveform, dims=[1]))
        #    if vad_waveform.shape[-1] > 0:
        #        x = torch.flip(vad_waveform, dims=[1]).unsqueeze(-1)
        #    else:
        #        pass
        
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
        self.m_loss = torch_nn.BCELoss()


    def compute(self, outputs, target):
        """ 
        """
        loss = self.m_loss(outputs[0], outputs[1])
        return loss

    
if __name__ == "__main__":
    print("Definition of model")

    
