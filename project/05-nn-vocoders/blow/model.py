#!/usr/bin/env python
"""
model.py for Blow
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
import sandbox.block_blow as nii_blow


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


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

        # load speaker map
        self.speaker_map = prj_conf.options['speaker_map']
        self.speaker_num = self.speaker_map.num()
        if 'conversion_map' in prj_conf.options:
            self.conversion_map = prj_conf.options['conversion_map']
        else:
            self.conversion_map = None

        self.cond_dim = 128
        self.num_block = 8
        self.num_flow_steps_perblock = 12
        self.num_conv_channel_size = 512
        self.num_conv_conv_kernel = 3


        self.m_spk_emd = torch.nn.Embedding(self.speaker_num, self.cond_dim)

        self.m_blow = nii_blow.Blow(
            self.cond_dim, self.num_block, 
            self.num_flow_steps_perblock,
            self.num_conv_channel_size,
            self.num_conv_conv_kernel)

        # only used for synthesis
        self.m_overlap = nii_blow.OverlapAdder(4096, 4096//4, False)
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

    def forward(self, wav, fileinfo):
        """loss = forward(self, input_feat, wav)

        input
        -----
          wav: tensor, target waveform (batchsize, length2, 1)
               it should be raw waveform, flot valued, between (-1, 1)
               the code will do mu-law conversion

          fileinfo: list, file information for each data in the batch

        output
        ------
          loss: tensor / scalar,
        
        Note: returned loss can be directly used as the loss value
        no need to write Loss()
        """
        # prepare speaker IDs
        # (batch, )
        speaker_ids = torch.tensor(
            [self.speaker_map.parse(x) for x in fileinfo],
            dtype=torch.long, device=wav.device)
        # convert to embeddings
        # (batch, 1, cond_dim)
        speaker_emd = self.m_spk_emd(speaker_ids).unsqueeze(1)

        # normalize conditiona feature
        #input_feat = self.normalize_input(input_feat)
        # compute 
        z, neg_logp, logp_z, log_detjac = self.m_blow(wav, speaker_emd)
        return [[-logp_z, -log_detjac], [True, True]]

    def inference(self, wav, fileinfo):
        """wav = inference(mels)

        input
        -----
          wav: tensor, target waveform (batchsize, length2, 1)

        output
        ------
          wav_new: tensor, same shape
        """ 
        # framing the input waveform into frames
        # 
        # framed_wav (batchsize, frame_num, frame_length)
        framed_wav = self.m_overlap(wav)
        batch, frame_num, frame_len = framed_wav.shape
        # framed_Wav (batchsize * frame_num, frame_length, 1)
        framed_wav = framed_wav.view(-1, frame_len).unsqueeze(-1)


        # get the speaker IDs
        # (batch, )
        speaker_ids = torch.tensor(
            [self.speaker_map.parse(x) for x in fileinfo],
            dtype=torch.long, device=wav.device)
        # (batch * frame_num)
        speaker_ids = speaker_ids.repeat_interleave(frame_num)
        
        # if conversion map is defined, change the speaker identity following
        #  the conversion map. Otherwise, specify target speaker ID
        #  through environment variable TEMP_TARGET_SPEAKER
        if self.conversion_map:
            target_speaker = torch.tensor(
                [self.speaker_map.get_idx(
                    self.conversion_map[self.speaker_map.parse(x, False)]) \
                    for x in fileinfo],
                device=wav.device, dtype=torch.long)
            target_speaker = target_speaker.repeat_interleave(frame_num)
        else:
            # target speaker ID
            target_speaker = torch.tensor(
                [int(os.getenv('TEMP_TARGET_SPEAKER'))] * speaker_ids.shape[0],
                device=wav.device, dtype=torch.long)
                    
        # if it is intended to swap the source / target speaker ID
        #  this can be used to recover the original waveform given converted
        #  speech
        flag_reverse = os.getenv('TEMP_FLAG_REVERSE')
        if flag_reverse and int(flag_reverse):
            # if this is for reverse conversion
            # swap the IDs
            print("revert IDs")
            tmp = speaker_ids
            speaker_ids = target_speaker
            target_speaker = tmp
            
        # print some information
        for idx in range(len(speaker_ids)):
            if idx % frame_num == 0:
                print("From ID {:3d} to {:3d}, ".format(
                    speaker_ids[idx], target_speaker[idx]), end=' ')
        
        # get embeddings (batch * frame_num, 1, cond_dim)
        speaker_emd = self.m_spk_emd(speaker_ids).unsqueeze(1)
        target_speaker_emb = self.m_spk_emd(target_speaker).unsqueeze(1)
        
        # compute 
        z, neg_logp, logp_z, log_detjac = self.m_blow(framed_wav, speaker_emd)
        
        # output_framed (batch * frame, frame_length, 1)
        output_framed = self.m_blow.reverse(z, target_speaker_emb)

        # overlap and add
        # view -> (batch, frame_num, frame_length)
        return self.m_overlap.reverse(output_framed.view(batch, -1, frame_len), 
                                      True)


    def convert(self, wav, src_id, tar_id):
        """wav = inference(mels)

        input
        -----
          wav: tensor, target waveform (batchsize, length2, 1)
          src_id: int, ID of source speaker
          tar_id: int, ID of target speaker
          
        output
        ------
          wav_new: tensor, same shape
        """ 
        # framing the input waveform into frames
        #   m_overlap.forward does framing
        # framed_wav (batchsize, frame_num, frame_length)
        framed_wav = self.m_overlap(wav)
        batch, frame_num, frame_len = framed_wav.shape
        
        # change frames into batch
        # framed_Wav (batchsize * frame_num, frame_length, 1)
        framed_wav = framed_wav.view(-1, frame_len).unsqueeze(-1)

        
        # source speaker IDs
        # (batch, )
        speaker_ids = torch.tensor([src_id for x in wav],
                                   dtype=torch.long, device=wav.device)
        # (batch * frame_num)
        speaker_ids = speaker_ids.repeat_interleave(frame_num)
        # get embeddings (batch * frame_num, 1, cond_dim)
        speaker_emd = self.m_spk_emd(speaker_ids).unsqueeze(1)
        
        
        # target speaker IDs
        tar_speaker_ids = torch.tensor([tar_id for x in wav],
                                   dtype=torch.long, device=wav.device)
        # (batch * frame_num)
        tar_speaker_ids = tar_speaker_ids.repeat_interleave(frame_num)
        target_speaker_emb = self.m_spk_emd(tar_speaker_ids).unsqueeze(1)
        
        # analysis 
        z, _, _, _ = self.m_blow(framed_wav, speaker_emd)
        
        # synthesis
        # output_framed (batch * frame, frame_length, 1)
        output_framed = self.m_blow.reverse(z, target_speaker_emb)

        # overlap and add
        # view -> (batch, frame_num, frame_length)
        return self.m_overlap.reverse(
            output_framed.view(batch, -1, frame_len), True)

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

    
