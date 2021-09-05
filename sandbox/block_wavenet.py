#!/usr/bin/env python
"""
model.py for WaveNet
version: 1
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
import sandbox.block_dist as nii_dist
import sandbox.util_dsp as nii_dsp

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

##############
# 

class CondModule(torch_nn.Module):
    """ Conditiona module: upsample and transform input features
    """
    def __init__(self, input_dim, output_dim, up_sample, \
                 blstm_s = 64, cnn_kernel_s = 3):
        """ CondModule(input_dim, output_dim, up_sample, 
        blstm_s=64, cnn_kernel_s=3)

        Args
        ----
          input_dim: int, input tensor should be (batchsize, len1, input_dim)
          output_dim: int, output tensor will be (batchsize, len2, output_dim)
          up_sample: int, up-sampling rate, len2 = len1 * up_sample
          
          blstm_s: int, layer size of the Bi-LSTM layer
          cnn_kernel_s: int, kernel size of the conv1d
        """
        super(CondModule, self).__init__()

        # configurations
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.blstm_s = blstm_s
        self.cnn_kernel_s = cnn_kernel_s

        # layers
        self.l_blstm = nii_nn.BLSTMLayer(input_dim, self.blstm_s)
        self.l_conv1d = nii_nn.Conv1dKeepLength(
            self.blstm_s, output_dim, 1, self.cnn_kernel_s)
        self.l_upsamp = nii_nn.UpSampleLayer(
            self.output_dim, self.up_sample, True)

    def forward(self, feature):
        """ transformed_feat = forward(input_feature)

        input
        -----
          feature: (batchsize, length, input_dim)
        
        output
        ------
          transformed_feat: tensor (batchsize, length*up_sample, out_dim)
        """ 
        return self.l_upsamp(self.l_conv1d(self.l_blstm(feature)))


class WaveNetBlock(torch_nn.Module):
    """ WaveNet block based on dilated-1D, gated-activation, and skip-connect.
    Based on http://tonywangx.github.io/slide.html#misc CURRENNT WaveNet, 
    page 19-31.
    """
    def __init__(self, input_dim, skip_ch_dim, gated_act_dim, cond_dim,
                 dilation_size, cnn_kernel_size=2, causal=True):
        """ WaveNetBlock(input_dim, skip_ch_dim, gated_act_dim, cond_dim,
        dilation_size, cnn_kernel_size = 2)

        Args
        ----
          input_dim: int, input tensor should be (batch-size, length, input_dim)
                     this is the dimension of residual channel
          skip_ch_dim: int, tensors to be send to output blocks is in shape
                            (batch-size, length, skip_ch_dim)
          gated_act_dim: int, tensors given by tanh(.) * sig(.) is in shape
                            (batch-size, length, gated_act_dim)
          cond_dim: int, conditional feature (batchsize, length, cond_dim)

          dilation_size: int, dilation size of the conv
          cnn_kernel_size: int, kernel size of dilated conv1d (default, 2)
          causal: bool, whether this block is used in AR model (default, True)
        
        Note that causal==False will raise error if step-by-step generation
        is conducted by inference(input_feat, cond_feat, step_idx) with 
        step_idx != None. 
        For causal==False, always use inference(input_feat, cond_feat, None)
        """
        super(WaveNetBlock, self).__init__()

        #####
        # configurations
        #####
        # input tensor: (batchsize, length, self.input_dim)
        self.input_dim = input_dim
        # tensor sent to next WaveNetBlock, same shape as input
        self.res_ch_dim = input_dim
        # 
        self.skip_ch_dim = skip_ch_dim
        self.gated_act_dim = gated_act_dim
        self.cond_dim = cond_dim
        self.dilation_size = dilation_size
        self.conv_kernel_s = cnn_kernel_size

        ######
        # layers
        ######
        # dilated convolution
        self.l_conv1d = nii_nn.Conv1dForARModel(
            self.input_dim, self.gated_act_dim * 2, self.dilation_size,
            self.conv_kernel_s, tanh=False)
        
        # condition feature transform
        self.l_cond_trans = torch_nn.Sequential(
            torch_nn.Linear(self.cond_dim, self.gated_act_dim*2),
            torch_nn.LeakyReLU())
        
        # transformation after gated act
        self.l_res_trans = torch_nn.Linear(self.gated_act_dim, self.res_ch_dim)
        
        # transformation for skip channels
        self.l_skip_trans = torch_nn.Linear(self.res_ch_dim, self.skip_ch_dim)
        
        return

    def _forward(self, input_feat, cond_feat, step_idx=None):
        """ res_feat, skip_feat = forward(input_feat, cond_feat)
        
        input
        -----
          input_feat: input feature tensor, (batchsize, length, input_dim)
          cond_feat: condition feature tensor, (batchsize, length, cond_dim)
          step_idx: None: tranining phase
                    int: idx of the time step during step-by-step generation

        output
        ------
          res_feat: residual channel feat tensor, (batchsize, length, input_dim)
          skip_feat: skip channel feat tensor, , (batchsize, length, skip_dim)
        """
        # dilated 1d convolution
        hid = self.l_conv1d(input_feat, step_idx)

        # transform and add condition feature
        hid = hid + self.l_cond_trans(cond_feat)

        # gated activation
        hid = torch.tanh(hid[:, :, 0:self.gated_act_dim]) \
              * torch.sigmoid(hid[:, :, self.gated_act_dim:])

        # res-channel transform
        res_feat = self.l_res_trans(hid) + input_feat
        # skip-channel transform
        skip_feat = self.l_skip_trans(res_feat)

        # done
        return res_feat, skip_feat


    def forward(self, input_feat, cond_feat):
        """ res_feat, skip_feat = forward(input_feat, cond_feat)
        
        input
        -----
          input_feat: input feature tensor, (batchsize, length, input_dim)
          cond_feat: condition feature tensor, (batchsize, length, cond_dim)

        output
        ------
          res_feat: residual channel feat tensor, (batchsize, length, input_dim)
          skip_feat: skip channel feat tensor, , (batchsize, length, skip_dim)
        
        Note that input_dim refers to the residual channel dimension.
        Thus, input_feat should be embedding(audio), not audio.
        """
        return self._forward(input_feat, cond_feat)


    def inference(self, input_feat, cond_feat, step_idx):
        """ res_feat, skip_feat = inference(input_feat, cond_feat, step_idx)
        
        input
        -----
          input_feat: input feature tensor, (batchsize, length, input_dim)
          cond_feat: condition feature tensor, (batchsize, length, cond_dim)
          step_idx: int, idx of the time step during step-by-step generation

        output
        ------
          res_feat: residual channel feat tensor, (batchsize, length, input_dim)
          skip_feat: skip channel feat tensor, , (batchsize, length, skip_dim)
        """
        return self._forward(input_feat, cond_feat, step_idx)



class WaveNetBlock_v2(torch_nn.Module):
    """ WaveNet block based on dilated-1D, gated-activation, and skip-connect.
    Based on http://tonywangx.github.io/slide.html#misc CURRENNT WaveNet
    (page 19-31) and WN in Pytorch WaveGlow. 
    The difference from WaveNetBlock 
    1. weight_norm
    2. skip_channel is computed from gated-activation's output, not res_channel
    """
    def __init__(self, input_dim, skip_ch_dim, gated_act_dim, cond_dim,
                 dilation_size, cnn_kernel_size=2, causal=True):
        """ WaveNetBlock(input_dim, skip_ch_dim, gated_act_dim, cond_dim,
        dilation_size, cnn_kernel_size = 2)

        Args
        ----
          input_dim: int, input tensor should be (batch-size, length, input_dim)
          skip_ch_dim: int, tensors to be send to output blocks is in shape
                            (batch-size, length, skip_ch_dim)
          gated_act_dim: int, tensors given by tanh(.) * sig(.) is in shape
                            (batch-size, length, gated_act_dim)
          cond_dim: int, conditional feature (batchsize, length, cond_dim)

          dilation_size: int, dilation size of the conv
          cnn_kernel_size: int, kernel size of dilated conv1d (default, 2)
          causal: bool, whether this block is used for AR model (default, True)
        
        Note that when causal == False, step-by-step generation using step_index
        will raise error.
        """
        super(WaveNetBlock_v2, self).__init__()

        #####
        # configurations
        #####
        # input tensor: (batchsize, length, self.input_dim)
        self.input_dim = input_dim
        # tensor sent to next WaveNetBlock, same shape as input
        self.res_ch_dim = input_dim
        # 
        self.skip_ch_dim = skip_ch_dim
        self.gated_act_dim = gated_act_dim
        self.cond_dim = cond_dim
        self.dilation_size = dilation_size
        self.conv_kernel_s = cnn_kernel_size

        ######
        # layers
        ######
        # dilated convolution
        tmp_layer = nii_nn.Conv1dForARModel(
            self.input_dim, self.gated_act_dim * 2, self.dilation_size,
            self.conv_kernel_s, tanh=False, causal = causal)
        self.l_conv1d = torch.nn.utils.weight_norm(tmp_layer, name='weight') 
        
        # condition feature transform
        tmp_layer = torch_nn.Linear(self.cond_dim, self.gated_act_dim*2)
        self.l_cond_trans = torch.nn.utils.weight_norm(tmp_layer, name='weight')
        
        # transformation after gated act
        tmp_layer = torch_nn.Linear(self.gated_act_dim, self.res_ch_dim)
        self.l_res_trans = torch.nn.utils.weight_norm(tmp_layer, name='weight') 
        
        # transformation for skip channels
        #tmp_layer = torch_nn.Linear(self.res_ch_dim, self.skip_ch_dim)
        tmp_layer = torch_nn.Linear(self.gated_act_dim, self.skip_ch_dim)
        self.l_skip_trans = torch.nn.utils.weight_norm(tmp_layer, name='weight')
        
        return

    def _forward(self, input_feat, cond_feat, step_idx=None):
        """ res_feat, skip_feat = forward(input_feat, cond_feat)
        
        input
        -----
          input_feat: input feature tensor, (batchsize, length, input_dim)
          cond_feat: condition feature tensor, (batchsize, length, cond_dim)
          step_idx: None: tranining phase
                    int: idx of the time step during step-by-step generation

        output
        ------
          res_feat: residual channel feat tensor, (batchsize, length, input_dim)
          skip_feat: skip channel feat tensor, , (batchsize, length, skip_dim)
        """
        # dilated 1d convolution, add condition feature
        hid = self.l_conv1d(input_feat, step_idx) + self.l_cond_trans(cond_feat)

        # gated activation
        hid = torch.tanh(hid[:, :, 0:self.gated_act_dim]) \
              * torch.sigmoid(hid[:, :, self.gated_act_dim:])

        # res-channel transform
        res_feat = self.l_res_trans(hid) + input_feat
        
        # skip-channel transform
        #   if we use skip_feat = self.l_skip_trans(res_feat), this cause
        #   exploding output when using skip_feat to produce scale and bias
        #   of affine transformation (e.g., in WaveGlow)
        skip_feat = self.l_skip_trans(hid)

        # done
        return res_feat, skip_feat


    def forward(self, input_feat, cond_feat):
        """ res_feat, skip_feat = forward(input_feat, cond_feat)
        
        input
        -----
          input_feat: input feature tensor, (batchsize, length, input_dim)
          cond_feat: condition feature tensor, (batchsize, length, cond_dim)

        output
        ------
          res_feat: residual channel feat tensor, (batchsize, length, input_dim)
          skip_feat: skip channel feat tensor, , (batchsize, length, skip_dim)
        """
        return self._forward(input_feat, cond_feat)


    def inference(self, input_feat, cond_feat, step_idx):
        """ res_feat, skip_feat = inference(input_feat, cond_feat, step_idx)
        
        input
        -----
          input_feat: input feature tensor, (batchsize, length, input_dim)
          cond_feat: condition feature tensor, (batchsize, length, cond_dim)
          step_idx: int, idx of the time step during step-by-step generation

        output
        ------
          res_feat: residual channel feat tensor, (batchsize, length, input_dim)
          skip_feat: skip channel feat tensor, , (batchsize, length, skip_dim)
        """
        return self._forward(input_feat, cond_feat, step_idx)



class OutputBlock(torch_nn.Module):
    """Output block to produce waveform distribution given skip-channel features
    """
    def __init__(self, input_dim, output_dim, hid_dim=512):
        """ OutputBlock(input_dim, output_dim)
        
        Args
        ----
          input_dim: int, input tensor should be (batchsize, length, input_dim)
                     it should be the sum of skip-channel features
          output_dim: int, output tensor will be (batchsize, length, output_dim)
          hid_dim: int, dimension of intermediate linear layers
        """
        super(OutputBlock, self).__init__()

        # config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim

        # transformation layers before softmax
        self.l_trans = torch_nn.Sequential(
            torch_nn.Linear(self.input_dim, self.hid_dim // 2),
            torch_nn.LeakyReLU(),
            torch_nn.Linear(self.hid_dim // 2, self.hid_dim),
            torch_nn.LeakyReLU(),
            torch_nn.Linear(self.hid_dim, self.output_dim))

        # output distribution
        self.l_dist = nii_dist.DistCategorical(self.output_dim)
        
        return
        
    def forward(self, input_feat, target):
        """loss = forward(input_feat, target) 
        This method is supposed to be used to compute the loss

        input
        -----
          input_feat: tensor in shape (batchsize, length, input_dim)
          target: waveform tensor in shape (batchsize, length, dim=1)
        
        output
        ------
          loss: tensor or scalar
        """
        # transform hidden feature vector to logit
        tmp_logit = self.l_trans(input_feat)
        # calculate the likelihood
        return self.l_dist(tmp_logit, target)

    def inference(self, input_feat):
        """output = inference(input_feat)

        input
        -----
          input_feat: tensor in shape (batchsize, length, input_dim)

        output
        ------
          target: waveform tensor in shape (batchsize, length, dim=1)
        """
        # transform hidden feature vector to logit
        tmp_logit = self.l_trans(input_feat)
        return self.l_dist.inference(tmp_logit)
        



################################
## Example of WaveNet definition
################################
class WaveNet_v1(torch_nn.Module):
    """ Model definition of WaveNet
    Example definition of WaveNet, version 1

    """
    def __init__(self, in_dim, up_sample_rate, num_bits = 10, wnblock_ver=1,
                 pre_emphasis=True):
        """ WaveNet(in_dim, up_sample_rate, num_bits=10, wnblock_ver=1,
        pre_emphasis=False)
        
        Args
        ----
          in_dim: int, dimension of condition feature (batch, length, in_dim)
          up_sample_rate, int, condition feature will be up-sampled by
                   using this rate
          num_bits: int, number of bits for mu-law companding, default 10
          wnblock_ver: int, version of the WaveNet Block, default 1
                       wnblock_ver = 1 uses WaveNetBlock
                       wnblock_ver = 2 uses WaveNetBlock_v2
          pre_emphasis: bool, whether use pre-emphasis on the target waveform

        up_sample_rate can be calculated using frame_shift of condition feature
        and waveform sampling rate. For example, 16kHz waveform, condition
        feature (e.g., Mel-spectrogram) extracted using 5ms frame shift, then
        up_sample_rate = 16000 * 0.005 = 80. In other words, every frame will
        be replicated 80 times.
        """
        super(WaveNet_v1, self).__init__()
        
        #################
        ## model config
        #################
        # number of bits for mu-law
        self.num_bits = num_bits
        self.num_classes = 2 ** self.num_bits
        
        # up-sample rate
        self.up_sample = up_sample_rate
        
        # wavenet blocks
        #  residual channel dim
        self.res_ch_dim = 64
        #  gated activate dim
        self.gate_act_dim = 64
        #  condition feature dim
        self.cond_dim = 64
        #  skip channel dim
        self.skip_ch_dim = 256
        #  dilation size
        self.dilations = [2 ** (x % 10) for x in range(30)]
        
        # input dimension of (conditional feature)
        self.input_dim = in_dim
        
        # version of wavenet block
        self.wnblock_ver = wnblock_ver

        # whether pre-emphasis
        self.pre_emphasis = pre_emphasis
        ###############
        ## network definition
        ###############
        # condition module
        self.l_cond = CondModule(self.input_dim, self.cond_dim, self.up_sample)
        
        # waveform embedding layer 
        self.l_wav_emb = torch_nn.Embedding(self.num_classes, self.res_ch_dim)

        # dilated convolution layers
        tmp_wav_blocks = []
        for dilation in self.dilations:
            if self.wnblock_ver == 2:
                tmp_wav_blocks.append(
                    WaveNetBlock_v2(
                        self.res_ch_dim, self.skip_ch_dim, self.gate_act_dim,
                        self.cond_dim, dilation))
            else:
                tmp_wav_blocks.append(
                    WaveNetBlock(
                        self.res_ch_dim, self.skip_ch_dim, self.gate_act_dim,
                        self.cond_dim, dilation))
        self.l_wavenet_blocks = torch_nn.ModuleList(tmp_wav_blocks)
        
        # output block
        self.l_output = OutputBlock(self.skip_ch_dim, self.num_classes)
        
        # done
        return
    
    def _waveform_encode_target(self, target_wav):
        return nii_dsp.mulaw_encode(target_wav, self.num_classes)

    def _waveform_decode_target(self, gen_wav):
        return nii_dsp.mulaw_decode(gen_wav, self.num_classes)

    def forward(self, input_feat, wav):
        """loss = forward(self, input_feat, wav)

        input
        -----
          input_feat: tensor, input features (batchsize, length1, input_dim)
          wav: tensor, target waveform (batchsize, length2, 1)
               it should be raw waveform, flot valued, between (-1, 1)
               it will be companded using mu-law automatically
        output
        ------
          loss: tensor / scalar
        
        Note: returned loss can be directly used as the loss value
        no need to write Loss()
        """
        
        # step1. prepare the target waveform and feedback waveform
        #  do mu-law companding
        #  shifting by 1 time step for feedback waveform
        with torch.no_grad():
            if self.pre_emphasis:
                wav[:, 1:, :] = wav[:, 1:, :] - 0.97 * wav[:, 0:-1, :]
                wav = wav.clamp(-1, 1)

            # mu-law companding (int values)
            # note that _waveform_encoder_target will produce int values
            target_wav = self._waveform_encode_target(wav)
            
            # feedback wav
            fb_wav = torch.zeros(
                target_wav.shape, device=wav.device, dtype=target_wav.dtype)
            fb_wav[:, 1:] = target_wav[:, :-1]
        
        # step2. condition feature
        hid_cond = self.l_cond(input_feat)

        # step3. feedback waveform embedding
        hid_wav_emb = self.l_wav_emb(fb_wav.squeeze(-1))

        # step4. stacks of wavenet
        #  buffer to save skip-channel features
        skip_ch_feat = torch.zeros(
            [target_wav.shape[0],target_wav.shape[1], self.skip_ch_dim],
            device=input_feat.device, dtype=input_feat.dtype)
        
        res_ch_feat = hid_wav_emb
        for l_wavblock in self.l_wavenet_blocks:
            res_ch_feat, tmp_skip_ch_feat = l_wavblock(res_ch_feat, hid_cond)
            skip_ch_feat += tmp_skip_ch_feat
        
        # step5. get output
        likelihood = self.l_output(skip_ch_feat, target_wav)

        return likelihood

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

        # prepare
        batchsize = input_feat.shape[0]
        wavlength = input_feat.shape[1] * self.up_sample
        
        time_idx_marker = wavlength // 10
        #
        # step1. condition features
        hid_cond = self.l_cond(input_feat)

        # step2. do computation step-by-step
        # initialzie the buffer
        gen_wav_buf = torch.zeros(
            [batchsize, wavlength, 1], 
            dtype=input_feat.dtype, device=input_feat.device)

        fb_wav_buf = torch.zeros(
            [batchsize, 1, 1], 
            dtype=input_feat.dtype, device=input_feat.device)

        skip_ch_feat = torch.zeros(
            [batchsize, 1, self.skip_ch_dim],
            dtype=input_feat.dtype, device=input_feat.device)

        # loop over all time steps
        print("Total time steps: {:d}. Progress: ".format(wavlength), 
              end=' ', flush=True)
        for time_idx in range(wavlength):
            # show messages
            if time_idx % 500 == 1:
                print(time_idx, end=' ', flush=True) 

            # feedback
            if time_idx > 0:
                fb_wav_buf = gen_wav_buf[:, time_idx-1:time_idx, :]
                
            # initialize skip
            skip_ch_feat *= 0

            # embedding
            hid_wav_emb = self.l_wav_emb(fb_wav_buf.squeeze(-1).to(torch.int64))

            # condition feature for current time step
            #  for other time steps, intermediate feat is saved by wave blocks
            hid_cond_tmp = hid_cond[:, time_idx:time_idx+1, :]

            # loop over wavblocks
            res_ch_feat = hid_wav_emb
            for l_wavblock in self.l_wavenet_blocks:
                res_ch_feat, tmp_skip_ch_feat = l_wavblock.inference(
                    res_ch_feat, hid_cond_tmp, time_idx)
                skip_ch_feat += tmp_skip_ch_feat

            # draw sample
            drawn_sample = self.l_output.inference(skip_ch_feat)
            gen_wav_buf[:, time_idx:time_idx+1, :] = drawn_sample


        # decode mu-law
        wave = self._waveform_decode_target(gen_wav_buf)
         
        # de-emphasis if necessary
        if self.pre_emphasis:
            for idx in range(wave.shape[1] - 1):
                wave[:, idx+1, :] = wave[:, idx+1, :] + 0.97 * wave[:, idx, :]
        return wave

    
if __name__ == "__main__":
    print("Definition of model")

    
