#!/usr/bin/env python
"""
"""
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init

import core_scripts.data_io.dsp_tools as nii_dsp_np

import sandbox.block_nn as nii_nn
import sandbox.block_dist as nii_dist
import sandbox.util_dsp as nii_dsp

import core_scripts.other_tools.debug as nii_debug

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


#######################################
# Numpy utilities for feature extraction
#######################################

def get_excit(wav, frame_length, frame_shift, lpc_order):
    lpc_handler = nii_dsp_np.LPClite(frame_length, frame_shift, lpc_order)
    lpc_coef, _, rc, gain, _, _ = lpc_handler.analysis(wav)
    return [wav, lpc_coef, rc, gain]
    

#######################################
# Network component
#######################################
class LPCLitePytorch(torch_nn.Module):
    """LPCLitePytorch
    
    This API is only used to do AR and MA operation, not LPC analysis
    
    Example:
        length = 1000
        frame_l = 320
        frame_s = 80

        wav = torch.randn([1, length, 1])
        frame_num = (length - frame_l) // frame_s + 1
        lpc_coef = torch.tensor([[1, -0.97]] * frame_num).unsqueeze(0)

        m_lpc = LPCLitePytorch(320, 80, 2)

        with torch.no_grad():
            # analysis
            wavp, excit = m_lpc.LPCAnalsysisPytorch(wav, lpc_coef)

            # synthesis
            output = torch.zeros_like(wavp)
            for idx in range(output.shape[1]):
                lpc_coef_tmp = lpc_coef[:, idx // frame_s, :].unsqueeze(1)
                output[:, idx:idx+1, :] = m_lpc.LPCSynthesisPytorch(
                    excit[:, idx:idx+1, :], lpc_coef_tmp, idx) 
            print(torch.mean(output[0] - wav[0][0:720]))
        
    """
    def __init__(self, framelen, frameshift, lpc_order, 
                 flag_emph=True, emph_coef=0.97, noise_std=0, noise_bit=8,
                 flag_emph_ignore_start=True, flag_win_begining=True):
        super(LPCLitePytorch, self).__init__()
        self.fl = framelen
        self.fs = frameshift
        self.lpc_order = lpc_order

        self.emph_coef = emph_coef
        self.noise_std = noise_std
        self.noise_mulaw_q =  2 ** noise_bit

        self.flag_emph = flag_emph
        self.flag_emph_ignore_start = flag_emph_ignore_start
        
        self.flag_win_begining = flag_win_begining
        self.m_winbuf = None
        
        self.m_buf = None
        return
    
    def _preemphasis(self, wav):
        """
        input
        -----
          wav: tensor, (batch, length, 1)
          
        output
        ------
          wav: tensor, (batch, length, 1)
        """
        wav_tmp = torch.zeros_like(wav) + wav
        wav_tmp[:, 1:] = wav_tmp[:, 1:] - self.emph_coef * wav_tmp[:, :-1]

        if self.flag_emph_ignore_start:
            wav_tmp[:, 0] = wav_tmp[:, 1]
        return wav_tmp
    
    def _deemphasis(self, wav):
        """
        input
        -----
          wav: tensor, (batch, length, 1)
          
        output
        ------
          wav: tensor, (batch, length, 1)
        """
        out = torch.zeros_like(wav) + wav
        for idx in range(1, wav.shape[1]):
            out[:, idx] = out[:, idx] + self.emph_coef * out[:, idx-1]
        return out
    
    def deemphasis(self, wav):
        """
        input
        -----
          wav: tensor, (batch, length, 1)
          
        output
        ------
          wav: tensor, (batch, length, 1)
        """
        if self.flag_emph:
            return self._deemphasis(wav)
        else:
            return wav

    def preemphasis(self, wav):
        """
        input
        -----
          wav: tensor, (batch, length, 1)
          
        output
        ------
          wav: tensor, (batch, length, 1)
        """
        if self.flag_emph:
            return self._preemphasis(wav)
        else:
            return wav
        
    def LPCAnalysisPytorch(self, wav, lpc_coef, gain=None):
        """
        input
        -----
          wav: tensor, (batch, length, 1)
          lpc_coef, tensor, (batch, frame, order)
          gain: tensor, (batch, frame, 1), or None

        output
        ------
          wavp: tensor, (batch, length2, 1)
          excit: tensor, (batch, length2, 1)
          wav_input: tensor, (batch, length2, 1), pre-processed wav

        Note that length2 depends on the frame length, shift
        length2 = min(
           ((length - frame_length) // frame_shift + 1) * frame_shift,
           lpc_coef.shape[1] * frame_shift)
        """
        if self.flag_emph:
            wav_input = self._preemphasis(wav)
        else:
            wav_input = torch.zeros_like(wav) + wav

        if self.flag_win_begining:
            if self.m_winbuf is None:
                self.m_winbuf = torch.hann_window(
                    self.fl, dtype=wav.dtype, device=wav.device)
            wav_input[:, 0:self.fl//2, 0] *= self.m_winbuf[0:self.fl//2]
        
        if self.noise_std > 0:
            wav_mulaw_noised = nii_dsp.mulaw_encode(
                wav, self.noise_mulaw_q, scale_to_int=True).to(dtype=wav.dtype)
            wav_mulaw_noised += (torch.rand_like(wav) - 0.5) * self.noise_std
            wav_mulaw_noised = wav_mulaw_noised.clamp(0, self.noise_mulaw_q-1)
            wav_feedback = nii_dsp.mulaw_decode(
                wav_mulaw_noised, self.noise_mulaw_q, input_int=True)
            wav_mulaw_noised_quan = wav_mulaw_noised.to(torch.int64)
        else:
            wav_mulaw_noised_quan = None
            wav_feedback = wav_input
        
        # 
        batch = lpc_coef.shape[0]
        # LPC oroder + 1
        poly_order = lpc_coef.shape[2]
        
        # take the minimum length
        wavlen = np.min([((wav.shape[1] - self.fl) // self.fs  + 1) * self.fs,
                         lpc_coef.shape[1] * self.fs])
        
        # to pad
        pad_zero = torch.zeros(
            [batch, poly_order-1], dtype=wav.dtype, device=wav.device)

        # (batch, length + poly_order - 1)
        # flip wavf = [x[n], x[n-1], ..., x[0], 0, 0, 0]
        wavf = torch.flip(
            torch.cat([pad_zero, wav_feedback.squeeze(-1)], dim=1), dims=[1])
        
        # (batch, length, poly_order)
        # unfold [[x[n], x[n-1], x[n-2], ], [x[n-1], x[n-2], x[n-3], ]]
        # flip back [[x[0], 0, 0...], ..., [x[n], x[n-1], x[n-2], ], ]
        # wavf[i, n, :] = [wav[n], wav[n-1], wav[n-2], ...m wav[n-order+1]]
        #   this can be used for LPC analysis for the n-th time step
        wavf = torch.flip(wavf.unfold(dimension=1, size=poly_order, step=1), 
                          dims=[1])[:, 0:wavlen, :]

        # duplicate lpc coefficients for each time step
        # (batch, length, poly_order)
        # lpcf[i, n, :] = [1, a_1, ..., a_order] for n-th step in i-th
        #   we need to pad
        lpcf = lpc_coef.repeat_interleave(self.fs, dim=1)[:, 0:wavlen, :]
        
        # excitation
        # wavf[i, n, :] * lpcf[i, n, :] = wav[n] * 1 + wav[n-1] * a_1 ...
        excit = torch.sum(wavf * lpcf, dim=-1).unsqueeze(-1)
        
        # predicted_wav = wav - excit
        wavp = wav_input[:, 0:wavlen, :] - excit
        
        if gain is not None:
            gain_tmp = gain.repeat_interleave(self.fs, dim=1)[:, 0:wavlen, :]
            excit = excit / gain_tmp

        return wavp, excit, wav_mulaw_noised_quan, wav_input

    def LPCSynthesisPytorchCore(self, lpc_coef, buf):
        """ predicted = LPCSynthesisPytorchCore(self, lpc_coef, buf):
        input
        -----
          lpc_coef: tensor, (batch, 1, poly_order)
          buf: tensor, (batch, order - 1, 1)
        output
        ------
          output: tensor, (batch, 1, 1)
        
        Compute (x[n-1] * a[2] + ...) for LP synthesis 
        This should be combined with excitation excit[n] - (x[n-1] * a[2] + ...)
        as the final output waveform
        """
        batch = lpc_coef.shape[0]
        poly_order = lpc_coef.shape[-1]
                
        # (batch, poly_order - 1)
        # flip so that data in order [x[n-1], ..., x[n-order+1]]
        pre_sample = torch.flip(buf, dims=[1]).squeeze(-1)

        
        # (batch, )
        # (x[n-1] * a[2] + ...)
        predicted = torch.sum(lpc_coef[:, 0, 1:] * pre_sample, dim=1)
        # (batch, 1, 1)
        # -(x[n-1] * a[2] + ...)
        return -1 * predicted.unsqueeze(-1).unsqueeze(-1)
            
    def LPCSynthesisPytorchStep(self, excit, lpc_coef, stepidx, gain=None):
        """
        input
        -----
          excit: tensor, (batch, 1, 1)
          lpc_coef: tensor, (batch, 1, poly_order)
          stepidx: int, which time step is this
          gain: tensor, (batch, 1, 1)

        output
        ------
          output: tensor, (batch, 1, 1)

        """
        batch = lpc_coef.shape[0]
        poly_order = lpc_coef.shape[-1]

        # if no buffer is provided. save the output here
        if stepidx == 0:
            # save as [n-order-1, ..., n-1]
            self.m_buf = torch.zeros(
                [batch, poly_order - 1, 1],
                dtype=lpc_coef.dtype, device=lpc_coef.device)
        
        
        pre = self.LPCSynthesisPytorchCore(lpc_coef, self.m_buf)
        
        # excit[n] + [- (x[n-1] * a[2] + ...)]
        if gain is None:
            output = excit + pre
        else:
            output = excit * gain + pre

        # save the previous value
        # roll and save as [x[n-order+2], ..., x[n-1], x[n]]
        self.m_buf = torch.roll(self.m_buf, -1, dims=1)
        self.m_buf[:, -1, :] = output[:, 0, :]
        return output


    def LPCSynthesisPytorch(self, excit, lpc_coef, gain=None):
        """
        input
        -----
          wav: tensor, (batch, length, 1)
          lpc_coef, tensor, (batch, frame, order)
          gain: tensor, (batch, frame, 1), or None

        output
        ------
          wavp: tensor, (batch, length2, 1)
          excit: tensor, (batch, length2, 1)
        
        Note that length2 depends on the frame length, shift
        length2 = min(
           ((length - frame_length) // frame_shift + 1) * frame_shift,
           lpc_coef.shape[1] * frame_shift)
        """
        # synthesis
        output = torch.zeros_like(excit)
        for idx in range(output.shape[1]):
            
            if gain is None:
                gain_tmp = None
            else:
                gain_tmp = gain[:, idx // self.fs, :].unsqueeze(1)
                
            output[:, idx:idx+1, :] = self.LPCSynthesisPytorchStep(
                excit[:, idx:idx+1, :], 
                lpc_coef[:, idx // self.fs, :].unsqueeze(1), idx,
                gain_tmp)
            
        if self.flag_emph:
            output = self._deemphasis(output)
        return output
    
    def LPC_rc2lpc(self, rc_input):
        """from reflection coefficients to LPC coefficients
        Based on numpy version in core_scripts/data_io/dsp_tools.py
        
        input
        -----
          rc: tensor, (batch, length, poly_order - 1)
          
        output
        ------
          lpc: tensor, (batch, length, poly_order)
        """
        # from (batch, length, poly_order - 1) to (batch * length, poly_order-1)
        batch, frame_num, order = rc_input.shape
        rc = rc_input.view(-1,  order)
        
        # (frame_num, order)
        frame_num, order = rc.shape 
        polyOrder = order + 1
        
        lpc_coef = torch.zeros([frame_num, 2, polyOrder], dtype=rc_input.dtype,
                               device=rc_input.device)
        lpc_coef[:, 0, 0] = 1.0
        
        for index in np.arange(1, polyOrder):
            lpc_coef[:, 1, index] = 1.0
            gamma = rc[:, index-1]
            lpc_coef[:, 1, 0] = -1.0 * gamma
            if index > 1:
                lpc_coef[:, 1, 1:index] = lpc_coef[:, 0, 0:index-1] \
                + lpc_coef[:, 1, 0:1] * torch.flip(lpc_coef[:, 0, 0:index-1], 
                                                   dims=[1])
            lpc_coef[:, 0, :] = lpc_coef[:, 1,:]
        
        lpc_coef = torch.flip(lpc_coef[:, 0, :], dims=[1])
        return lpc_coef.view(batch, -1, order+1)


class CondNetwork(torch_nn.Module):
    """CondNetwork(input_dim, output_dim)
    
    Predict reflection coefficients and gain factor from the input
    
    Args
    ----
      input_dim: int, input feature dimension, 
      output_dim: int, output feature dimension, 
              = dimension of reflection coeffcients + 1 for gain
    """
    def __init__(self, input_dim, output_dim):
        super(CondNetwork, self).__init__()  
        self.m_layer1 = nii_nn.Conv1dKeepLength(input_dim, input_dim, 1, 5)
        self.m_layer2 = nii_nn.Conv1dKeepLength(input_dim, input_dim, 1, 5)
        self.m_layer3 = torch_nn.Sequential(
            nii_nn.GRULayer(input_dim, 64, True),
            torch_nn.Linear(128, 128),
            torch_nn.Tanh(),
            torch_nn.Linear(128, output_dim)
        )
        self.m_act_1 = torch_nn.Tanh()
        return                                                               
                                                                             
    def forward(self, x):
        """ rc_coef, gain = CondNetwork(x)
        
        input
        -----
          x: tensor, (batch, length, input_dim)
          
        output
        ------
          rc_coef: tensor, reflection coefficients, (batch, length, out_dim - 1)
          gain: tensor, gain, (batch, length, 1)
        
        Length denotes the number of frames
        """
        x_tmp = self.m_layer3(self.m_layer2(self.m_layer1(x)) + x)
        part1, part2 = x_tmp.split([x_tmp.shape[-1]-1, 1], dim=-1)
        # self.m_act_1(part1) is the predicted reflection coefficients
        # torch.exp(part2) is the gain. 
        return self.m_act_1(part1), torch.exp(part2)

class FrameRateNet(torch_nn.Module):
    """FrameRateNet(input_dim, output_dim)
    
    Args
    ----
      input_dim: int, input feature dimension, 
      output_dim: int, output feature dimension, 
    """
    def __init__(self, input_dim, output_dim):
        super(FrameRateNet, self).__init__()
        self.m_layer1 = nii_nn.Conv1dKeepLength(input_dim, 128, 1, 3)
        self.m_layer2 = nii_nn.Conv1dKeepLength(128, 128, 1, 3)
        self.m_layer3 = torch_nn.Sequential(
            torch_nn.Linear(128, 128),
            torch_nn.Tanh(),
            torch_nn.Linear(128, output_dim),
            torch_nn.Tanh())
        return
    
    def forward(self, x):
        """y = FrameRateNet(x)
        
        input
        -----
          x: tensor, (batch, length, input_dim)
          
        output
        ------
          y: tensor, (batch, length, out_dim)
        
        Length denotes the number of frames
        """
        return self.m_layer3(self.m_layer2(self.m_layer1(x)))



class OutputNet(torch_nn.Module):
    """OutputNet(cond_dim, feedback_dim, out_dim)
    
    Args
    ----
      cond_dim: int, dimension of input condition feature
      feedback_dim: int, dimension of feedback feature
      out_dim: int, output dimension
    """
    def __init__(self, cond_dim, feedback_dim, out_dim):
        super(OutputNet, self).__init__()
        # Use the wrapper of GRULayer in nii_nn
        self.m_gru1 = nii_nn.GRULayer(cond_dim + feedback_dim, 256)
        self.m_gru2 = nii_nn.GRULayer(256 + feedback_dim, 16)
        self.m_outfc = torch_nn.Sequential(
            torch_nn.utils.weight_norm(
                torch_nn.Linear(16, 2, bias=False), name='weight')
        )
        self.m_stdact = torch_nn.ReLU()
        return
    
    def _forward(self, cond, feedback):
        """Forward for training stage
        """
        tmp = self.m_gru1(torch.cat([cond, feedback], dim=-1))
        tmp = self.m_gru2(torch.cat([tmp, feedback], dim=-1))
        return self.m_outfc(tmp)

    def _forward_step(self, cond, feedback, stepidx):
        """Forward for inference stage
        
        """
        tmp = self.m_gru1(torch.cat([cond, feedback], dim=-1), stepidx)
        tmp = self.m_gru2(torch.cat([tmp, feedback], dim=-1), stepidx)
        return self.m_outfc(tmp)

    def forward(self, cond, feedback, stepidx=None):
        """mean, std = forward(cond, feedback, stepidx=Non)
        
        input
        -----
          cond: tensor, input condition feature, (batch, length, input_dim)
          feedback: tensor, feedback feature,  (batch, length, feedback_dim)
          stepidx: int or None, the index of the time step, starting from 0
          
        output
        ------
          mean: tensor, mean of the Gaussian dist., (batch, length, out_dim//2)
          std: tensor, std of the Gaussian dist.,  (batch, length, out_dim//2)
          
        If stepidx is None, length is equal to the waveform sequence length,
        and the forward() method is in the training mode (self._forward)
        
        If stepidx is from 0 to T, length is equal to 1, and forward() only
        computes output for the t=stepidx. (self._forward_step)
          
        The nii_nn.GRULayer will save the hidden states of GRULayer. 
        Note that self._forward_step must be called from stepidx=0 to stepidx=T.
        """
        if stepidx is None:
            output = self._forward(cond, feedback)
        else:
            output = self._forward_step(cond, feedback, stepidx)
        mean, log_std = output.chunk(2, dim=-1)
        return mean, torch.exp(self.m_stdact(log_std + 9) - 9)

class LPCNetV1(torch_nn.Module):
    """LPCNetV1(framelen, frameshift, lpc_order, cond_dim, flag_fix_cond)
    
    Args
    ----
      framelen: int, frame length for LP ana/syn, in number of waveform points
      frameshift: int, frame shift for LP ana/syn, in number of waveform points 
      lpc_order: int, LP order
      cond_dim: input condition feature dimension
      flag_fix_cond: bool, whether fix the condition network or not
      
    In training, we can use a two-staged training approach: train the condition 
    network. first, then fix the condition network and train the iLPCNet. 
    flag_fix_cond is used to indicate the stage
    """
    def __init__(self, framelen, frameshift, lpc_order, cond_dim, 
                 flag_fix_cond):
        super(LPCNetV1, self).__init__()
        
        # =====
        # options
        # =====
        self.m_fl = framelen
        self.m_fs = frameshift
        self.m_lpc_order = lpc_order
        
        # dimension of input conditional feature
        self.m_cond_dim = cond_dim
        
        # =====
        # hyper-parameters
        # =====
        # dimension of output of frame-rate network
        self.m_hid_dim = 256
        # dimension of waveform
        self.m_wav_dim = 1
    
        # number of discrete pitch classes
        # We can directly fed pitch to the network, but in LPCNet the pitch
        # is quantized and embedded 
        self.m_pitch_cat = 256   
        self.m_pitch_emb = 64
        self.m_emb_f0 = torch_nn.Embedding(self.m_pitch_cat, self.m_pitch_emb)

        # =====
        # network definition
        # =====
        # lpc analyszer
        self.m_lpc = LPCLitePytorch(framelen, frameshift, lpc_order)
        
        # frame rate network
        self.m_net_framerate = FrameRateNet(cond_dim + self.m_pitch_emb, 
                                            self.m_hid_dim)
        # output network
        self.m_net_out = OutputNet(self.m_hid_dim, self.m_wav_dim, 
                                   self.m_wav_dim * 2)

        # condition network to convert conditional feature to rc coef and gain
        # (although we will not use gain)
        self.m_cond_net = CondNetwork(cond_dim, lpc_order + 1)
        
        # loss for condition network
        self.m_cond_loss = torch_nn.MSELoss()
        
        # fix
        self.flag_fix_cond = flag_fix_cond
        return

    def _negloglikelihood(self, data, mean, std):
        """ nll =  self._negloglikelihood(data, mean, std)
        neg log likelihood of normal distribution on the data
        
        input
        -----
          data: tensor, data, (batch, length, dim)
          mean: tensor, mean of dist., (batch, length, dim)
          std: tensor, std of dist., (batch, length, dim)
        
        output
        ------
          nll: scalar, neg log likelihood 
        """
        return (0.5 * np.log(2 * np.pi) \
                + torch.log(std) + 0.5 * (((data - mean)/std) ** 2)).mean()

    def _convert_pitch(self, pitch_value):
        """ output = self._convert_pitch(pitch_value)
        
        input
        -----
          pitch_value: tensor, any shape
          
        output
        ------
          output: tensor in int64, quantized pitch
        """
        return torch.clamp((pitch_value - 33) // 2, 0, 
                           self.m_pitch_cat-1).to(torch.int64)

    
    def forward(self, cond_feat, cond_feat_normed, 
                lpc_coef, rc_coef, gain, wav):
        """ loss_wav, loss_cond = forward(cond_feat, cond_feat_normed, 
            lpc_coef, rc_coef, gain, wav)
            
        input
        -----
          cond_feat: tensor, condition feature, unnormed 
                     (batch, frame_num1, cond_dim)
          cond_feat_normed: tensor, condition feature, normed 
                     (batch, frame_num1, cond_dim)
          
          lpc_coef: tensor, LP coefficients, (batch, frame_num2, lpc_order + 1)
          rc_coef: tensor, reflection coeffs (batch, frame_num2, lpc_order)
          gain: tensor, gain (batch, frame_num2, 1)
          
          wav: tensor, target waveform, (batch, length, 1)
          
        output
        ------
          loss_wav: scalar, loss for the waveform modeling (neg log likelihood)
          loss_cond: scalar, loss for the condition network
        """
        
        # == step 1 ==
        # network to convert cond_feat predict 
        #  cond -> rc_coef
        # we can compute the loss for condition network if necessary
        cond_len = np.min([rc_coef.shape[1], cond_feat.shape[1]])
        if self.flag_fix_cond:
            with torch.no_grad():
                rc_coef_pre, gain_pre = self.m_cond_net(cond_feat[:, :cond_len])
                loss_cond = (self.m_cond_loss(rc_coef, rc_coef_pre) 
                             + self.m_cond_loss(gain_pre, gain)) * 0
        else:
            rc_coef_pre, gain_pre = self.m_cond_net(cond_feat[:, :cond_len])
            loss_cond = (self.m_cond_loss(rc_coef, rc_coef_pre) 
                         + self.m_cond_loss(gain_pre, gain))
    
        # == step 2 ==
        # do LP analysis given the LP coeffs (from predicted reflection coefs)
        with torch.no_grad():
            
            # convert predicted rc_coef to LP coef
            lpc_coef_pre = self.m_lpc.LPC_rc2lpc(rc_coef_pre)
        
            # LP analysis given LPC coefficients
            # wav_pre is the LP prediction
            # wav_err is the LP residual
            # wav is the pre-processed waveform by the LP analyzer.
            #  here wav_err + wav_pre = wav
            wav_pre, wav_err, _, wav = self.m_lpc.LPCAnalysisPytorch(
                wav, lpc_coef_pre)
            
            # quantize pitch, we put this line here just for convenience
            pitch_quantized = self._convert_pitch(cond_feat[:, :cond_len, -1])

            
        # == step 3 ==
        # frame-rate network

        # embedding F0
        pitch_emb = self.m_emb_f0(pitch_quantized)
        
        # cond -> feature 
        lpccond_feat = self.m_net_framerate(
            torch.cat([cond_feat_normed[:, :cond_len, :], pitch_emb], dim=-1))

        # duplicate (upsampling) to waveform level
        lpccond_feat = lpccond_feat.repeat_interleave(self.m_fs, dim=1)
        
        # == step 4 ==
        # waveform generation network
        # 
        # get the minimum length
        wavlen = np.min([wav.shape[1],wav_pre.shape[1],lpccond_feat.shape[1]])

        # feedback waveform (add noise and dropout)  16384 = np.power(2, 14)
        noise1 = torch.randn_like(wav[:, :wavlen, :]) / 16384
        feedback_wav = torch.roll(wav[:, :wavlen, :], 1, dims=1)
        feedback_wav[:, 0, :] *= 0
        feedback_wav += noise1
        
        # compute LP residual mean and std
        mean, std = self.m_net_out(lpccond_feat[:, :wavlen, :], feedback_wav)
        
        # compute wav dist. mean and std
        mean_wav = mean + wav_pre[:, :wavlen]
        std_wav = std
        
        # get likelihood
        loss_wav = self._negloglikelihood(wav[:, :wavlen], mean_wav, std_wav)

        if self.flag_fix_cond:
            loss_cond = loss_cond * 0
        else:
            loss_wav = loss_wav * 0

        return loss_cond, loss_wav
    
    def inference(self, cond_feat, cond_feat_normed):
        """ wav = inference(cond_feat)
        input
        -----
          cond_feat: tensor, condition feature, unnormed 
                     (batch, frame_num1, cond_dim)
          cond_feat_normed, tensor, condition feature, unnormed 
                     (batch, frame_num1, cond_dim)
        output
        ------
          wav, tensor, (batch, frame_num1 * frame_shift, 1)
        """
        # prepare
        batch, frame_num, _ = cond_feat.shape
        xtyp = cond_feat.dtype
        xdev = cond_feat.device

        # quantize F0 and F0 embedding
        pitch_quantized = self._convert_pitch(cond_feat[:, :, -1])
        pitch_emb = self.m_emb_f0(pitch_quantized)

        # == step1. ==
        # predict reflection coeff from cond_feat
        # rc_coef_pre (batch, frame_num, lpc_order)
        rc_coef_pre, gain_pre = self.m_cond_net(cond_feat)
        
        # == step2. ==
        # from reflection coeff to LPC coef
        # (batch, frame_num, lpc_order + 1)
        lpc_coef_pre = self.m_lpc.LPC_rc2lpc(rc_coef_pre)
        
        # == step3. ==
        # frame rate network
        lpccond_feat = self.m_net_framerate(
            torch.cat([cond_feat_normed, pitch_emb], dim=-1))

        # == step4. ==
        # step-by-step generation
        
        # waveform length = frame num * up-sampling rate
        wavlen = frame_num * self.m_fs
        
        # many buffers
        #  buf to store output waveofmr
        wavbuf = torch.zeros([batch, wavlen, 1], dtype=xtyp, device=xdev)
        #  buf to store excitation signal
        exibuf = torch.zeros_like(wavbuf)
        #  buf to store LPC predicted wave
        prebuf = torch.zeros_like(wavbuf)
        #  buf to store LPC input x[n-1], ... x[n - poly_order+1]
        lpcbuf = torch.zeros([batch, self.m_lpc_order, 1], 
                             dtype=xtyp, device=xdev)        
        #  mean and std buf 
        meanbuf = torch.zeros_like(wavbuf)
        stdbuf = torch.zeros_like(wavbuf)
        
        #  loop
        for idx in range(0, wavlen):
            if idx % 1000 == 0:
                print(idx, end=' ', flush=True)
            frame_idx = idx // self.m_fs

            # 4.1 LP predicted wav 
            # [- (x[n-1] * a_1 + x[n-2] * a_2 ... )]
            pre_raw = self.m_lpc.LPCSynthesisPytorchCore(
                lpc_coef_pre[:, frame_idx : frame_idx+1, :], lpcbuf)
            # save it (for debugging)
            prebuf[:, idx:idx+1, :] = pre_raw
            
            # 4.2 predict excitation
            if idx == 0:
                wav_raw = torch.zeros_like(pre_raw)
            else:
                pass
                
            #   mean, std
            mean, std = self.m_net_out(lpccond_feat[:, frame_idx:frame_idx+1], 
                                       wav_raw, idx)
            meanbuf[:, idx:idx+1, :]= mean
            stdbuf[:, idx:idx+1, :] = std
            
            #   sampling
            exi_raw = torch.randn_like(mean) * std * 0.7 + mean

            #   save excit (for debugging)
            exibuf[:, idx:idx+1,:] = exi_raw 
                        
            # 4.3 waveform output
            #   excit[n] + [-(x[n-1] * a_1 + x[n-2] * a_2 ... )]
            wav_raw = exi_raw + pre_raw
            
            #    save waveform 
            wavbuf[:, idx:idx+1,:] = wav_raw
            #    save it to the LPC buffer. 
            #    It will be used by LPCSynthesisPytorchCore
            lpcbuf = torch.roll(lpcbuf, -1, dims=1)
            lpcbuf[:, -1, :] = wav_raw[:, 0, :]
        
        return self.m_lpc.deemphasis(wavbuf)


if __name__ == "__main__":
    print("Components of LPCNet")
