#!/usr/bin/env python
"""
Building blocks for Blow

Serra, J., Pascual, S. & Segura, C. Blow: a single-scale hyperconditioned flow 
for non-parallel raw-audio voice conversion. in Proc. NIPS (2019). 

Reference: https://github.com/joansj/blow
"""
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init

import sandbox.block_glow as nii_glow
import core_scripts.data_io.wav_tools as nii_wav_tk
import core_scripts.data_io.conf as nii_io_conf
import core_scripts.other_tools.debug as nii_debug

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


#######################################
# Numpy utilities for data augmentation
#######################################
def flip(x):
    """y=flip(x) flips the sign of x
    input: x, np.array
    output: y, np.array
    """
    return np.sign(np.random.rand(1)-0.5) * x

def ampscale(x):
    """y=ampscale(x) randomly scale the amplitude of x
    input: x, np.array
    output: y, np.array
    """
    return (2*np.random.rand(1)-1) * x / (np.max(np.abs(x)) + 1e-07)

def framejitter(x, framelen):
    """y=framejitter(x, framelen)
    input: x, np.array, original waveform (length, 1)
           framelen, int, framelen
    output: y, np.array, segment of the waveform
    """
    framelen = x.shape[0] if framelen > x.shape[0] else framelen
    random_start = int(np.ceil(np.random.rand(1) * (x.shape[0] - framelen)))
    return x[random_start:random_start+framelen]


def emphasis_rand(x, coef_val):
    """y=deemphasis(x, coef_val)
    input: x, np.array, original waveform (length, 1) or (length) 
           framelen, int, framelen
    output: y, np.array, segment of the waveform
    """
    coef = (2 * np.random.rand(1) - 1) * coef_val
    x_new = np.zeros_like(x) + x
    x_new[1:] = x_new[1:] - coef * x[:-1]
    return x_new

def wav_aug(x, framelen, coef_val, sr):
    """y = wav_aug(x, framelen, coef_val, sr)
    input
    -----
      x: np.array, original waveform (length, 1) 
      framelen: int, frame length
      coef_val: float, reference coefficient for emphasis-rand
      sr: int, sampling rate (e.g., 16000)
    
    output
    ------
      y: np.array, pre-processed waveform (length, 1)
    """
    trimmed_x = nii_wav_tk.silence_handler_wrapper(x, sr, flag_output=1)
    x_frame = framejitter(trimmed_x, framelen)
    return ampscale(emphasis_rand(x_frame, coef_val))



class OverlapAdder(torch_nn.Module):
    """OverlapAdder
    """
    def __init__(self, fl, fs, flag_win_analysis=True):
        """OverlapAdder(flag_windowing_before=True)
        Args
        ----
          fl: int, frame length
          fs: int, frame shift
          flag_win_analysis: bool (default True)
              True: apply windowing during analysis
              False: apply windowing during synthesis          
        """
        super(OverlapAdder, self).__init__()
        self.fl = fl
        self.fs = fs
        self.flag_win_ana = flag_win_analysis
        
        # assume even 
        self.m_win = torch_nn.Parameter(torch.hann_window(self.fl))
        return
    
    def get_frame_num(self, wav_length):
        """frame_num = get_frame_num(wav_length)
        wav_length: int, waveform length
        frame_num: int, number of frames
        """
        return (wav_length - self.fl) // self.fs + 1
    
    def get_wavlength(self, frame_num):
        """wav_length = get_wavlength(self, frame_num)
        wav_length: int, waveform length
        frame_num: int, number of frames
        """
        return (frame_num - 1) * self.fs + self.fl
    
    def forward(self, x):
        """OverlapAdder(x)
        
        input
        -----
          x: tensor, (batch, length, 1)
        
        output
        ------
          y: tensor, (batch, frame_num, frame_length)
        """
        frame_num = self.get_frame_num(x.shape[1])
        
        # (batch, num_patches, 1, patch_size)
        # num_patches = (length - length) // shift + 1
        # and copy the data
        # note that unfold put each patch as the last dimension
        # x_tmp (batch, frame_num, 1, frame_length)
        x_tmp = x.unfold(1, self.fl, self.fs)

        # apply window
        if self.flag_win_ana:
            x_tmp = x_tmp * self.m_win
            
        # (batch, frame_num, frame_length)
        return x_tmp.view(x.shape[0], x_tmp.shape[1], -1)

    def reverse(self, x_framed, flag_scale=False):
        """OverlapAdder(x)
        
        input
        -----
          x: tensor, (batch, frame_num, frame_length)
          flag_scale: bool, whether scale the ampltidue to (-1, 1)
                      default False
        output
        ------
          y: tensor, (batch, length, 1)
        """
        batch, frame_num, frame_len = x_framed.shape
        x_len = self.get_wavlength(frame_num)
        x_buf = torch.zeros(
            [batch, x_len], device=x_framed.device, dtype=x_framed.dtype)
        x_win = torch.zeros_like(x_buf)
        
        for idx in range(frame_num):
            sdx = idx * self.fs
            edx = sdx + self.fl
            x_win[:, sdx:edx] += self.m_win
            if not self.flag_win_ana: 
                x_buf[:, sdx:edx] += x_framed[:, idx] * self.m_win
            else:
                x_buf[:, sdx:edx] += x_framed[:, idx]
        # assume the overlapped window has a constant amplitude
        x_buf = x_buf / x_win.mean()

        # normalize the amplitude between (-1, 1)
        if flag_scale:
            # if input is between (-1, 1), there is no need to 
            # do this normalization
            x_buf = x_buf / (x_buf.abs().max())
        return x_buf.unsqueeze(-1)

#######################################
# Torch model definition
#######################################

class AffineCouplingBlow_core(torch_nn.Module):
    """AffineCouplingBlow_core
    
    AffineCoupling core layer the produces the scale and bias parameters.
    
    Example:
        feat_dim = 10
        cond_dim = 20

        m_layer = AffineCouplingBlow_core(feat_dim, cond_dim, 64, 2)

        data = torch.randn([2, 100, feat_dim])
        cond = torch.randn([2, 1, cond_dim])
        scale, bias, log_scale = m_layer(data, cond)
    """
    def __init__(self, feat_dim, cond_dim, num_ch, kernel_size=3):
        """AffineCouplingBlow_core(feat_dim, cond_dim, num_ch, kernel_size=3)
        
        Args
        ----
          feat_dim: int, dimension of input feature
          cond_dim: int, dimension of conditional features
          num_ch: int, number of channels for conv layers
          kernel_size: int, kernel size of conv layer, default 3
          
        input_feature -------> func.conv1d -----> conv1ds -> scale, bias
                                    ^
                                    |
        cond_dim ---> Adapter -> conv weight/bias
        """
        super(AffineCouplingBlow_core, self).__init__()
        
        self.feat_dim = feat_dim
        self.cond_dim = cond_dim
        
        # make sure that kernel is odd
        if kernel_size % 2 == 0:
            self.kernel_s = kernel_size + 1
            print("\tAffineCouplingBlow_core", end=" ")
            print("kernel size {:d} -> {:d}".format(kernel_size, self.kernel_s))
        else:
            self.kernel_s = kernel_size
            
        if num_ch % feat_dim != 0:
            # make sure that number of channel is good
            self.num_ch = num_ch // feat_dim * feat_dim
            print("\tAffineCouplingBlow_core", end=" ")
            print("conv channel {:d} -> {:d}".format(num_ch, self.num_ch))
        else:
            self.num_ch = num_ch
            
        # Adapter
        # (batch, 1, cond_dim) -> (batch, 1, kernel_size * num_ch) for weight
        #                      -> (batch, 1, num_ch) for bias
        self.m_adapter = torch_nn.Linear(cond_dim, 
                                         (self.kernel_s+1) * self.num_ch)
        
        # conv1d with condition-independent parameters
        self.m_conv1ds = torch_nn.Sequential(
            torch_nn.ReLU(),
            torch_nn.Conv1d(self.num_ch, self.num_ch, 1),
            torch_nn.ReLU(),
            torch_nn.Conv1d(self.num_ch, feat_dim * 2, self.kernel_s, 
                           padding=(self.kernel_s-1)//2)
        )
        
        # zero initialization for the last conv layers
        # similar to Glow and WaveGlow
        self.m_conv1ds[-1].weight.data.zero_()
        self.m_conv1ds[-1].bias.data.zero_()
        return
    
    def forward(self, x, cond):
        """scale, bias = AffineCouplingBlow_core(x, cond)
        
        input
        -----
          x: tensor, input tensor (batch, length, feat_dim)
          cond: tensor, condition feature (batch, 1, cond_dim)
        
        output
        ------
          scale: tensor, scaling parameters (batch, length, feat_dim)
          bias: tensor, bias paramerters (batch, length, feat_dim) 
        """
        # cond_dim -> Adapter -> conv weight/bias
        # cond[:, 0, :] -> (batch, cond_dim)
        # adapter(cond[:, 0, :]) -> (batch, kernel_size * num_ch + num_ch)
        # view(...) -> (batch * num_ch, kernel_size + 1)
        weight_bias = self.m_adapter(cond[:, 0, :]).view(-1, self.kernel_s+1)
        # (batch * num_ch, 1, kernel_size)
        weight = weight_bias[:, 0:self.kernel_s].unsqueeze(1)
        # (batch * num_ch)
        bias = weight_bias[:, self.kernel_s]
        
        # convolution given weight_bias
        padsize = (self.kernel_s - 1) // 2
        groupsize = x.shape[0] * self.feat_dim
        length = x.shape[1]
        
        #  x.permute(0, 2, 1)...view -> (1, batch*feat_dim, length)
        #  conv1d -> (1, batch * num_ch, length)
        #  view -> (batch, num_ch, length)
        x_tmp = torch_nn_func.conv1d(
            x.permute(0, 2, 1).contiguous().view(1, -1, length),
            weight, 
            bias = bias,
            padding = padsize,
            groups = groupsize
        ).view(x.shape[0], -1, length)
        
        # condition invariant conv -> (batch, feat_dim * 2, length)
        x_tmp = self.m_conv1ds(x_tmp)
        
        # scale and bias (batch, feat_dim, length)
        raw_scale, bias = torch.chunk(x_tmp, 2, dim=1)

        #  -> (batch, length, feat_dim)
        bias = bias.permute(0, 2, 1)

        #  re-parameterize
        #   Here we need to add a small number, otherwise, log(scale)
        #   somtime times become -inf during training
        scale = torch.sigmoid(raw_scale + 2).permute(0, 2, 1) * 0.5 + 0.5
        
        log_scale = torch.log(scale)

        #print("Debug: {:.3f} {:.3f} {:.3f} {:3f}".format(
        #    log_scale.max().item(), log_scale.min().item(), 
        #    scale.max().item(), scale.min().item()), 
        #    file=sys.stderr)
        return scale, bias, log_scale
    
    
    
class AffineCouplingBlow(torch_nn.Module):
    """AffineCouplingBlow
    
    AffineCoupling block in Blow
    
    Example:
        feat_dim = 10
        cond_dim = 20

        m_layer = AffineCouplingBlow(feat_dim, cond_dim,60,3, flag_detjac=True)

        data = torch.randn([2, 100, feat_dim])
        cond = torch.randn([2, 1, cond_dim])
        out, detjac = m_layer(data, cond)

        data_rever = m_layer.reverse(out, cond)

        torch.std(data - data_rever)
    """
    def __init__(self, in_dim, cond_dim,  
                 conv_dim_channel, conv_kernel_size,
                 flag_detjac=False):
        """AffineCouplingBlow(in_dim, cond_dim,  
            wn_num_conv1d, wn_dim_channel, wn_kernel_size, 
            flag_affine=True, flag_detjac=False)
        
        Args:
        -----
          in_dim: int, dim of input audio data (batch, length, in_dim)
          cond_dim, int, dim of condition feature (batch, length, cond_dim)
          conv_dim_channel: int, dime of the convolution channels
          conv_kernel_size: int, kernel size of the convolution layers
          flag_detjac: bool, whether return the determinant of Jacobian,
                       default False
        
        y -> split() -> y1, y2 -> concate([y1, (y2+bias) * scale])
        When flag_affine == True, y1 -> H() -> scale, bias
        When flag_affine == False, y1 -> H() -> bias, scale=1 
        Here, H() is AffineCouplingBlow_core layer
        """
        super(AffineCouplingBlow, self).__init__()
        
        self.flag_detjac = flag_detjac
        
        if in_dim % 2 > 0:
            print("AffineCouplingBlow(feat_dim), feat_dim is an odd number?!")
            sys.exit(1)
        
        # Convolution block to get scale and bias
        self.m_core = AffineCouplingBlow_core(
            in_dim // 2, cond_dim, conv_dim_channel, conv_kernel_size)
        
        return
    
    def _detjac(self, log_scale, factor=1):
        # (batch, dim1, dim2, ..., feat_dim) -> (batch)
        # sum over dim1, ... feat_dim
        return nii_glow.sum_over_keep_batch(log_scale / factor)
        
    def _nn_trans(self, y1, cond):
        """_nn_trans(self, y1, cond)
        
        input
        -----
          y1: tensor, input feature, (batch, lengh, input_dim//2)
          cond: tensor, condition feature, (batch, length, cond_dim)
          
        output
        ------
          scale: tensor, (batch, lengh, input_dim // 2)
          bias: tensor, (batch, lengh, input_dim // 2)
          log_scale: tensor, (batch, lengh, input_dim // 2)
        
        Affine transformaiton can be done by scale * feature + bias
        log_scale is used for det Jacobian computation
        """
        scale, bias, log_scale = self.m_core(y1, cond)
        return scale, bias, log_scale
        
    def forward(self, y, cond, factor=1):
        """AffineCouplingBlow.forward(y, cond)
        
        input
        -----
          y: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, 1, cond_dim)
          
        output
        ------
          x: tensor, input feature, (batch, lengh, input_dim)
          detjac: tensor, det of jacobian, (batch,)
        
        y1, y2 = split(y)
        scale, bias = Conv(y1)
        x2 = y2 * scale + bias or (y2 + bias) * scale
        return [y1, x2]
        """
        # split
        y1, y2 = y.chunk(2, -1)
        scale, bias, log_scale = self._nn_trans(y1, cond)
        
        # transform
        x1 = y1
        x2 = (y2 + bias) * scale

        # concatenate
        x = torch.cat([x1, x2], dim=-1)
        if self.flag_detjac:
            return x, self._detjac(log_scale, factor)
        else:
            return x
        
        
    def reverse(self, x, cond):
        """AffineCouplingBlow.reverse(y, cond)
        
        input
        -----
          x: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, 1, cond_dim)
          
        output
        ------
          y: tensor, input feature, (batch, lengh, input_dim)
        
        x1, x2 = split(x)
        scale, bias = conv(x1)
        y2 = x2 / scale - bias
        return [x1, y2]
        """
        # split
        x1, x2 = x.chunk(2, -1)
        # reverse transform
        y1 = x1
        scale, bias, log_scale = self._nn_trans(y1, cond)
        y2 = x2 / scale - bias
        return torch.cat([y1, y2], dim=-1)
        
        
class SqueezeForBlow(torch_nn.Module):
    """SqueezeForBlow
    
    Squeeze input feature for Blow.
    
    Example
        data = torch.randn([2, 10, 3])
        m_sq = SqueezeForBlow()
        data_out = m_sq(data)
        data_rev = m_sq.reverse(data_out)
        torch.std(data_rev - data)
    """
    def __init__(self, mode=1):
        """SqueezeForBlow(mode=1)
        
        Args
        ----
          mode: int, mode of squeeze, default 1
          
        Mode 1: squeeze by a factor of 2 as in original paper
        """
        super(SqueezeForBlow, self).__init__()
        
        self.m_mode = mode
        
        if self.m_mode == 1:
            self.squeeze_factor = 2
        else:
            print("SqueezeForBlow mode {:d} not implemented".format(mode))
            sys.exit(1)
        return

    
    def get_expected_squeeze_length(self, orig_length):
        # return expected length after squeezing
        if self.m_mode == 1:
            return orig_length // self.squeeze_factor
        else:
            print("unknown mode for SqueezeForBlow")
            sys.exit(1)
    
    def get_recovered_length(self, squeezed_length):
        # return original length before squeezing
        if self.m_mode == 1:
            return squeezed_length * self.squeeze_factor
        else:
            print("unknown mode for SqueezeForBlow")
            sys.exit(1)
    
    def get_squeeze_factor(self):
        # return the configuration for squeezing
        if self.m_mode == 1:
            return self.squeeze_factor
        else:
            print("unknown mode for SqueezeForBlow")
            sys.exit(1)
    
    def forward(self, x):
        """SqueezeForBlow(x)
        
        input
        -----
          x: tensor, (batch, length, feat_dim)
        
        output
        ------
          y: tensor, (batch, length//squeeze_factor, feat_dim*squeeze_factor)
        
        """
        if self.m_mode == 1:
            # squeeze, the 8 points should be the last dimension
            squeeze_len = self.get_expected_squeeze_length(x.shape[1])
            # trim length first
            trim_len = squeeze_len * self.squeeze_factor
            x_tmp = x[:, 0:trim_len, :]
            
            # (batch, time//squeeze_size, squeeze_size, dim)
            x_tmp = x_tmp.view(x_tmp.shape[0], squeeze_len, 
                               self.squeeze_factor, -1)
            
            # (batch, time//squeeze_size, dim, squeeze_size)
            x_tmp = x_tmp.permute(0, 1, 3, 2).contiguous()
            
            # (batch, time//squeeze_size, dim * squeeze_size)
            return x_tmp.view(x_tmp.shape[0], squeeze_len, -1)
        else:
            print("SqueezeForWaveGlow not implemented")
            sys.exit(1)
        return x_squeezed

    def reverse(self, x_squeezed):
        if self.m_mode == 1:
            # (batch, time//squeeze_size, dim * squeeze_size)
            batch, squeeze_len, squeeze_dim = x_squeezed.shape
            
            # (batch, time//squeeze_size, dim, squeeze_size)
            x_tmp = x_squeezed.view(
                batch, squeeze_len, squeeze_dim // self.squeeze_factor, 
                self.squeeze_factor)
            
            # (batch, time//squeeze_size, squeeze_size, dim)
            x_tmp = x_tmp.permute(0, 1, 3, 2).contiguous()
            
            # (batch, time, dim)
            x = x_tmp.view(batch, squeeze_len * self.squeeze_factor, -1)
        else:
            print("SqueezeForWaveGlow not implemented")
            sys.exit(1)
        return x

    
    
class FlowStepBlow(torch_nn.Module):
    """FlowStepBlow
    One flow step for Blow
    y -> intertical_1x1() -> ActNorm -> AffineCoupling -> x
    
    Example
        feat_dim = 10
        cond_dim = 20

        m_layer = FlowStepBlow(feat_dim, cond_dim, 60, 3)

        data = torch.randn([2, 100, feat_dim])
        cond = torch.randn([2, 1, cond_dim])
        out, detjac = m_layer(data, cond)

        data_rever = m_layer.reverse(out, cond)

        torch.std(data - data_rever) 
    """
    def __init__(self, in_dim, cond_dim, conv_dim_channel, conv_kernel_size):
        """FlowStepBlow(in_dim, cond_dim, 
                            conv_dim_channel, conv_kernel_size)
        
        Args
        ----
          in_dim: int, input feature dim, (batch, length, in_dim)
          cond_dim:, int, conditional feature dim, (batch, length, cond_dim)
          cond_dim_channel: int, dim of the convolution layers
          conv_kernel_size: int, kernel size of the convolution layers

        For cond_dim_channel and conv_kernel_size, see AffineCouplingBlow
        """
        super(FlowStepBlow, self).__init__()
        
        # Invertible transformation layer
        self.m_invtrans = nii_glow.InvertibleTrans(in_dim, flag_detjac=True)
        
        # Act norm layer
        self.m_actnorm = nii_glow.ActNorm(in_dim, flag_detjac=True)
        
        # coupling layer
        self.m_coupling = AffineCouplingBlow(
            in_dim, cond_dim, conv_dim_channel, conv_kernel_size, 
            flag_detjac=True)
        
        return
    
    def forward(self, y, cond, factor=1):
        """FlowStepBlow.forward(y, cond, factor=1)
        
        input
        -----
          y: tensor, input feature, (batch, lengh, in_dim)
          cond: tensor, condition feature , (batch, 1, cond_dim)
          factor: int, this is used to divde likelihood, default 1
                  if we directly sum all detjac, they will become very large
                  however, we cannot average them directly on y because y
                  may have a different shape from the actual data y
        output
        ------
          x: tensor, input feature, (batch, lengh, input_dim)
          detjac: tensor, det of jacobian, (batch,)
        """
        # 1x1 transform
        x_tmp, log_det_1 = self.m_invtrans(y, factor)
        
        # Actnorm
        x_tmp, log_det_2 = self.m_actnorm(x_tmp, factor)
        
        # coupling
        x_tmp, log_det_3 = self.m_coupling(x_tmp, cond, factor)
        return x_tmp, log_det_1 + log_det_2 + log_det_3
    
    def reverse(self, x, cond):
        """FlowStepBlow.reverse(y, cond)
        
        input
        -----
          x: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, 1, cond_dim)
          
        output
        ------
          y: tensor, input feature, (batch, lengh, input_dim)
        """
        y_tmp1 = self.m_coupling.reverse(x, cond) 
        y_tmp2 = self.m_actnorm.reverse(y_tmp1) 
        y_tmp3 = self.m_invtrans.reverse(y_tmp2)
        #print("Debug: {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        #    y_tmp1.max().item(), y_tmp1.min().item(),
        #    y_tmp2.max().item(), y_tmp2.min().item(),
        #    y_tmp3.max().item(), y_tmp3.min().item()))
        return y_tmp3


    
class BlowBlock(torch_nn.Module):
    """BlowBlock
    A BlowBlok includes multiple steps of flow for Blow.
    
    Each block conducts:
    x -> squeeze -> flow step1 -> ... -> flow step N

    Compared with WaveGlowBlock, this is easier because there is no
    multi-scale structure, no need to split the latent z.
    
    Example:
        
    """
    def __init__(self, in_dim, cond_dim, n_flow_steps,
                 conv_dim_channel, conv_kernel_size):
        """BlowBlock(in_dim, cond_dim, n_flow_steps,
                 conv_dim_channel, conv_kernel_size)
        Args
        ----
          in_dim: int, input feature dim, (batch, length, in_dim)
          cond_dim:, int, conditional feature dim, (batch, length, cond_dim)
          n_flow_steps: int, number of flow steps in one block
          conv_dim_channel: int, dim of the conv residual and skip channels
          conv_kernel_size: int, kernel size of the convolution layers
         
        For conv_dim_channel and conv_kernel_size, see AffineCouplingBlow
        """
        super(BlowBlock, self).__init__()
        
        # squeeze
        self.m_squeeze = SqueezeForBlow()
        
        squeezed_feat_dim = in_dim * self.m_squeeze.get_squeeze_factor()
                
        # flow steps
        tmp_flows = []
        for i in range(n_flow_steps):
            tmp_flows.append(
                FlowStepBlow(
                    squeezed_feat_dim, cond_dim, 
                    conv_dim_channel, conv_kernel_size))
        self.m_flows = torch_nn.ModuleList(tmp_flows)            
        
        self.m_out_dim = squeezed_feat_dim
        return
    
    def get_out_feat_dim(self):
        return self.m_out_dim
    
    def get_expected_squeeze_length(self, orig_length):
        return self.m_squeeze.get_expected_squeeze_length(orig_length)
    
    def forward(self, y, cond, factor=1):
        """z, log_detjac = BlowBlock(y) 
        
        y -> squeeze -> H() -> z, log_det_jacobian
        H() consists of multiple flow steps (1x1conv + Actnorm + AffineCoupling)
        
        input
        -----
          y: tensor, (batch, length, dim)
          cond, tensor, (batch, 1, cond_dim)
          factor, None or int, this is used to divde likelihood, default 1

        output
        ------
         log_detjac: tensor or scalar
         z: tensor, (batch, length, dim), for N(z; 0, I) or next flow block
        """
        # squeeze
        x_tmp = self.m_squeeze(y)

        # flows
        log_detjac = 0
        for idx, l_flow in enumerate(self.m_flows):
            x_tmp, log_detjac_tmp = l_flow(x_tmp, cond, factor)
            log_detjac = log_detjac + log_detjac_tmp
            
        return x_tmp, log_detjac
    
    def reverse(self, z, cond):
        """y = BlowBlock.reverse(z, cond) 
        
        z -> H^{-1}() -> unsqueeze -> y
        
        input
        -----
          z: tensor, (batch, length, in_dim)
          cond, tensor, (batch, 1, cond_dim)
          
        output
        ------
          y: tensor, (batch, length, in_dim)          
        """
        y_tmp = z
        for l_flow in self.m_flows[::-1]:
            y_tmp = l_flow.reverse(y_tmp, cond)
        y = self.m_squeeze.reverse(y_tmp)
        return y

    
class Blow(torch_nn.Module):
    """Blow                     
    """
    def __init__(self, cond_dim, num_blocks, num_flows_inblock, 
                 conv_dim_channel, conv_kernel_size):
        """Blow(cond_dim, num_blocks, num_flows_inblock, 
                conv_dim_channel, conv_kernel_size)
        
        Args
        ----
          cond_dim:, int, conditional feature dim, (batch, length, cond_dim)
          num_blocks: int, number of WaveGlowBlocks
          num_flows_inblock: int, number of flow steps in one WaveGlowBlock
          conv_dim_channel: int, dim of convolution layers channels
          conv_kernel_size: int, kernel size of the convolution layers
          
        This model defines:
        
        cond (global) ----- -> | ------> | --------> | 
                               v         v           v   
        y --------------> BlowBlock1 -> BlowBlock2 -> ... -> z
        """
        super(Blow, self).__init__()
        
        # input is assumed to be waveform
        self.m_input_dim = 1
        
        # save the dimension for get_z_noises
        self.m_z_dim = 0

        # define blocks
        tmp_squeezed_in_dim = self.m_input_dim
        tmp_flow_blocks = []
        for i in range(num_blocks):
            tmp_flow_blocks.append(
                BlowBlock(
                    tmp_squeezed_in_dim, cond_dim, num_flows_inblock,
                    conv_dim_channel, conv_kernel_size))
            
            tmp_squeezed_in_dim = tmp_flow_blocks[-1].get_out_feat_dim()
        self.m_z_dim = tmp_squeezed_in_dim
        
        self.m_flowblocks = torch_nn.ModuleList(tmp_flow_blocks)
        
        # done
        return
    
    
    def get_expected_squeeze_length(self, wave_length):
        """length = get_expected_squeeze_length(self, wave_length)
        Return expected length of latent z
        
        input
        -----
          wave_length: int, length of original waveform
        
        output
        ------
          length: int, length of latent z
        """
        
        length = wave_length
        for glowblock in self.m_flowblocks:
            length = glowblock.get_expected_squeeze_length(length)
        return length
    
    def _normal_lh(self, noise):
        # likelihood of normal distribution on the given noise
        return -0.5 * np.log(2 * np.pi) - 0.5 * noise ** 2
    
    def forward(self, y, cond):
        """z, neg_logp_y, logp_z, logdet = Blow.forward(y, cond) 
        
        cond (global) ----- -> | ------> | --------> | 
                               v         v           v   
        y --------------> BlowBlock1 -> BlowBlock2 -> ... -> z
                             
        input
        -----
          y: tensor, (batch, waveform_length, 1)
          cond: tensor,  (batch, 1, cond_dim)
          
        output
        ------
          z: tensor
          neg_logp_y: scalar, - log p(y)
          logp_z: scalar, -log N(z), summed over one data sequence, but averaged
                  over batch.
          logdet: scalar, -|det dH(.)/dy|, summed over one data sequence, 
                  but averaged
                  over batch.
        """
        
        # Rather than summing the likelihood and divide it by the number of 
        #  data in the final step, we divide this factor from the likelihood
        #  caculating by each flow step and sum the scaled likelihood. 
        # Two methods are equivalent, but the latter may prevent numerical 
        #  overflow of the likelihood value for long sentences
        factor = np.prod([dim for dim in y.shape])
        
        # flows
        log_detjac = 0
        log_pz = 0

        x_tmp = y
        for m_block in self.m_flowblocks:
            x_tmp, log_detjac_tmp = m_block(
                x_tmp, cond, factor)
            
            # accumulate log det jacobian
            log_detjac += log_detjac_tmp
        
        z_tmp = x_tmp
        # compute N(z; 0, I)
        # accumulate log_N(z; 0, I) only if it is valid
        if z_tmp is not None:
            log_pz += nii_glow.sum_over_keep_batch2(
                self._normal_lh(z_tmp), factor)
        
        # average over batch and data points
        neg_logp_y = -(log_pz + log_detjac).sum()
        return z_tmp, neg_logp_y, \
            log_pz.sum(), log_detjac.sum()
        
    def reverse(self, z, cond):
        """y = Blow.reverse(z_bags, cond) 
        
        cond (global) ----- -> | ------> | --------> | 
                               v         v           v   
        y <--------------- BlowBlock1 <- BlowBlock2 <- ... <- z
                             
        input
        -----
          z: tensor, shape decided by the model configuration
          cond: tensor,  (batch, 1, cond_dim)
          
        output
        ------
          y: tensor, (batch, waveform_length, 1)
        """
        # initial
        y_tmp = z
        for m_block in self.m_flowblocks[::-1]:
            y_tmp = m_block.reverse(y_tmp, cond)
        return y_tmp
    
    def get_z_noises(self, length, noise_std=0.7, batchsize=1):
        """z_bags = Blow.get_z_noises(length, noise_std=0.7, batchsize=1)
        Return random noise for random sampling
        
        input
        -----
          length: int, length of target waveform (without squeeze)
          noise_std: float, std of Gaussian noise, default 0.7
          batchsize: int, batch size of this random data, default 1
        
        output
        ------
          z: tensor, shape decided by the network
        
        Blow.reverse(z, cond) can be used to generate waveform
        """
        squeeze_length = self.get_expected_squeeze_length(length)
        
        device = next(self.parameters()).device
        
        z_tmp = torch.randn(
            [batchsize, squeeze_length, self.m_z_dim], 
            dtype=nii_io_conf.d_dtype, 
            device=device)
        return z_tmp
    
    
if __name__ == "__main__":
    print("Definition of Blow")
