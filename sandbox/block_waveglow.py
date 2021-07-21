#!/usr/bin/env python
"""
Building blocks for waveglow

"""
from __future__ import absolute_import

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init

import sandbox.block_nn as nii_nn
import sandbox.block_wavenet as nii_wavenet
import sandbox.block_glow as nii_glow
import core_scripts.data_io.conf as nii_io_conf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


class Invertible1x1ConvWaveGlow(torch.nn.Module):
    def __init__(self, feat_dim, flag_detjac=False):
        super(Invertible1x1ConvWaveGlow, self).__init__()

        torch.manual_seed(100)
        with torch.no_grad():
            W = torch.qr(torch.FloatTensor(feat_dim, feat_dim).normal_())[0]
            
            # Ensure determinant is 1.0 not -1.0
            if torch.det(W) < 0:
                W[:,0] = -1*W[:,0]
                
            # not necessary
            W = W.transpose(0, 1)
        self.weight = torch_nn.Parameter(W)
        self.weight_inv = torch_nn.Parameter(W.clone())
        self.weight_inv_flag = False
        self.flag_detjac = flag_detjac
        return
    
    def forward(self, y, factor):
        batch_size, length, feat_dim = y.size()

        # Forward computation
        log_det_W = length / factor * torch.logdet(self.weight)
        z = torch.matmul(y, self.weight)
        if self.flag_detjac:
            return z, log_det_W
        else:
            return z
    
    def reverse(self, x):
        if not self.weight_inv_flag:
            self.weight_inv.data = torch.inverse(self.weight.data)
            self.weight_inv_flag = True
        return torch.matmul(x, self.weight_inv)

class upsampleByTransConv(torch_nn.Module):
    """upsampleByTransConv
    Upsampling layer using transposed convolution
    """
    def __init__(self, feat_dim, upsample_rate, window_ratio=5):
        """upsampleByTransConv(feat_dim, upsample_rate, window_ratio=5)
        
        Args
        ----
          feat_dim: int, input feature should be (batch, length, feat_dim)
          upsample_rate, int, output feature will be 
                (batch, length*upsample_rate, feat_dim)
          window_ratio: int, default 5, window length of transconv will be 
                upsample_rate * window_ratio
        """
        super(upsampleByTransConv, self).__init__()
        window_l = upsample_rate * window_ratio
        self.m_layer = torch_nn.ConvTranspose1d(
            feat_dim, feat_dim, window_l, stride=upsample_rate)
        self.m_uprate = upsample_rate
        return
    
    def forward(self, x):
        """ y = upsampleByTransConv(x)
        
        input
        -----
          x: tensor, (batch, length, feat_dim)
          
        output
        ------
          y: tensor, (batch, length*upsample_rate, feat_dim)
        """
        l = x.shape[1] * self.m_uprate
        y = self.m_layer(x.permute(0, 2, 1))[:, :, 0:l]
        return y.permute(0, 2, 1).contiguous()

class SqueezeForWaveGlow(torch_nn.Module):
    """SqueezeForWaveGlow
    Squeeze layer for WaveGlow
    """
    def __init__(self, mode = 1):
        """SqueezeForGlow(mode=1)
        Args
        ----
          mode: int, mode of this squeeze layer
          
        mode == 1: original squeeze method by squeezing 8 points
        """
        super(SqueezeForWaveGlow, self).__init__()
        self.m_mode = mode
        # mode 1, squeeze by 8
        self.m_mode_1_para = 8
        return
    
    def get_expected_squeeze_length(self, orig_length):
        # return expected length after squeezing
        if self.m_mode == 1:
            return orig_length//self.m_mode_1_para
    
    def get_squeeze_factor(self):
        # return the configuration for squeezing
        if self.m_mode == 1:
            return self.m_mode_1_para
    
    def forward(self, x):
        """SqueezeForWaveGlow(x)
        
        input
        -----
          x: tensor, (batch, length, feat_dim)
        
        output
        ------
          y: tensor, (batch, length // squeeze, feat_dim * squeeze)
        """
        if self.m_mode == 1:
            # squeeze, the 8 points should be the last dimension
            squeeze_len = x.shape[1] // self.m_mode_1_para
            # trim length first
            trim_len = squeeze_len * self.m_mode_1_para
            x_tmp = x[:, 0:trim_len, :]
            
            # (batch, time//squeeze_size, squeeze_size, dim)
            x_tmp = x_tmp.view(x_tmp.shape[0], squeeze_len, 
                               self.m_mode_1_para, -1)
            
            # (batch, time//squeeze_size, dim, squeeze_size)
            x_tmp = x_tmp.permute(0, 1, 3, 2).contiguous()
            
            # (batch, time//squeeze_size, dim * squeeze_size)
            return x_tmp.view(x_tmp.shape[0], squeeze_len, -1)
        else:
            print("SqueezeForWaveGlow not implemented")
        return x_squeezed

    def reverse(self, x_squeezed):
        if self.m_mode == 1:
            # (batch, time//squeeze_size, dim * squeeze_size)
            batch, squeeze_len, squeeze_dim = x_squeezed.shape
            
            # (batch, time//squeeze_size, dim, squeeze_size)
            x_tmp = x_squeezed.view(
                batch, squeeze_len, squeeze_dim // self.m_mode_1_para, 
                self.m_mode_1_para)
            
            # (batch, time//squeeze_size, squeeze_size, dim)
            x_tmp = x_tmp.permute(0, 1, 3, 2).contiguous()
            
            # (batch, time, dim)
            x = x_tmp.view(batch, squeeze_len * self.m_mode_1_para, -1)
        else:
            print("SqueezeForWaveGlow not implemented")
        return x

    
class AffineCouplingWaveGlow_legacy(torch_nn.Module):
    """AffineCouplingWaveGlow_legacy
    
    AffineCoupling block in WaveGlow
    
    Example:
        m_tmp = AffineCouplingWaveGlow_legacy(10, 10, 8, 512, 3, True, True)
        data1 = torch.randn([2, 100, 10])
        cond = torch.randn([2, 100, 10])
        output, log_det = m_tmp(data1, cond)
        data1_re = m_tmp.reverse(output, cond)
        torch.std(data1 - data1_re)
    """
    def __init__(self, in_dim, cond_dim,  
                 wn_num_conv1d, wn_dim_channel, wn_kernel_size, 
                 flag_affine=True, flag_detjac=False):
        """AffineCouplingWaveGlow_legacy(in_dim, cond_dim,  
            wn_num_conv1d, wn_dim_channel, wn_kernel_size, 
            flag_affine=True, flag_detjac=False)
        
        Args:
        -----
          in_dim: int, dim of input audio data (batch, length, in_dim)
          cond_dim, int, dim of condition feature (batch, length, cond_dim)
          wn_num_conv1d: int, number of dilated conv WaveNet blocks
          wn_dim_channel: int, dime of the WaveNet residual & skip channels
          wn_kernel_size: int, kernel size of the dilated convolution layers
          flag_affine: bool, whether use affine or additive transformation?
                       default True
          flag_detjac: bool, whether return the determinant of Jacobian,
                       default False
        
        y -> split() -> y1, y2 -> concate([y1, (y2+bias) * scale])
        When flag_affine == True, y1 -> H() -> scale, bias
        When flag_affine == False, y1 -> H() -> bias, scale=1 
        Here, H() is WaveNet blocks (dilated conv + gated activation)
        """
        super(AffineCouplingWaveGlow_legacy, self).__init__()
        
        self.flag_affine = flag_affine
        self.flag_detjac = flag_detjac
        
        if in_dim % 2 > 0:
            print("AffineCoulingGlow(feat_dim), feat_dim is an odd number?!")
            sys.exit(1)
        
        if self.flag_affine:
            # scale and bias
            self.m_nn_outdim = in_dim // 2 * 2
        else:
            # only bias
            self.m_nn_outdim = in_dim // 2
        
        # pre-transform, change input audio dimension
        #  only half of the features will be used to produce scale and bias
        tmp_l = torch_nn.Linear(in_dim // 2, wn_dim_channel)
        #  weight normalization
        self.m_wn_pre = torch_nn.utils.weight_norm(tmp_l, name='weight')
        
        # WaveNet blocks (dilated conv, gated activation functions)
        tmp_wn = []
        for i in range(wn_num_conv1d):
            dilation = 2 ** i
            tmp_wn.append(nii_wavenet.WaveNetBlock_v2(
                wn_dim_channel, wn_dim_channel, wn_dim_channel, cond_dim,
                dilation, cnn_kernel_size=wn_kernel_size, causal=False))
        self.m_wn = torch_nn.ModuleList(tmp_wn)
        
        # post-transform, change dim from WN channel to audio feature
        tmp_l = torch_nn.Linear(wn_dim_channel, self.m_nn_outdim)
        # For better initialization, bias=0, scale=1 for first mini-batch
        tmp_l.weight.data.zero_()
        tmp_l.bias.data.zero_()
        self.m_wn_post = tmp_l
        
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
        # pre-transformation (batch, length, in_dim//2) 
        # -> (batch, length, WN_channel)
        y1_trans = self.m_wn_pre(y1)
        
        # WaveNet blocks
        wn_output = 0
        res_ch = y1_trans
        for wn_layer in self.m_wn:
            res_ch, ski_ch = wn_layer(res_ch, cond)
            wn_output = wn_output + ski_ch / len(self.m_wn)
        #wn_output = wn_output + res_ch / len(self.m_wn)

        # post-transformation
        y1_tmp = self.m_wn_post(wn_output)
        
        if self.flag_affine:
            log_scale, bias = y1_tmp.chunk(2, -1)
            scale = torch.exp(log_scale)
        else:
            bias = y1_tmp
            scale = torch.ones_like(y1)
            log_scale = torch.zeros_like(y1)
        return scale, bias, log_scale
        
    def forward(self, y, cond, factor=1):
        """AffineCouplingWaveGlow_legacy.forward(y, cond)
        
        input
        -----
          y: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, lengh, cond_dim)
          
        output
        ------
          x: tensor, input feature, (batch, lengh, input_dim)
          detjac: tensor, det of jacobian, (batch,)
        
        y1, y2 = split(y)
        scale, bias = WN(y1)
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
        """AffineCouplingWaveGlow_legacy.reverse(y, cond)
        
        input
        -----
          x: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, lengh, cond_dim)
          
        output
        ------
          y: tensor, input feature, (batch, lengh, input_dim)
        
        x1, x2 = split(x)
        scale, bias = WN(x1)
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


class WaveNetModuleForNonAR(torch_nn.Module):
    """WaveNetModuleWaveGlow
    Casecade of multiple WaveNet blocks:
    x -> ExpandDim -> conv1 -> gated -> res -> conv1 -> gated -> res  ...
                                ^        |
                                |        v
                               cond     skip
    output = sum(skip_channels)
    """
    def __init__(self, input_dim, cond_dim, out_dim, n_blocks, 
                 gate_dim, res_ch, skip_ch, kernel_size=3):
        super(WaveNetModuleForNonAR, self).__init__()
        
        self.m_block_num = n_blocks
        self.m_res_ch_dim = res_ch
        self.m_skip_ch_dim = skip_ch
        self.m_gate_dim = gate_dim
        self.m_kernel_size = kernel_size
        self.m_n_blocks = n_blocks
        if self.m_gate_dim % 2 != 0:
            self.m_gate_dim = self.m_gate_dim // 2 * 2
        
        # input dimension expanding
        tmp = torch_nn.Conv1d(input_dim, res_ch, 1)
        self.l_expand = torch_nn.utils.weight_norm(tmp, name='weight')
        
        # end dimension compressing
        tmp = torch_nn.Conv1d(skip_ch, out_dim, 1)
        tmp.weight.data.zero_()
        tmp.bias.data.zero_()
        self.l_compress = tmp
        
        # dilated convolution and residual-skip-channel transformation
        self.l_conv1 = []
        self.l_resskip = []
        for idx in range(n_blocks):
            dilation = 2 ** idx
            padding = int((kernel_size * dilation - dilation)/2)
            conv1 = torch_nn.Conv1d(
                res_ch, gate_dim, self.m_kernel_size, 
                dilation = dilation, padding=padding)
            conv1 = torch_nn.utils.weight_norm(conv1, name='weight')
            self.l_conv1.append(conv1)
            
            if idx < n_blocks - 1:
                outdim = self.m_res_ch_dim + self.m_skip_ch_dim
            else:
                outdim = self.m_skip_ch_dim
            resskip = torch_nn.Conv1d(self.m_gate_dim//2, outdim, 1)
            resskip = torch_nn.utils.weight_norm(resskip, name='weight')
            self.l_resskip.append(resskip)    
        self.l_conv1 = torch_nn.ModuleList(self.l_conv1)
        self.l_resskip = torch_nn.ModuleList(self.l_resskip)
        
        # a single conditional feature transformation layer
        cond_layer = torch_nn.Conv1d(cond_dim, gate_dim * n_blocks, 1)
        cond_layer = torch_nn.utils.weight_norm(cond_layer, name='weight')
        self.l_cond = cond_layer
        return
    
    def forward(self, x, cond):
        """
        """
        
        # input feature expansion
        # change the format to (batch, dimension, length)
        x_expanded = self.l_expand(x.permute(0, 2, 1))
        # condition feature transformation
        cond_proc = self.l_cond(cond.permute(0, 2, 1))

        # skip-channel accumulation
        skip_ch_out = 0
        
        conv_input = x_expanded
        for idx, (l_conv1, l_resskip) in \
            enumerate(zip(self.l_conv1, self.l_resskip)):
            
            tmp_dim = idx * self.m_gate_dim
            # condition feature of this layer
            cond_tmp = cond_proc[:, tmp_dim : tmp_dim + self.m_gate_dim, :]
            # conv transformed
            conv_tmp = l_conv1(conv_input)
            
            # gated activation
            gated_tmp = cond_tmp + conv_tmp
            t_part = torch.tanh(gated_tmp[:, :self.m_gate_dim//2, :])
            s_part = torch.sigmoid(gated_tmp[:, self.m_gate_dim//2:, :])
            gated_tmp = t_part * s_part
            
            # transformation into skip / residual channels
            resskip_tmp = l_resskip(gated_tmp)
            
            # reschannel 
            if idx == self.m_n_blocks - 1:
                skip_ch_out = skip_ch_out + resskip_tmp
            else:
                conv_input = conv_input + resskip_tmp[:, 0:self.m_res_ch_dim, :]
                skip_ch_out = skip_ch_out + resskip_tmp[:, self.m_res_ch_dim:,:]
        output = self.l_compress(skip_ch_out)
        
        # permute back to (batch, length, dimension)
        return output.permute(0, 2, 1)
    
    
    
class AffineCouplingWaveGlow(torch_nn.Module):
    """AffineCouplingWaveGlow
    
    AffineCoupling block in WaveGlow
    
    Example:
        m_tmp = AffineCouplingWaveGlow(10, 10, 8, 512, 3, True, True)
        data1 = torch.randn([2, 100, 10])
        cond = torch.randn([2, 100, 10])
        output, log_det = m_tmp(data1, cond)
        data1_re = m_tmp.reverse(output, cond)
        torch.std(data1 - data1_re)
    """
    def __init__(self, in_dim, cond_dim,  
                 wn_num_conv1d, wn_dim_channel, wn_kernel_size, 
                 flag_affine=True, flag_detjac=False):
        """AffineCouplingWaveGlow(in_dim, cond_dim,  
            wn_num_conv1d, wn_dim_channel, wn_kernel_size, 
            flag_affine=True, flag_detjac=False)
        
        Args:
        -----
          in_dim: int, dim of input audio data (batch, length, in_dim)
          cond_dim, int, dim of condition feature (batch, length, cond_dim)
          wn_num_conv1d: int, number of dilated conv WaveNet blocks
          wn_dim_channel: int, dime of the WaveNet residual & skip channels
          wn_kernel_size: int, kernel size of the dilated convolution layers
          flag_affine: bool, whether use affine or additive transformation?
                       default True
          flag_detjac: bool, whether return the determinant of Jacobian,
                       default False
        
        y -> split() -> y1, y2 -> concate([y1, (y2+bias) * scale])
        When flag_affine == True, y1 -> H() -> scale, bias
        When flag_affine == False, y1 -> H() -> bias, scale=1 
        Here, H() is WaveNet blocks (dilated conv + gated activation)
        """
        super(AffineCouplingWaveGlow, self).__init__()
        
        self.flag_affine = flag_affine
        self.flag_detjac = flag_detjac
        
        if in_dim % 2 > 0:
            print("AffineCoulingGlow(feat_dim), feat_dim is an odd number?!")
            sys.exit(1)
        
        if self.flag_affine:
            # scale and bias
            self.m_nn_outdim = in_dim // 2 * 2
        else:
            # only bias
            self.m_nn_outdim = in_dim // 2
        
        # WaveNet blocks (dilated conv, gated activation functions)
        self.m_wn = WaveNetModuleForNonAR(
            in_dim // 2, cond_dim, self.m_nn_outdim, wn_num_conv1d,
            wn_dim_channel * 2, wn_dim_channel, wn_dim_channel, 
            wn_kernel_size 
        )
        
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
        y1_tmp = self.m_wn(y1, cond)
        
        if self.flag_affine:
            log_scale, bias = y1_tmp.chunk(2, -1)
            scale = torch.exp(log_scale)
        else:
            bias = y1_tmp
            scale = torch.ones_like(y1)
            log_scale = torch.zeros_like(y1)
        return scale, bias, log_scale
        
    def forward(self, y, cond, factor=1):
        """AffineCouplingWaveGlow.forward(y, cond)
        
        input
        -----
          y: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, lengh, cond_dim)
          
        output
        ------
          x: tensor, input feature, (batch, lengh, input_dim)
          detjac: tensor, det of jacobian, (batch,)
        
        y1, y2 = split(y)
        scale, bias = WN(y1)
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
        """AffineCouplingWaveGlow.reverse(y, cond)
        
        input
        -----
          x: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, lengh, cond_dim)
          
        output
        ------
          y: tensor, input feature, (batch, lengh, input_dim)
        
        x1, x2 = split(x)
        scale, bias = WN(x1)
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
        
        

class FlowStepWaveGlow(torch_nn.Module):
    """FlowStepWaveGlow
    One flow step for waveglow
    y -> intertical_1x1() -> AffineCoupling -> x
    
    Example
        m_tmp = FlowStepWaveGlow(10, 10, 8, 512, 3, flag_affine=True)
        output, log_det = m_tmp(data1, cond)
        data1_re = m_tmp.reverse(output, cond)

        torch.std(data1 - data1_re)
    """
    def __init__(self, in_dim, cond_dim, 
                 wn_num_conv1d, wn_dim_channel, wn_kernel_size, flag_affine,
                 flag_affine_block_legacy=False):
        """FlowStepWaveGlow(in_dim, cond_dim, 
            wn_num_conv1d, wn_dim_channel, wn_kernel_size, flag_affine,
            flag_affine_block_legacy=False)
        
        Args
        ----
          in_dim: int, input feature dim, (batch, length, in_dim)
          cond_dim:, int, conditional feature dim, (batch, length, cond_dim)
          wn_num_conv1d: int, number of 1Dconv WaveNet block in this flow step
          wn_dim_channel: int, dim of the WaveNet residual and skip channels
          wn_kernel_size: int, kernel size of the dilated convolution layers
          flag_affine: bool, whether use affine or additive transformation?
                       default True
          flag_affine_block_legacy, bool, whether use AffineCouplingWaveGlow or 
                       AffineCouplingWaveGlow_legacy.

        For wn_dim_channel and wn_kernel_size, see AffineCouplingWaveGlow
        For flag_affine == False, scale will be 1.0
        """
        super(FlowStepWaveGlow, self).__init__()
        
        # Invertible transformation layer
        #self.m_invtrans = nii_glow.InvertibleTrans(in_dim, flag_detjac=True)
        self.m_invtrans = Invertible1x1ConvWaveGlow(in_dim, flag_detjac=True)
        
        # Coupling layer
        if flag_affine_block_legacy:
            self.m_coupling = AffineCouplingWaveGlow_legacy(
                in_dim, cond_dim, wn_num_conv1d, wn_dim_channel, wn_kernel_size,
                flag_affine, flag_detjac=True)
        else:
            self.m_coupling = AffineCouplingWaveGlow(
                in_dim, cond_dim, wn_num_conv1d, wn_dim_channel, wn_kernel_size,
                flag_affine, flag_detjac=True)
        return
    
    def forward(self, y, cond, factor=1):
        """FlowStepWaveGlow.forward(y, cond, factor=1)
        
        input
        -----
          y: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, lengh, cond_dim)
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
        # coupling
        x_tmp, log_det_2 = self.m_coupling(x_tmp, cond, factor) 
        return x_tmp, log_det_1 + log_det_2
    
    def reverse(self, x, cond):
        """FlowStepWaveGlow.reverse(y, cond)
        
        input
        -----
          x: tensor, input feature, (batch, lengh, input_dim)
          cond: tensor, condition feature , (batch, lengh, cond_dim)
          
        output
        ------
          y: tensor, input feature, (batch, lengh, input_dim)
        """
        y_tmp = self.m_coupling.reverse(x, cond) 
        y_tmp = self.m_invtrans.reverse(y_tmp)
        return y_tmp


    
class WaveGlowBlock(torch_nn.Module):
    """WaveGlowBlock
    A WaveGlowBlock includes multiple steps of flow.
    
    The Nvidia WaveGlow does not define WaveGlowBlock but directly
    defines 12 flow steps. However, after every 4 flow steps, two
    dimension of z will be extracted (multi-scale approach).
    It is not convenient to decide when to extract z.
    
    Here, we define a WaveGlowBlock as the casecade of multiple flow
    steps, and this WaveGlowBlock can extract the two dimensions from
    the output of final flow step. 
    
    Example:
        data1 = torch.randn([2, 10, 10])
        cond = torch.randn([2, 10, 16])
        m_block = WaveGlowBlock(10, 16, 5, 8, 512, 3)
        x, z, log_det = m_block(data1, cond)
        data_re = m_block.reverse(x, z, cond)
        print(torch.std(data_re - data1))
    """
    def __init__(self, in_dim, cond_dim, n_flow_steps,
                 wn_num_conv1d, wn_dim_channel, wn_kernel_size,
                 flag_affine=True, 
                 flag_split = False, 
                 flag_final_block=False,
                 split_dim = 2, 
                 flag_affine_block_legacy=False):
        """WaveGlowBlock(in_dim, cond_dim, n_flow_steps,
                 wn_num_conv1d, wn_dim_channel, wn_kernel_size,
                 flag_affine=True, flag_split = False, split_dim = 2,
                 flag_affine_block_legacy=False)
        Args
        ----
          in_dim: int, input feature dim, (batch, length, in_dim)
          cond_dim:, int, conditional feature dim, (batch, length, cond_dim)
          n_flow_steps: int, number of flow steps in one block
          wn_num_conv1d: int, number of dilated conv WaveNet blocks
          wn_dim_channel: int, dim of the WaveNet residual and skip channels
          wn_kernel_size: int, kernel size of the dilated convolution layers
          flag_affine: bool, whether use affine or additive transformation?
                       default True
          flag_split: bool, whether split output z for multi-scale structure
                       default True
          flag_final_block: bool, whether this block is the final block
                       default False
          split_dim: int, if flag_split==True, z[:, :, :split_dim] will be
                     extracted, z[:, :, split_dim:] can be used for the next
                     WaveGlowBlock
          flag_affine_block_legacy, bool, whether use the legacy implementation
                       of wavenet-based affine transformaiton layer
                       default False. 
         
        For wn_dim_channel and wn_kernel_size, see AffineCouplingWaveGlow
        For flag_affine, see AffineCouplingWaveGlow
        """
        super(WaveGlowBlock, self).__init__()
        
        tmp_flows = []
        for i in range(n_flow_steps):
            tmp_flows.append(
                FlowStepWaveGlow(
                    in_dim, cond_dim,
                    wn_num_conv1d, wn_dim_channel, wn_kernel_size,
                    flag_affine, flag_affine_block_legacy))
        self.m_flows = torch_nn.ModuleList(tmp_flows)

        self.flag_split = flag_split
        self.flag_final_block = flag_final_block
        self.split_dim = split_dim
        
        if self.flag_split and self.flag_final_block:
            print("WaveGlowBlock: flag_split and flag_final_block are True")
            print("This is unexpected. Please check model definition")
            sys.exit(1)
        if self.flag_split and self.split_dim <= 0:
            print("WaveGlowBlock: split_dim should be > 0")
            sys.exit(1)
            
        return
    
    def forward(self, y, cond, factor=1):
        """x, z, log_detjac = WaveGlowBlock(y) 
        
        y -> H() -> [z, x], log_det_jacobian
        H() consists of multiple flow steps (1x1conv + AffineCoupling)
        
        input
        -----
          y: tensor, (batch, length, dim)
          cond, tensor, (batch, length, cond_dim)
          factor, None or int, this is used to divde likelihood, default 1

        output
        ------
         log_detjac: tensor or scalar
         
         if self.flag_split:
           x: tensor, (batch, length, in_dim - split_dim), 
           z: tensor, (batch, length, split_dim), 
         else:
           if self.flag_final_block:
              x: None, no input to the next block
              z: tensor, (batch, length, dim), for N(z; 0, I)
           else:
              x: tensor, (batch, length, dim), 
              z: None, no latent for N(z; 0, I) from this block
        concate([x,z]) should have the same size as y
        """
        # flows
        log_detjac = 0

        x_tmp = y
        for l_flow in self.m_flows:
            x_tmp, log_detjac_tmp = l_flow(x_tmp, cond, factor)
            log_detjac = log_detjac + log_detjac_tmp
            
        if self.flag_split:
            z = x_tmp[:, :, :self.split_dim]
            x = x_tmp[:, :, self.split_dim:]
        else:
            if self.flag_final_block:
                z = x_tmp
                x = None
            else:
                z = None
                x = x_tmp
        return x, z, log_detjac
    
    def reverse(self, x, z, cond):
        """y = WaveGlowBlock.reverse(x, z, cond) 
        
        [z, x] -> H^{-1}() -> y
        
        input
        -----
        if self.flag_split:
          x: tensor, (batch, length, in_dim - split_dim), 
          z: tensor, (batch, length, split_dim), 
        else:
          if self.flag_final_block:
              x: None
              z: tensor, (batch, length, in_dim)
          else:
              x: tensor, (batch, length, in_dim)
              z: None
        output
        ------
          y: tensor, (batch, length, in_dim)          
        """
        if self.flag_split:
            if x is None or z is None:
                print("WaveGlowBlock.reverse: x and z should not be None")
                sys.exit(1)
            y_tmp = torch.cat([z, x], dim=-1)
        else:
            if self.flag_final_block:
                if z is None: 
                    print("WaveGlowBlock.reverse: z should not be None")
                    sys.exit(1)
                y_tmp = z
            else:
                if x is None: 
                    print("WaveGlowBlock.reverse: x should not be None")
                    sys.exit(1)
                y_tmp = x
        
        for l_flow in self.m_flows[::-1]:
            # affine
            y_tmp = l_flow.reverse(y_tmp, cond)
        return y_tmp
        
        
class WaveGlow(torch_nn.Module):
    """WaveGlow
    
    Example
      cond_dim = 4
      upsample = 80
      num_blocks = 4
      num_flows_inblock = 5
      wn_num_conv1d = 8
      wn_dim_channel = 512
      wn_kernel_size = 3

      # waveforms of length 1600
      wave1 = torch.randn([2, 1600, 1])   
      # condition feature
      cond = torch.randn([2, 1600//upsample, cond_dim])

      # model
      m_model = nii_waveglow.WaveGlow(
         cond_dim, upsample, 
         num_blocks, num_flows_inblock, wn_num_conv1d, 
         wn_dim_channel, wn_kernel_size)

      # forward computation, neg_log = -(logp + log_detjac)
      # neg_log.backward() can be used for backward
      z, neg_log, logp, log_detjac = m_model(wave1, cond)    

      # recover the signal
      wave2 = m_model.reverse(z, cond)
      
      # check difference between original wave and recovered wave
      print(torch.std(wave1 - wave2))                        
    """
    def __init__(self, cond_dim, upsample_rate, 
                 num_blocks, num_flows_inblock, 
                 wn_num_conv1d, wn_dim_channel, wn_kernel_size,
                 flag_affine = True,
                 early_hid_dim=2, 
                 flag_affine_block_legacy=False):
        """WaveGlow(cond_dim, upsample_rate,
                 num_blocks, num_flows_inblock, 
                 wn_num_conv1d, wn_dim_channel, wn_kernel_size,
                 flag_affine = True,
                 early_hid_dim=2,
                 flag_affine_block_legacy=False)
        
        Args
        ----
          cond_dim:, int, conditional feature dim, (batch, length, cond_dim)
          upsample_rate: int, up-sampling rate for condition features
          num_blocks: int, number of WaveGlowBlocks
          num_flows_inblock: int, number of flow steps in one WaveGlowBlock
          wn_num_conv1d: int, number of 1Dconv WaveNet block in this flow step
          wn_dim_channel: int, dim of the WaveNet residual and skip channels
          wn_kernel_size: int, kernel size of the dilated convolution layers
          flag_affine: bool, whether use affine or additive transformation?
                       default True
          early_hid_dim: int, dimension for z_1, z_2 ... , default 2
          flag_affine_block_legacy, bool, whether use the legacy implementation
                       of wavenet-based affine transformaiton layer
                       default False. The difference is on the WaveNet part
                       Please configure AffineCouplingWaveGlow and 
                       AffineCouplingWaveGlow_legacy
        

        This model defines:
        
        cond -> upsample/squeeze -> | ------> | --------> | 
                                    v         v           v   
        y -> squeeze   -> WaveGlowBlock -> WGBlock ... WGBlock -> z
                             |-> z_1          |-> z_2 
        
        z_1, z_2, ... are the extracted z from a multi-scale flow structure
        concate([z_1, z_2, z]) is expected to be the white Gaussian noise
        
        If early_hid_dim == 0, z_1 and z_2 will not be extracted
        """
        super(WaveGlow, self).__init__()
        
        # input is assumed to be waveform
        self.m_input_dim = 1
        self.m_early_hid_dim = early_hid_dim
        
        # squeeze layer
        self.m_squeeze = SqueezeForWaveGlow()
        
        # up-sampling layer
        #self.m_upsample = nii_nn.UpSampleLayer(cond_dim, upsample_rate, True)
        self.m_upsample = upsampleByTransConv(cond_dim, upsample_rate)
        
        # wavenet-based flow blocks
        # squeezed input dimension
        squeezed_in_dim = self.m_input_dim * self.m_squeeze.get_squeeze_factor()
        # squeezed condition feature dimension
        squeezed_cond_dim = cond_dim * self.m_squeeze.get_squeeze_factor()
        
        # save the dimension for get_z_noises
        self.m_feat_dim = []
        
        # define blocks
        tmp_squeezed_in_dim = squeezed_in_dim
        tmp_flow_blocks = []
        for i in range(num_blocks):
            # if this is not the last block and early_hid_dim >0
            flag_split = (i < (num_blocks-1)) and early_hid_dim > 0
            flag_final_block = i == (num_blocks-1)
            
            # save the dimension for get_z_noises
            if flag_final_block:
                self.m_feat_dim.append(tmp_squeezed_in_dim)
            else:
                self.m_feat_dim.append(early_hid_dim if flag_split else 0)
            
            tmp_flow_blocks.append(
                WaveGlowBlock(
                    tmp_squeezed_in_dim, squeezed_cond_dim, num_flows_inblock,
                    wn_num_conv1d, wn_dim_channel, wn_kernel_size, flag_affine,
                    flag_split = flag_split, flag_final_block=flag_final_block,
                    split_dim = early_hid_dim, 
                    flag_affine_block_legacy = flag_affine_block_legacy))
            
            # multi-scale approach will extract a few dimensions for next flow
            # thus, input dimension to the next block will be this 
            tmp_squeezed_in_dim = tmp_squeezed_in_dim - early_hid_dim
            
        self.m_flowblocks = torch_nn.ModuleList(tmp_flow_blocks)
        
        # done
        return
    
        
    def _normal_lh(self, noise):
        # likelihood of normal distribution on the given noise
        return -0.5 * np.log(2 * np.pi) - 0.5 * noise ** 2
    
    def forward(self, y, cond):
        """z, neg_logp_y, logp_z, logdet = WaveGlow.forward(y, cond) 
        
        cond -> upsample/squeeze -> | ------> | --------> | 
                                    v         v           v   
        y -> squeeze   -> WaveGlowBlock -> WGBlock ... WGBlock -> z
                             |-> z_1          |-> z_2 
                             
        input
        -----
          y: tensor, (batch, waveform_length, 1)
          cond: tensor,  (batch, cond_length, 1)
          
        output
        ------
          z: list of tensors, [z_1, z_2, ... ,z ] in figure above
          neg_logp_y: scalar, - log p(y)
          logp_z: scalar, -log N(z), summed over one data sequence, but averaged
                  over batch.
          logdet: scalar, -|det dH(.)/dy|, summed over one data sequence, 
                  but averaged
                  over batch.
        
        If self.early_hid_dim == 0, z_1, z_2 ... will be None
        """
        
        # Rather than summing the likelihood and divide it by the number of 
        #  data in the final step, we divide this factor from the likelihood
        #  caculating by each flow step and sum the scaled likelihood. 
        # Two methods are equivalent, but the latter may prevent numerical 
        #  overflow of the likelihood value for long sentences
        factor = np.prod([dim for dim in y.shape])
        
        # waveform squeeze (batch, squeezed_length, squeezed_dim)
        y_squeezed = self.m_squeeze(y)
        squeezed_dim = y_squeezed.shape[-1]
        
        # condition feature upsampling and squeeze
        #  (batch, squeezed_length, squeezed_dim_cond)
        cond_up_squeezed = self.m_squeeze(self.m_upsample(cond))
        
        # flows
        z_bags = []
        log_detjac = 0
        log_pz = 0

        x_tmp = y_squeezed
        for m_block in self.m_flowblocks:
            x_tmp, z_tmp, log_detjac_tmp = m_block(
                x_tmp, cond_up_squeezed, factor)
            
            # accumulate log det jacobian
            log_detjac += log_detjac_tmp
            
            # compute N(z; 0, I)
            # save z_tmp (even if it is None)
            z_bags.append(z_tmp)
            # accumulate log_N(z; 0, I) only if it is valid
            if z_tmp is not None:
                log_pz += nii_glow.sum_over_keep_batch2(
                    self._normal_lh(z_tmp), factor)
        
        # average over batch and data points
        neg_logp_y = -(log_pz + log_detjac).sum()
        return z_bags, neg_logp_y, \
            log_pz.sum(), log_detjac.sum()
        
    def reverse(self, z_bags, cond):
        """y = WaveGlow.reverse(z_bags, cond) 
        
        cond -> upsample/squeeze -> | ------> | --------> | 
                                    v         v           v   
        y <- unsqueeze  <- WaveGlowBlock -> WGBlock ... WGBlock <- z
                             |<- z_1          |<- z_2 
                             
        input
        -----
          z: list of tensors, [z_1, z_2, ... ,z ] in figure above
          cond: tensor,  (batch, cond_length, 1)
          
        output
        ------
          y: tensor, (batch, waveform_length, 1)
        
        If self.early_hid_dim == 0, z_1, z_2 ... should be None
        """
        # condition feature upsampling and squeeze
        #  (batch, squeezed_length, squeezed_dim_cond)
        cond_up_sqe = self.m_squeeze(self.m_upsample(cond))
        
        # initial
        y_tmp = None
        for z, m_block in zip(z_bags[::-1], self.m_flowblocks[::-1]):
            y_tmp = m_block.reverse(y_tmp, z, cond_up_sqe)
        y = self.m_squeeze.reverse(y_tmp)
        return y
    
    def get_z_noises(self, length, noise_std=0.7, batchsize=1):
        """z_bags = WaveGlow.get_z_noises(length, noise_std=0.7, batchsize=1)
        Return a list of random noises for random sampling
        
        input
        -----
          length: int, length of target waveform (without squeeze)
          noise_std: float, std of Gaussian noise, default 0.7
          batchsize: int, batch size of this random data, default 1
        
        output
        ------
          z_bags: list of tensors
        
        Shape of tensor in z_bags is decided by WaveGlow configuration.
        WaveGlow.reverse(z_bags, cond) can be used to generate waveform
        """
        squeeze_length = self.m_squeeze.get_expected_squeeze_length(length)
        
        device = next(self.parameters()).device
        z_bags = []
        
        # generate the z for each WaveGlowBlock
        for feat_dim in self.m_feat_dim:
            if feat_dim is not None and feat_dim > 0:
                z_tmp = torch.randn(
                    [batchsize, squeeze_length, feat_dim], 
                    dtype=nii_io_conf.d_dtype, 
                    device=device)
                z_bags.append(z_tmp * noise_std)
            else:
                z_bags.append(None)
        return z_bags

if __name__ == "__main__":
    print("Definition of WaveGlow")
