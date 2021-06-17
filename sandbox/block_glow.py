#!/usr/bin/env python
"""
Building blocks for glow
"""
from __future__ import absolute_import

import os
import sys
import time
import numpy as np
import scipy.linalg

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
import torch.nn.init as torch_init

import sandbox.block_nn as nii_nn
import core_scripts.data_io.conf as nii_io_conf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

def sum_over_keep_batch(data):
    # (batch, dim1, dim2, ..., ) -> (batch)
    # sum over dim1, dim2, ...
    sum_dims = [x for x in range(data.ndim)][1:]
    return torch.sum(data, dim=sum_dims)

def sum_over_keep_batch2(data, factor):
    # (batch, dim1, dim2, ..., ) -> (batch)
    # device each value by factor and 
    # sum over dim1, dim2, ...
    sum_dims = [x for x in range(data.ndim)][1:]
    return torch.sum(data / factor, dim=sum_dims)


class ActNorm(torch_nn.Module):
    """Activation Normalization
    
    Activation normalization layer used in 
     Kingma, D. P. & Dhariwal, P. Glow
     Generative Flow with Invertible 1x1 Convolutions. 
     arXiv Prepr. arXiv1807.03039 (2018) 
     
     
    For debug:
        m_actnorm = ActNorm(5, flag_detjac=True)
        data = torch.rand([2, 5, 5])
        out, detjac = m_actnorm(data)
        data_new = m_actnorm.reverse(out)

        print(detjac)
        #print(data.mean(dim=[0, 1]))
        #print(data.std(dim=[0, 1]))
        #print(m_actnorm.m_bias)
        #print(m_actnorm.m_scale)
        print(torch.sum(torch.log(torch.abs(m_actnorm.m_scale))) * 5 * 2)
        print(data - data_new)
    """
    def __init__(self, feat_dim, flag_detjac=False):
        """ActNorm(feat_dim, flag_detjac)
        
        Args
        ----
          feat_dim: int,   feature dimension (channel for image),
                    input tensor (batch, ..., feature dimension)
          flag_detjac: bool, whether output determinant of jacobian
        
        Note that, it assumes y -> H(.) -> x, where H(.) is ActNorm.forward,
        it then returns |det(dH(y)/dy)|
        """
        super(ActNorm, self).__init__()
        
        # flag
        # whether return det of jacobian matrix
        self.flag_detjac = flag_detjac
        
        # 
        self.feat_dim = feat_dim
        
        # parameter
        self.m_scale = torch_nn.Parameter(torch.ones(feat_dim), 
                                          requires_grad=True)
        self.m_bias = torch_nn.Parameter(torch.zeros(feat_dim), 
                                          requires_grad=True)
        # flag to prevent re-initialization of the scale and bias
        self.m_init_flag = torch_nn.Parameter(torch.zeros(1), 
                                              requires_grad=False)
        return
    
    def _log(self, x):
        # add a floor
        #return torch.log(x + torch.finfo(x.dtype).eps)
        return torch.log(x)
    
    def _detjac(self, factor=1):
        """
        """
        # \sum log |s|, this same value is used for all data 
        # in this mini-batch, no need to duplicate to (batch,)
        return torch.sum(self._log(torch.abs(self.m_scale)) / factor)
    
    def _detjac_size_factor(self, y):
        """ h * w * detjac
        we need to compute h * w
        """
        with torch.no_grad():
            # tensor in shape (batch, d1, d2, ... feat_dim)
            # then the factor will be d1 x d2 ...
            data_size = torch.tensor(y.shape[1:-1])
            data_factor = torch.prod(data_size)
        return data_factor
        
        
    def _init_scale_m(self, y):
        """ initialize scale and bias for transformation
        """
        with torch.no_grad():
            # (batch, ... ,feat_dim) -> (-1, feat_dim)
            tmp_y = y.view(-1, self.feat_dim)
            # get mean and std per feat_dim
            m = torch.mean(tmp_y, dim=0)
            std = torch.std(tmp_y, dim=0) + 1e-6
            
            # because the transform is (y + bias) * scale
            # save scale = 1/std and bias = -m
            self.m_scale.data = 1 / std 
            self.m_bias.data = -1 * m
            
            # prevent further initialization 
            self.m_init_flag += 1
        return
    
    def forward(self, y, factor=1):
        """x = ActNorm.forward(y)
        
        input
        -----
          y: tensor, (batch, dim1, ..., feat_dim)

        output
        ------
          x: tensor, (batch, dim1, ..., feat_dim)
        
        if self.flag_detjac, also returns log_detjac (scalar)
        """
        # do initialization for the 1st time
        if self.m_init_flag.item() < 1:
            self._init_scale_m(y)
            
        # in initial stage, this is equivalent to (y - m)/std
        x = (y + self.m_bias) * self.m_scale
        
        if self.flag_detjac:
            log_detjac = self._detjac(factor) * self._detjac_size_factor(y)
            return x, log_detjac
        else:
            return x
            
        
    def reverse(self, x):
        """y = ActNorm.reverse(x)
        
        input
        -----
          x: tensor, (batch, dim1, ..., feat_dim)

        output
        ------
          y: tensor, (batch, dim1, ..., feat_dim)
        """
        return x / self.m_scale - self.m_bias
    
    
class InvertibleTrans(torch_nn.Module):
    """InvertibleTransformation
    
    Invertible transformation layer used in 
     Kingma, D. P. & Dhariwal, P. Glow
     Generative Flow with Invertible 1x1 Convolutions. 
     arXiv Prepr. arXiv1807.03039 (2018) 
    
    1x1 convolution is implemented using torch.matmul
    
    Example:
        feat_dim = 5
        m_trans = InvertibleTrans(feat_dim, flag_detjac=True)
        data = torch.rand([2, feat_dim, feat_dim])
        out, detjac = m_trans(data)
        data_new = m_trans.reverse(out)

        print(data_new - data)
        print(detjac)
    """
    def __init__(self, feat_dim, flag_detjac=False):
        """InvertibleTrans(feat_dim, flag_detjac)
        
        Args
        ----
          feat_dim: int,   feature dimension (channel for image),
                    input tensor (batch, ..., feature dimension)
          flag_detjac: bool, whether output determinant of jacobian
        
        It assumes y -> H(.) -> x, where H(.) is InvertibleTrans.forward,
        it then returns |det(dH(y)/dy)|
        """
        super(InvertibleTrans, self).__init__()

        # 
        self.feat_dim = feat_dim

        # create initial permutation, lower, and upper triangle matrices
        seed_mat = np.random.randn(feat_dim, feat_dim)
        # qr decomposition, rotation_mat is a unitary matrix
        rotation_mat, _ = scipy.linalg.qr(seed_mat)
        # LU decomposition
        permute_mat, lower_mat, upper_mat = scipy.linalg.lu(rotation_mat)
        
        # mask matrix (with zero on the diagonal line)
        u_mask = np.triu(np.ones_like(seed_mat), k=1)
        d_mask = u_mask.T
                
        # permuate matrix, fixed
        self.m_permute_mat = torch_nn.Parameter(
            torch.tensor(permute_mat.copy(), dtype=nii_io_conf.d_dtype), 
            requires_grad=False)
        # Lower triangle matrix, trainable
        self.m_lower_tria = torch_nn.Parameter(
            torch.tensor(lower_mat.copy(), dtype=nii_io_conf.d_dtype), 
            requires_grad=True)
        # Uppper triangle matrix, trainable
        self.m_upper_tria = torch_nn.Parameter(
            torch.tensor(upper_mat.copy(), dtype=nii_io_conf.d_dtype), 
            requires_grad=True)

        # diagonal line
        tmp_diag_line = torch.tensor(
            upper_mat.diagonal().copy(),dtype=nii_io_conf.d_dtype)
        # use log(|s|)
        self.m_log_abs_diag = torch_nn.Parameter(
            torch.log(torch.abs(tmp_diag_line)), requires_grad=True)
        # save the sign of s as fixed parameter
        self.m_diag_sign = torch_nn.Parameter(
            torch.sign(tmp_diag_line), requires_grad=False) 

        # mask and all-1 diangonal line
        self.m_l_mask = torch_nn.Parameter(
            torch.tensor(d_mask.copy(), dtype=nii_io_conf.d_dtype), 
            requires_grad=False)
        self.m_u_mask = torch_nn.Parameter(
            torch.tensor(u_mask.copy(), dtype=nii_io_conf.d_dtype), 
            requires_grad=False)
        self.m_eye = torch_nn.Parameter(
            torch.eye(self.feat_dim, dtype=nii_io_conf.d_dtype),
            requires_grad=False)
        
        # buffer for inverse matrix
        self.flag_invered = False
        self.m_inver = torch_nn.Parameter(
            torch.tensor(permute_mat.copy(), dtype=nii_io_conf.d_dtype), 
            requires_grad=False)
        
        # 
        self.flag_detjac = flag_detjac
        return
    

    def _inverse(self):
        """ inverse of the transformation matrix
        """
        return torch.inverse(self._compose_mat())
    
    def _compose_mat(self):
        """ compose the transformation matrix
        W = P L (U + sign * exp( log|s|))
        """
        # U + sign * exp(log|s|)
        tmp_u = torch.diag(self.m_diag_sign * torch.exp(self.m_log_abs_diag))
        tmp_u = tmp_u + self.m_upper_tria * self.m_u_mask
        # L
        tmp_l = self.m_lower_tria * self.m_l_mask + self.m_eye
        return torch.matmul(self.m_permute_mat, torch.matmul(tmp_l, tmp_u))
    
    def _log(self, x):
        # add a floor
        #return torch.log(x + torch.finfo(x.dtype).eps)
        return torch.log(x)
    
    def _detjac(self, factor=1):
        """
        """
        # \sum log|s|
        # no need to duplicate to each data in the batch
        # they all use the same detjac
        return torch.sum(self.m_log_abs_diag / factor)
    
    def _detjac_size_factor(self, y):
        with torch.no_grad():
            # tensor in shape (batch, d1, d2, ... feat_dim)
            # then the factor will be d1 x d2 ...
            data_size = torch.tensor(y.shape[1:-1])
            data_factor = torch.prod(data_size)
        return data_factor
    
    def forward(self, y, factor=1):
        # y W
        # for other implementation, this is done with conv2d 1x1 convolution
        # to be consistent, we can use .T to transpose the matrix first
        if self.flag_detjac:
            detjac = self._detjac(factor) * self._detjac_size_factor(y)
            return torch.matmul(y, self._compose_mat()), detjac
        else:
            return torch.matmul(y, self._compose_mat()), 
    
    def reverse(self, x):
        if self.training:
            # if it is for training, compute inverse everytime
            self.m_inver.data = self._inverse().clone()
        else:
            # during inference, only do this once
            if self.flag_invered is False:
                self.m_inver.data = self._inverse().clone()
                # only compute inverse matrix once
                self.flag_invered = True
        return torch.matmul(x, self.m_inver)

class ZeroInitConv2dForGlow(torch_nn.Module):
    """ZeroIniConv2dForGlow
    
    Last Conv2d layer of Glow uses zero-initialized conv2d
    This is only used for images
    """
    def __init__(self, in_feat_dim, out_feat_dim, kernel_size=3, padding=1):
        super().__init__()
        # conv
        self.m_conv = torch_nn.Conv2d(in_feat_dim, out_feat_dim, 
                                      kernel_size, padding=0)
        self.m_conv.weight.data.zero_()
        self.m_conv.bias.data.zero_()
        # scale parameter, following https://github.com/rosinality/glow-pytorch/
        self.m_scale = torch_nn.Parameter(
            torch.zeros(out_feat_dim, dtype=nii_io_conf.d_dtype))
        #
        self.m_pad_size = padding
        return
    
    def _zerobias(self):
        self.m_conv.bias.data.zero_()
        return
    
    def _normal_weight(self):
        self.m_conv.weight.data.normal_(0, 0.05)
        return
    
    
    def forward(self, x):
        p = self.m_pad_size
        # pad
        y = torch_nn_func.pad(x.permute(0, 3, 1, 2), [p,p,p,p], value=1)
        # conv
        y = self.m_conv(y).permute(0, 2, 3, 1).contiguous()
        # scale parameter, following https://github.com/rosinality/glow-pytorch/
        return y * torch.exp(self.m_scale * 3)


class Conv2dForGlow(torch_nn.Module):
    """Conv2dForGlow
    
    Other Conv2d layer of Glow uses zero-initialized conv2d
    This is only used for images
    """
    def __init__(self, in_feat_dim, out_feat_dim, kernel_size=3, padding=1):
        super().__init__()
        self.m_conv = torch_nn.Conv2d(in_feat_dim, out_feat_dim, 
                                      kernel_size, padding=padding)
        return
    
    def _zerobias(self):
        self.m_conv.bias.data.zero_()
        return

    def _normal_weight(self):
        self.m_conv.weight.data.normal_(0, 0.05)
        return

    def forward(self, x):
        return self.m_conv(x.permute(0, 3, 1, 2)).permute(0,2,3,1).contiguous()

    
class AffineCouplingGlow(torch_nn.Module):
    """AffineCouplingGlow
    
    AffineCoupling block in Glow
    
    
    Example:
       m_affine = AffineCouplingGlow(10, 32, flag_affine=False,flag_detjac=True)
       data = torch.randn([2, 4, 4, 10])
       data_out, detjac = m_affine(data)

       data_inv = m_affine.reverse(data_out)
       print(data_inv - data)
       print(detjac)
    """
    def __init__(self, feat_dim, conv_out_dim=512, 
                 flag_affine=True, flag_detjac=False):
        """AffineCouplingGlow(feat_dim, conv_out_dim=512, 
        flag_affine=True, flag_detjac=False)
        
        Args:
        -----
          feat_dim: int, dimension of input feature (channel number of image)
                    feat_dim must be an even number
          conv_out_dim: int, dimension of output feature of the intermediate
                        conv layer, default 512
          flag_affine: bool, whether use affine or additive transformation?
                       default True
          flag_detjac: bool, whether return the determinant of Jacobian,
                       default False
        
        It assumes that y -> H(.) -> x, where H(.) is AffineCouplingGlow.forward
        When flag_affine == True,  H(y) = concante([y1, exp(s) \odot y_2 + b])
        When flag_affine == False, H(y) = concante([y1, y_2 + b])
        where, [s, b] = NN(y1)
        """
        super(AffineCouplingGlow, self).__init__()
        
        self.flag_affine = flag_affine
        self.flag_detjac = flag_detjac
        
        if feat_dim % 2 > 0:
            print("AffineCoulingGlow(feat_dim), feat_dim is an odd number?!")
            sys.exit(1)
        
        if self.flag_affine:
            self.m_nn_outdim = feat_dim
        else:
            self.m_nn_outdim = feat_dim//2
        
        # create network
        self.m_conv = torch_nn.Sequential(
            Conv2dForGlow(feat_dim//2, conv_out_dim, kernel_size=3, padding=1),
            torch_nn.ReLU(),
            Conv2dForGlow(conv_out_dim, conv_out_dim, kernel_size=1, padding=0),
            torch_nn.ReLU(),
            ZeroInitConv2dForGlow(conv_out_dim, self.m_nn_outdim, 
                                  kernel_size=3, padding=1)
        )
        
        # no bias, normal initial weight
        self.m_conv[0]._zerobias()
        self.m_conv[0]._normal_weight()
        self.m_conv[2]._zerobias()
        self.m_conv[2]._normal_weight()
        return
    
    def _detjac(self, log_scale, factor=1):
        # (batch, dim1, dim2, ..., feat_dim) -> (batch)
        # sum over dim1, ... feat_dim
        return sum_over_keep_batch(log_scale/factor)
        
    def _nn_trans(self, y1):
        if self.flag_affine:
            log_scale, bias = self.m_conv(y1).chunk(2, -1)
            # follow openai implementation
            scale = torch.sigmoid(log_scale + 2)
            log_scale = torch.log(scale)            
        else:
            bias = self.m_conv(y1)
            scale = torch.ones_like(y1)
            log_scale = torch.zeros_like(y1)
        return scale, bias, log_scale
        
    def forward(self, y, factor=1):
        """AffineCoulingGlow(y)
        input
        -----
          y: tensor, (batch, dim1, dim2, ..., feat_dim)
        
        output
        ------
          out: tensor, (batch, dim1, dim2, ..., feat_dim)
        """
        # split
        y1, y2 = y.chunk(2, -1)
        scale, bias, log_scale = self._nn_trans(y1)
        
        # transform
        x1 = y1
        x2 = (y2 + bias) * scale

        # concatenate
        x = torch.cat([x1, x2], dim=-1)
        
        if self.flag_detjac:
            return x, self._detjac(log_scale, factor)
        else:
            return x
        
        
    def reverse(self, x):
        # split
        x1, x2 = x.chunk(2, -1)
        # reverse transform
        y1 = x1
        scale, bias, log_scale = self._nn_trans(y1)
        y2 = x2 / scale - bias
        #
        return torch.cat([y1, y2], dim=-1)
        


class SqueezeForGlow(torch_nn.Module):
    """SqueezeForGlow
    Squeeze layer for Glow
    See doc of __init__ for different operation modes
    
    Example:
        data = torch.randn([2, 4, 4, 3])
        m_squeeze = SqueezeForGlow()
        data_squeezed = m_squeeze(data)
        data_unsqu = m_squeeze.reverse(data_squeezed)
        print(data)
        print(data_squeezed)
        print(torch.std(data_unsqu - data))

        print(data[0, :, :, 0])
        print(data_squeezed[0, :, :, 0])
        print(data_squeezed[0, :, :, 1])
        print(data_squeezed[0, :, :, 2])
        print(data_squeezed[0, :, :, 3])
    """
    def __init__(self, mode = 1):
        """SqueezeForGlow(mode=1)
        Args
        ----
          mode: int, 1: for image
                     2: for audio
        mode == 1:
          (batch, height, width, channel)->(batch, height/2, width/2, channel*4)
          
        """
        super(SqueezeForGlow, self).__init__()
        self.m_mode = mode
        return
    
    def get_squeeze_factor(self):
        if self.m_mode == 1:
            # for image, the channel number will be compressed by 4
            return 4
    
    def forward(self, x):
        """
        """
        if self.m_mode == 1:
            # assume (batch, height, weight, channel)
            if len(x.shape) != 4:
                print("SqueezeForGlow(mode=1)")
                print(", input should be (batch, height, weight, channel)")
                sys.exit(1)
            batch, height, width, channel = x.shape
            # (batch, height, 2, width, 2, channel)
            x_squeezed = x.view(batch, height // 2, 2, width // 2, 2, channel)
            # (batch, height, width, channel * 2 * 2)
            x_squeezed = x_squeezed.permute(0, 1, 3, 5, 2, 4).contiguous()
            x_squeezed = x_squeezed.view(batch, height//2, width//2, channel*4)
        else:
            print("SqueezeForGlow not implemented")
        return x_squeezed

    def reverse(self, x_squeezed):
        if self.m_mode == 1:
            # assume (batch, height, weight, channel)
            if len(x_squeezed.shape) != 4:
                print("SqueezeForGlow(mode=1)")
                print(", input should be (batch, height, weight, channel)")
                sys.exit(1)
            batch, height, width, channel = x_squeezed.shape
            x = x_squeezed.view(batch, height, width, channel // 4, 2, 2)
            # (batch, height * 2, width * 2, channel)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(batch, height*2, width*2, channel//4)
        else:
            print("SqueezeForGlow not implemented")
        return x


    
class PriorTransform(torch_nn.Module):
    """Prior transformation at the end of each Glow block
    
    This is not written in paper but implemented in official code. 
    https://github.com/rosinality/glow-pytorch/issues/11
    This is wrapper around the split operation. However, additional
    affine transformation is included.

    Given y,
    If flag_split == True:
      x, z_1 <- y.split()
      z_0 <- (z_1 - f_bias(x)) / f_scale(x)
      In native implementation, we can directly evaluate N(z_1; 0, I).
      However, this block further converts z_1 -> z_0
    
    If flag_split == False:
      if flag_final_block == True:
         z_1 <- y     
         z_0 <- (z_1 - f_bias(0)) / f_scale(0), final latent
         x <- None    , no input for the next Glowblock
      else
         x   <- y     , which is used for the next Glowblock
         x   <- (x - f_bias(0)) / f_scale(0), input to the next GlowBlock
         z_0 <- None  , no split output
    """
    def __init__(self, feat_dim, flag_split, flag_final_block):
        """PriorTransform(feat_dim)
        
        Args
        ----
          feat_dim: int, feature dimension or channel number
                    input tensor should be (batch, dim1, dim2, ..., feat_dim)
                    image should be (batch, height, weight, feat_dim)
          flag_split: bool, split or not split
          flag_final_block: bool, whether this is the for the final block
        """
        super(PriorTransform, self).__init__()
        
        self.flag_split = flag_split
        if flag_split:
            self.m_nn = ZeroInitConv2dForGlow(feat_dim // 2, feat_dim)
        else:
            self.m_nn = ZeroInitConv2dForGlow(feat_dim, feat_dim * 2)
        self.flag_final_block = flag_final_block

        if flag_final_block and flag_split:
            print("PriorTransform flag_split and flag_final_block are True")
            print("This is unexpected. please check model definition")
            sys.exit(1)
        return
    
    def _detjac(self, log_scale, factor=1):
        # log|\prod 1/exp(log_scale)| = -\sum log_scale
        # note that we should return a tensor (batch,)
        return sum_over_keep_batch(-1 * log_scale / factor)
    
    def forward(self, y, factor=1):
        """PriorTransform(y)

        y -> H() -> [x, z_0]
        
        input
        -----
          y: (batch, dim1, ..., feat_dim)
          
        output
        ------
          x: tensor or None, input to the next GlowBlock
          z_0: tensor or None, latent variable for evaluating N(z_0; 0, I)
          log_detjac: scalar
        
        Note that
        If self.flag_split==True, x, z_0 will (batch, dim1, ..., feat_dim//2)
        If self.flag_split==False and self.flag_final_block==True:
           x = None, which indicates no input for the next GlowBlock
           z_0,  (batch, dim1, ..., feat_dim)
        If self.flag_split==False and self.flag_final_block==False:
           z_0 = None, which indicates no latent output from this GlowBlock
           x,  (batch, dim1, ..., feat_dim), input to the next GlowBlock
        """
        if not self.flag_split:            
            zeros = torch.zeros_like(y)
            z_mean, z_log_std = self.m_nn(zeros).chunk(2, -1)
            if self.flag_final_block:
                # For z_1 <- y
                # z_0 <- (z_1 - f_bias(zero)) / f_scale(zero)
                # x <- None
                z_0 = (y - z_mean) / torch.exp(z_log_std)
                x = None
            else:
                # z_0 <- None
                # x <- (z_1 - f_bias(zero)) / f_scale(zero)
                z_0 = None
                x = (y - z_mean) / torch.exp(z_log_std)
        else:
            # For x, z_1 <- y.split()
            # z_0 <- (z_1 - f_bias(x)) / f_scale(x)
            x, z_1 = y.chunk(2, -1)

            z_mean, z_log_std = self.m_nn(x).chunk(2, -1)
            z_0 = (z_1 - z_mean) / torch.exp(z_log_std)            
        return x, z_0, self._detjac(z_log_std, factor)
    
    def reverse(self, x, z_out):
        """PriorTransform(y)
        y <- H() <- x, z_0
        
        input
        -----
          x:  tensor or None
          z_0: tensor or None
        output
        ------
          y: (batch, dim1, ..., feat_dim)

        Note that
        If self.flag_split==True 
           x, z_out should be (batch, dim1, ..., feat_dim//2)
        If self.flag_split==False and self.flag_final_block==True:
           x = None, which indicates no input for from the following GlowBlock
           z_0,  (batch, dim1, ..., feat_dim)
        If self.flag_split==False and self.flag_final_block==False:
           z_0 = None, which indicates no latent additional this GlowBlock
           x,  (batch, dim1, ..., feat_dim), input from the following GlowBlock
        """
        if self.flag_split:
            if x is not None:
                z_mean, z_log_std = self.m_nn(x).chunk(2, -1)
                z_tmp = z_out * torch.exp(z_log_std) + z_mean
                y_tmp = torch.cat([x, z_tmp], -1)
            else:
                print("PriorTransform.reverse receives None")
                sys.exit(1)
        else:
            if self.flag_final_block:
                zeros = torch.zeros_like(z_out)
                z_mean, z_log_std = self.m_nn(zeros).chunk(2, -1)
                y_tmp = z_out * torch.exp(z_log_std) + z_mean
            else:
                zeros = torch.zeros_like(x)
                z_mean, z_log_std = self.m_nn(zeros).chunk(2, -1)
                y_tmp = x * torch.exp(z_log_std) + z_mean
        return y_tmp
    

class FlowstepGlow(torch_nn.Module):
    """FlowstepGlow
    One flow step in Glow
    """
    def __init__(self, feat_dim, flag_affine=True, conv_coup_dim=512):
        """FlowstepGlow(feat_dim, flag_affine=True)
        
        Args:
        -----
          feat_dim: int, dimension of input feature (channel number of image)
                    feat_dim must be an even number
          flag_affine: bool, whether use affine or additive transformation in
                       AffineCouplingGlow layer (see AffineCouplingGlow)
                       default True.
          conv_coup_dim: int, dimension of intermediate cnn layer in coupling
                       default 512,  (see AffineCouplingGlow)
        
        It assumes that y -> H(.) -> x, where H(.) is FlowstepGlow.forward
        """
        super(FlowstepGlow, self).__init__()
        
        self.flag_affine = flag_affine
        
        # layers
        self.m_actnorm = ActNorm(feat_dim, flag_detjac=True)
        self.m_invtrans = InvertibleTrans(feat_dim, flag_detjac=True)
        self.m_coupling = AffineCouplingGlow(feat_dim, conv_coup_dim, 
                                            flag_affine, flag_detjac=True)
        return
    
    def forward(self, y):
        x_tmp, log_tmp1 = self.m_actnorm(y)
        x_tmp, log_tmp2 = self.m_invtrans(x_tmp)
        x_tmp, log_tmp3 = self.m_coupling(x_tmp)
        return x_tmp, log_tmp1 + log_tmp2 + log_tmp3
    
    def reverse(self, x):
        # prevent accidental reverse during training
        y_tmp = self.m_coupling.reverse(x)
        y_tmp = self.m_invtrans.reverse(y_tmp)
        y_tmp = self.m_actnorm.reverse(y_tmp)
        return y_tmp
    
    
class GlowBlock(torch_nn.Module):
    """GlowBlock 
    
    One Glow block, squeeze + step_of_flow + (split), Fig2.(b) in original paper
    
    Example:
        m_glow = GlowBlock(3, num_flow_step=32)
        data = torch.randn([2, 64, 64, 3])
        x, z, detjac = m_glow(data)

        m_glow.eval()
        data_new = m_glow.reverse(x, z)
        #print(m_glow.training)
        #print(x, z)
        print(torch.std(data_new - data))
    """
    def __init__(self, feat_dim, num_flow_step=12, conv_coup_dim = 512,
                 flag_split=True, flag_final_block=False, 
                 flag_affine=True, squeeze_mode=1):
        """GlowBlock(feat_dim, num_flow_step=12, conv_coup_dim = 512,
         flag_split=True, flag_affine=True, squeeze_mode=1)
        
        Args
        ----
          feat_dim: int, dimension of input feature (channel number of image)
                    feat_dim must be an even number
          num_flow_step: int, number of flow steps, default 12
          conv_coup_dim: int, dimension of intermediate cnn layer in coupling
                       default 512,  (see AffineCouplingGlow)
          flag_split:  bool, whether split out. 
                       Last GlowBlock uses flag_split=False
                       default True
          flag_final_block: bool, whether this is the final GlowBlock
                       default False
          flag_affine: bool, whether use affine or additive transformation in
                       AffineCouplingGlow layer (see AffineCouplingGlow)
                       default True.
          squeeze_mode: int, mode for squeeze, default 1 (see SqueezeForGlow)
        """
        super(GlowBlock, self).__init__()
        
        # squeeze
        self.m_squeeze = SqueezeForGlow(squeeze_mode)
        # number of feat-dim after sequeeze (other channels)
        squeezed_feat_dim = feat_dim * self.m_squeeze.get_squeeze_factor()
        
        # steps of flow
        self.m_flow_steps = []
        for i in range(num_flow_step):
            self.m_flow_steps.append(
                FlowstepGlow(squeezed_feat_dim, flag_affine, conv_coup_dim))
        self.m_flow_steps = torch_nn.ModuleList(self.m_flow_steps)
        
        # prior transform
        self.flag_split = flag_split
        self.flag_final_block = flag_final_block

        if self.flag_final_block and self.flag_split:
            print("GlowBlock flag_split and flag_final_block are True")
            print("This is unexpected. Please check model definition")
            sys.exit(1)
        
        self.m_prior = PriorTransform(
            squeezed_feat_dim, self.flag_split, self.flag_final_block)
        return
        
    def forward(self, y):
        """x, z, log_detjac = GlowBlock(y) 
        
        input
        -----
          y: tensor, (batch, height, width, channel)
          
        output
        ------
          x: tensor, (batch, height, width, channel//2), 
          z: tensor, (batch, height, width, channel//2), 
          log_detjac: tensor or scalar
          
        For multi-scale glow, z is the whitenned noise
        """
        log_detjac = 0
        
        # squeeze
        y_suqeezed = self.m_squeeze(y)

        # flows
        x_tmp = y_suqeezed                
        for m_flow in self.m_flow_steps:
            x_tmp, log_detjac_tmp = m_flow(x_tmp)
            log_detjac += log_detjac_tmp

        # prior transform
        x, z, log_detjac_tmp = self.m_prior(x_tmp)
        log_detjac += log_detjac_tmp

        
        # [x, z] should have the same size as input y_suqeezed
        return x, z, log_detjac
            
    def reverse(self, x, z):
        """
        """
        # prior
        x_tmp = self.m_prior.reverse(x, z)

        # flow
        for m_flow in self.m_flow_steps[::-1]:
            x_tmp = m_flow.reverse(x_tmp)

        # squeeze
        y = self.m_squeeze.reverse(x_tmp)
        return y

    

class Glow(torch_nn.Module):
    """Glow
    """
    def __init__(self, feat_dim, flow_step_num=32, flow_block_num=4,
                flag_affine=False, conv_coup_dim=512, squeeze_mode=1):
        """Glow(feat_dim, flow_step_num=32, flow_block_num=4, 
                flag_affine=True, conv_coup_dim=512, squeeze_mode=1)
                
        Args
        ----
          feat_dim: int, dimension of feature, or channel of input image
          flow_step_num: int, number of flow steps per block, default 32
          flow_block_num: int,  number of flow blocks, default 4
          flag_affine: bool, whether use affine transformation or not
                       default True, see AffineCouplingLayer
          conv_coup_dim: int, channel size of intermediate conv layer in 
                      coupling layer NN(). see AffineCouplingLayer
          squeeze_mode: int, mode for suqeezing. 
                        1 for image. See squeezeLayer
        """
        super(Glow, self).__init__()
        
        
        self.m_blocks = []
        self.m_flag_splits = []
        for i in range(flow_block_num):
            
            # Whether the block uses split or not is completely determined by
            #  whether this block is the last block or not
            
            # last block does not split output
            flag_split = True if i < (flow_block_num - 1) else False
            
            # save this information for generating random noise
            self.m_flag_splits.append(flag_split)
            
            # whether this is the final block
            flag_final_block = True if i == (flow_block_num - 1) else False
            
            self.m_blocks.append(
                GlowBlock(
                    feat_dim * (2**i), flow_step_num, conv_coup_dim,
                    flag_split=flag_split, flag_final_block=flag_final_block,
                    flag_affine=flag_affine, 
                    squeeze_mode=1))
        self.m_blocks = torch_nn.ModuleList(self.m_blocks)
        return

    def _normal_lh(self, noise):
        # likelihood of normal distribution on the given noise
        return -0.5 * np.log(2 * np.pi) - 0.5 * noise ** 2
    
    def forward(self, y):
        """Glow.forward(y)
        Conducts y -> H(.) -> z, where z is supposed to be Gaussian noise
        
        
        input
        -----
          y: tensor, (batch, dim1, dim2, ..., feat_dim)
             for image, (batch, height, width, channel)
             
        output
        ------
          z: list of tensor, random noise from each block
          neg_logp_y: scalar, - log p(y)
          logp_z: scalar, -log N(z), averaged over batch and pixels
          logdet: scalar, -|det dH(.)/dy|, averaged over batch and pixels
        
        Because Glow uses multi-scale structure, z will be a list of noise
        """
        batch_size = y.shape[0]
        
        # for image, np.log(2) computes bit
        # np.prod([dim for dim in y.shape[1:]]) is the image size in pixels
        factor = np.log(2) * np.prod([dim for dim in y.shape[1:]])

        z_bags = []
        log_detjac = 0
        log_pz = 0
        
        h_tmp = y
        for m_block in self.m_blocks:
            h_tmp, z_tmp, log_detjac_tmp = m_block(h_tmp)
            
            z_bags.append(z_tmp)
            log_detjac += log_detjac_tmp / factor

            # keep log_pz for each data in batch (batchsize,)
            log_pz += sum_over_keep_batch(self._normal_lh(z_tmp)) / factor
            
            
        # average over batch and pixels
        neg_logp_y = -(log_pz + log_detjac).mean()
        
        return z_bags, neg_logp_y, \
            log_pz.mean(), log_detjac.mean()
        
    def reverse(self, z_bags):
        """ y = Glow.reverse(z_bags)
        
        input
        -----
          z_bags: list of tensors
        
        output
        ------
          y: tensor, (batch, dim1, dim2, ..., feat_dim)

        The random noise in z_bags should be compatible with the 
        model. You may use Glow.get_z_noises to retrieve a z_bags
        """
        for i, (z, m_block) in enumerate(zip(z_bags[::-1],
                                             self.m_blocks[::-1])):
            if i == 0:
                # the last block without split
                y_tmp = m_block.reverse(None, z)
            else:
                y_tmp = m_block.reverse(y_tmp, z)
        return y_tmp

    def get_z_noises(self, image_size, noise_std=0.7, batchsize=16):
        """z_bags = Glow.get_z_noises(image_size, noise_std=0.7, batchsize=16)
        Return a list of random noises for random sampling
        
        input
        -----
          image_size: int, size of the image, assume image is square, 
                      this number just specifies the height / width
          noise_std: float, std of Gaussian noise, default 0.7
          batchsize: int, batch size of this random data, default 16
        
        output
        ------
          z_bags: list of tensors
        
        Shape of the random noise in z_bags is decided by Glow configuration.
        Glow.reverse(z_bags) can be used to produce image from this z_bags
        """

        device = next(self.parameters()).device

        z_bags = []
        tmp_im_size = image_size
        tmp_chan = 3
        for flag_split in self.m_flag_splits:
            if flag_split:
                tmp_im_size = tmp_im_size // 2
                tmp_chan = tmp_chan * 2
            else:
                tmp_im_size = tmp_im_size // 2
                tmp_chan = tmp_chan * 4
            z_tmp = torch.randn([batchsize, tmp_im_size, tmp_im_size, tmp_chan],
                                dtype=nii_io_conf.d_dtype, device=device)
            z_bags.append(z_tmp * noise_std)
        return z_bags
            
            
        


if __name__ == "__main__":
    print("Definition of Glow and its components")
