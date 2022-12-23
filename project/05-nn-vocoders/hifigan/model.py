#!/usr/bin/env python
"""
model.py for HiFiGAN

HifiGAN is based on code in from https://github.com/jik876/hifi-gan

HiFi-GAN: Generative Adversarial Networks for Efficient and 
High Fidelity Speech Synthesis
By Jungil Kong, Jaehyeon Kim, Jaekyoung Bae

MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

import core_scripts.other_tools.debug as nii_debug
import sandbox.util_frontend as nii_frontend

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


#########
## Loss definition
#########

class LossMel(torch_nn.Module):
    """ Wrapper to define loss function 
    """
    def __init__(self, sr):
        super(LossMel, self).__init__()
        """
        """
        # extractir
        fl = 1024 
        fs = 256
        fn = 1024
        num_mel = 80

        self.m_frontend = nii_frontend.MFCC(
            fl, fs, fn, sr, num_mel,
            with_emphasis=False, with_delta=False, flag_for_MelSpec=True)
        # loss function
        self.loss_weight = 45
        self.loss = torch_nn.L1Loss()
        return

    def forward(self, outputs, target):
        with torch.no_grad():
            # (batch, length, 1) -> (batch, length) -> (batch, length, dim)
            target_mel = self.m_frontend(target.squeeze(-1))
        output_mel = self.m_frontend(outputs.squeeze(-1))
        # done
        return self.loss(output_mel, target_mel) * self.loss_weight

#####
## Model Generator definition
##### 

from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

def get_padding(kernel_size, dilation=1):
    # L_out = (L_in + 2*pad - dila * (ker - 1) - 1) // stride + 1
    # stride -> 1
    # L_out = L_in + 2*pad - dila * (ker - 1) 
    # L_out == L_in ->
    # 2 * pad = dila * (ker - 1) 
    return int((kernel_size*dilation - dilation)/2)

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)

class ResBlock1(torch_nn.Module):
    """
    """
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        
        self.leaky_relu_slope = 0.1
        
        self.convs1 = torch_nn.ModuleList([
            weight_norm(
                torch_nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=dilation[0],
                    padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(
                torch_nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=dilation[1],
                    padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(
                torch_nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=dilation[2],
                    padding=get_padding(kernel_size, dilation[2])))
        ])
        # initialize the weight 
        self.convs1.apply(init_weights)

        self.convs2 = torch_nn.ModuleList([
            weight_norm(
                torch_nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=1,
                    padding=get_padding(kernel_size, 1))),
            weight_norm(
                torch_nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=1,
                    padding=get_padding(kernel_size, 1))),
            weight_norm(
                torch_nn.Conv1d(
                    channels, channels, kernel_size, 1, dilation=1,
                    padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)
        return
        
        
    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch_nn_func.leaky_relu(x, self.leaky_relu_slope)
            xt = c1(xt)
            xt = torch_nn_func.leaky_relu(xt, self.leaky_relu_slope)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

            
            
class Generator(torch_nn.Module):
    """
    """
    def __init__(self, in_dim,
                 resblock_kernel_sizes, resblock_dilation_sizes,
                 upsample_rates, upsample_kernel_sizes, 
                 upsample_init_channel):
        super(Generator, self).__init__()
        
        self.leaky_relu_slope = 0.1
        
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.num_kernels = len(resblock_kernel_sizes)
        
        self.upsample_rates = upsample_rates
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.num_upsamples = len(upsample_rates)        
        
        self.upsample_init_channel = upsample_init_channel
        
        self.conv_pre = weight_norm(
            torch_nn.Conv1d(in_dim, upsample_init_channel, 7, 1, padding=3))

        self.ups = torch_nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            # L_out = (L_in - 1) * stride - 2 * pad + dila * (ker - 1) + 1
            # dilation = 1 -> 
            # L_out = (L_in - 1) * stride - 2 * pad + ker
            # L_out = L_in * stride - 2 * pad + (ker - stride)
            self.ups.append(weight_norm(
                torch_nn.ConvTranspose1d(
                    upsample_init_channel//(2**i), 
                    upsample_init_channel//(2**(i+1)),
                    k, u, padding=(k-u)//2)))
        
        # 
        resblock = ResBlock1
        self.resblocks = torch_nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_init_channel//(2**(i+1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = weight_norm(torch_nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
        return
        
    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = torch_nn_func.leaky_relu(x, self.leaky_relu_slope)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = torch_nn_func.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


        
class ModelGenerator(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(ModelGenerator, self).__init__()

        ########## basic config ########
        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(in_dim,out_dim,\
                                                         args, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)
        self.input_dim = in_dim
        self.output_dim = out_dim
        ###############################

        ######
        ## model definition
        ######
        h = prj_conf.options['hifigan_config']
        
        self.m_gen = Generator(in_dim, 
                  h['resblock_kernel_sizes'], h['resblock_dilation_sizes'], 
                  h['upsample_rates'], h['upsample_kernel_sizes'], 
                  h['upsample_initial_channel'])
        
        self.m_mel_loss = LossMel(prj_conf.wav_samp_rate)
        
        self.flag_removed_weight_norm = False


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
    
    def forward(self, x):
        if not self.training and not self.flag_removed_weight_norm:
            self.m_gen.remove_weight_norm()
            self.flag_removed_weight_norm = True
            
        x = self.normalize_input(x)
        gen_output = self.m_gen(x.permute(0, 2, 1)).permute(0, 2, 1)
        return gen_output
    
    def loss_aux(self, nat_wav, gen_wav, data_in):
        return self.m_mel_loss(gen_wav, nat_wav)

#########
## Model Discriminator definition
#########

class DiscriminatorP(torch_nn.Module):
    def __init__(self, period, 
                 kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.leaky_relu_slope = 0.1
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch_nn.ModuleList([
            norm_f(
                torch_nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), 
                                padding=(get_padding(5, 1), 0))),
            norm_f(
                torch_nn.Conv2d(1024, 1024, (kernel_size, 1), 1, 
                                padding=(2, 0))),
        ])
        self.conv_post = norm_f(
            torch_nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        return
    
    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = torch_nn_func.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = torch_nn_func.leaky_relu(x, self.leaky_relu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

class MultiPeriodDiscriminator(torch_nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = torch_nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    
class DiscriminatorS(torch_nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.leaky_relu_slope = 0.1
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = torch_nn.ModuleList([
            norm_f(
                torch_nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(
                torch_nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(
                torch_nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(
                torch_nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(torch_nn.Conv1d(1024, 1, 3, 1, padding=1))
        return
    
    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = torch_nn_func.leaky_relu(x, self.leaky_relu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

    
class MultiScaleDiscriminator(torch_nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = torch_nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = torch_nn.ModuleList([
            torch_nn.AvgPool1d(4, 2, padding=2),
            torch_nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    


class ModelDiscriminator(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(ModelDiscriminator, self).__init__()
        self.m_mpd = MultiPeriodDiscriminator()
        self.m_msd = MultiScaleDiscriminator()
        # done
        return

    
    
    def _feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss*2

    def _discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses
    

    def _generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l
        return loss, gen_losses
    

    def loss_for_D(self, nat_wav, gen_wav_detached, input_feat):
        # gen_wav has been detached
        nat_wav_tmp = nat_wav.permute(0, 2, 1)
        gen_wav_tmp = gen_wav_detached.permute(0, 2, 1)
        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.m_mpd(nat_wav_tmp, gen_wav_tmp)
        
        loss_disc_f, _, _ = self._discriminator_loss(y_df_hat_r, y_df_hat_g)
        
        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.m_msd(nat_wav_tmp, gen_wav_tmp)
        loss_disc_s, _, _ = self._discriminator_loss(y_ds_hat_r, y_ds_hat_g)
        
        return loss_disc_f + loss_disc_s


    def loss_for_G(self, nat_wav, gen_wav, input_feat):
        nat_wav_tmp = nat_wav.permute(0, 2, 1)
        gen_wav_tmp = gen_wav.permute(0, 2, 1)
        # MPD
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.m_mpd(nat_wav_tmp, 
                                                                gen_wav_tmp)
        # MSD
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.m_msd(nat_wav_tmp, 
                                                                gen_wav_tmp)

        loss_fm_f = self._feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = self._feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, _ = self._generator_loss(y_df_hat_g)
        loss_gen_s, _ = self._generator_loss(y_ds_hat_g)
        
        return loss_fm_f + loss_fm_s + loss_gen_f + loss_gen_s


        
if __name__ == "__main__":
    print("Definition of model")

    
