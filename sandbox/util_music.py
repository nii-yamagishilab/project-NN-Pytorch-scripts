#!/usr/bin/env python
"""
util_music.py

Utilities for music applications
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

import sandbox.dynamic_prog as nii_dy

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


class HzCentConverter(torch_nn.Module):
    """
    HzCentConverter: an interface to convert F0 to cent, probablity matrix
    and do reverse conversions
    """
    def __init__(self, 
                 u_hz = 0, 
                 hz_ref = 10, 
                 base_hz = 31.77219916398751,
                 top_hz = 2033.4207464952,
                 bins = 360,
                 cent_1st = 32.70319566257483,
                 cent_last = 1975.5332050244956,
    ):
        super(HzCentConverter, self).__init__()

        # unvoiced F0
        self.m_v_hz = u_hz
        
        # reference for cent calculation
        self.m_hz_ref = hz_ref
        
        # quantized resolution
        # originally, bins = 360 -> 12 * 6 * 5, 12 semitones, 6 octaves
        # each semitone is further divided to 5 intervals
        self.m_fine_interval = 5

        #####
        # for quantization
        #####
        # one semitone cover 100 cents
        # thus, -50 on bottom, +50 on top
        # cent2hz(hz2cent(librosa.note_to_hz('C1'))-50)
        self.m_base_hz = torch.tensor([base_hz])
        # cent2hz(hz2cent(librosa.note_to_hz('B6'))+50)
        self.m_top_hz = torch.tensor([top_hz])
        # quantization interval
        self.m_bins = bins
        
        
        self.m_base_cent = self.hz2cent(self.m_base_hz)
        self.m_top_cent = self.hz2cent(self.m_top_hz)
        
        #####
        # for de-quantization
        #####
        # librosa.note_to_hz('C1')
        self.m_1st_cent = self.hz2cent(torch.tensor([cent_1st]))
        # librosa.note_to_hz('B6')
        self.m_last_cent = self.hz2cent(torch.tensor([cent_last]))
        # quantized cent per bin
        self.m_quan_cent_dis = (self.m_last_cent - self.m_1st_cent)/self.m_bins

        # quantized cents as a tentor
        self.m_dis_cent = torch_nn.Parameter(
            torch.linspace(self.m_1st_cent.numpy()[0], 
                           self.m_last_cent.numpy()[0], 
                           self.m_bins),
            requires_grad=False)

        # quantized F0 as a tensor
        self.m_dis_f0 = self.cent2hz(
            torch.linspace(self.m_1st_cent.numpy()[0], 
                           self.m_last_cent.numpy()[0], 
                           self.m_bins))

        #####
        # for viterbi decoding
        #####
        self.m_viterbi_decode = True
        # initial state probablity
        self.m_viterbi_init = np.ones(self.m_bins * 2) / (self.m_bins * 2)

        # transition probability
        def _trans_mat():
            max_cent = 12
            p_vv = 0.99
            p_uv = 1 - p_vv
            # transition probabilities inducing continuous pitch
            xx, yy = np.meshgrid(range(self.m_bins), range(self.m_bins))
            tran_m_v = np.maximum(max_cent - abs(xx - yy), 0)
            tran_m_v = tran_m_v / np.sum(tran_m_v, axis=1)[:, None]

            # unvoiced part
            tran_m_u = np.ones([self.m_bins, self.m_bins])/self.m_bins
            tmp1 = np.concatenate([tran_m_v * p_vv, tran_m_u * p_uv], axis=1)
            tmp2 = np.concatenate([tran_m_v * p_uv, tran_m_u * p_vv], axis=1)
            trans = np.concatenate([tmp1, tmp2], axis=0)
            return trans
        self.m_viterbi_tran = _trans_mat()

    def hz2cent(self, hz):
        """
        hz2cent(self, hz)
        Convert F0 Hz in to Cent
        
        Parameters
        ----------
        hz: torch.tensor
        
        Return
        ------
        : torch.tensor
        """
        return 1200 * torch.log2(hz/self.m_hz_ref)

    def cent2hz(self, cent):
        return torch.pow(2, cent/1200) * self.m_hz_ref

    def quantize_hz(self, hz):
        cent = self.hz2cent(hz)
        q_bin = torch.round((cent - self.m_base_cent) * self.m_bins /\
                            (self.m_top_cent - self.m_base_cent)) 
        q_bin = torch.min([torch.max([0, q_bin]), self.m_bins - 1]) +1 
        return q_bin 

    def dequantize_hz(self, quantized_cent):
        cent = quantized_cent * self.m_quan_cent_dis + self.m_1st_cent
        return self.cent2hz(cent)    

    def f0_to_mat(self, f0_seq, var=625):
        """
        f0_to_mat(self, f0_seq)
        Convert F0 sequence (hz) into a probability matrix.

           Jong Wook Kim, Justin Salamon, Peter Li, and Juan Pablo Bello. 2018. 
           CREPE: A Convolutional Representation for Pitch Estimation. 
           In Proc. ICASSP, 161-165
        
        Parameters
        ----------
        f0_seq: torch.tensor (1, N, 1)

        Return
        ------
        target_mat: torch.tensor (1, N, bins)
            created probability matrix for f0
        """
        if f0_seq.dim() != 3:
            print("f0 sequence loaded in tensor should be in shape (1, N, 1)")
            sys.exit(1)
        
        # voiced / unvoiced indix
        v_idx = f0_seq > self.m_v_hz
        u_idx = ~v_idx

        # convert F0 Hz to cent
        target = torch.zeros_like(f0_seq)
        target[v_idx] = self.hz2cent(f0_seq[v_idx])
        target[u_idx] = 0

        # target
        # since target is (1, N, 1), the last dimension size is 1
        # self.m_dis_cent (bins) -> propagated to (1, N, bins)
        target_mat = torch.exp(-torch.pow(self.m_dis_cent - target, 2)/2/std)
        
        # set unvoiced to zero
        for idx in range(target_mat.shape[0]):
            target_mat[idx, u_idx[idx, :, 0], :] *= 0.0
        #target_mat[0, u_idx[0, :, 0], :] *= 0.0
        
        # return
        return target_mat
        
    def recover_f0(self, bin_mat):
        """ 
        recover_f0(self, bin_mat)
        Produce F0 from a probability matrix.
        This is the inverse function of f0_to_mat.

        By default, use Viterbi decoding to produce F0.
        
          Matthias Mauch, and Simon Dixon. 2014. 
          PYIN: A Fundamental Frequency Estimator Using Probabilistic 
          Threshold Distributions. In Proc. ICASSP, 659-663.

        Parameters
        ----------
        bin_mat: torch.tensor (1, N, bins)

        Return
        ------
        f0: torch.tensor(1, N, 1)
        """
        # check
        if bin_mat.shape[0] != 1:
            print("F0 generation only support batchsize=1")
            sys.exit(1)

        if bin_mat.dim() != 3 or bin_mat.shape[-1] != self.m_bins:
            print("bin_mat should be in shape (1, N, bins)")
            sys.exit(1)
            
        # generation
        if not self.m_viterbi_decode:
            # normal sum 
            cent = torch.sum(bin_mat * self.m_dis_cent, axis=2) / \
                   torch.sum(bin_mat, axis=2)
            return self.cent2hz(cent)
        else:
            tmp_bin_mat = bin_mat.to('cpu')

            # viterbi decode:
            with torch.no_grad():
                
                # observation probablity for unvoiced states
                prob_u = torch.ones_like(tmp_bin_mat) \
                         - torch.mean(tmp_bin_mat, axis=2, keepdim=True)
                
                # concatenate to observation probability matrix 
                #  [Timestep, m_bins * 2], 
                #  m_bins is the number of quantized F0 bins
                #  another m_bins is for the unvoiced states
                tmp_bin_mat = torch.cat([tmp_bin_mat, prob_u],axis=2).squeeze(0)
                
                # viterbi decoding. Numpy is fast?
                tmp_bin_mat = tmp_bin_mat.numpy()
                quantized_cent = nii_dy.viterbi_decode(
                    self.m_viterbi_init, self.m_viterbi_tran, tmp_bin_mat * 0.5)
                
                # unvoiced state sequence (states in [m_bins, m_bins*2])
                u_idx = quantized_cent>=self.m_bins
                
                # based on viterbi best state, do weighted sum over a beam
                # Equation from 
                # https://github.com/marl/crepe/blob/master/crepe/core.py#L108
                prob_m = torch.zeros_like(bin_mat)
                for idx, i in enumerate(quantized_cent):
                    s_idx = np.max([i - 4, 0])
                    e_idx = np.min([i+5, self.m_bins])
                    prob_m[0, idx, s_idx:e_idx] = bin_mat[0, idx, s_idx:e_idx]
                
                cent = torch.sum(prob_m * self.m_dis_cent, axis=2) / \
                       torch.sum(prob_m, axis=2)

                # from cent to f0
                f0 = self.cent2hz(cent)
                # unvoiced
                f0[0, u_idx]=0
                
            return f0

    def f0_probmat_postprocessing(self, f0_prob_mat):
        """
        f0_prob_mat = f0_prob_mat_post(f0_prob_mat)

        input
        -----
          f0_prob_mat: torch tensor of shape (bathcsize, length, bins)
        
        output
        ------
          f0_prob_mat_new: same shape as f0_prob_mat
        """
        if f0_prob_mat.shape[-1] != self.m_bins:
            print("Last dimension of F0 prob mat != {:d}".format(self.m_bins))
            sys.exit(1)

        if f0_prob_mat.shape[0] > 1:
            print("Cannot support batchsize > 1 for dynamic programming")
            sys.exit(1)
        
        
        # observation probablity for unvoiced states
        prob_u = torch.ones_like(f0_prob_mat) \
                 - torch.mean(f0_prob_mat, axis=2, keepdim=True)    
        tmp_bin_mat = torch.cat([f0_prob_mat, prob_u],axis=2).squeeze(0)

        # viterbi decoding. Numpy is fast?
        tmp_bin_mat = tmp_bin_mat.to('cpu').numpy()
        quantized_cent = nii_dy.viterbi_decode(
            self.m_viterbi_init, self.m_viterbi_tran, tmp_bin_mat * 0.5)
        u_idx = quantized_cent>=self.m_bins

        mat_new = torch.zeros_like(f0_prob_mat)
        for idx, i in enumerate(quantized_cent):
            if i < self.m_bins:
                sidx = np.max([i - 4, 0])
                eidx = np.min([i+5, self.m_bins])
                mat_new[0, idx, sidx:eidx] = f0_prob_mat[0,idx,sidx:eidx]
                mat_new[0, idx, sidx:eidx] /= mat_new[0, idx, sidx:eidx].sum()
                
        return mat_new
        
if __name__ == "__main__":
    print("util_music")
    
