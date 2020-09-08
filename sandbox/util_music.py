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

import config as prj_conf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


class HzCentConverter(torch_nn.Module):
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
        """
        """
        self.m_v_hz = u_hz

        self.m_hz_ref = hz_ref
        # one semitone cover 100 cents
        # thus, -50 on bottom, +50 on top
        # cent2hz(hz2cent(librosa.note_to_hz('C1'))-50)
        self.m_base_hz = torch.tensor([base_hz])
        # cent2hz(hz2cent(librosa.note_to_hz('B6'))+50)
        self.m_top_hz = torch.tensor([top_hz])
        # quantization interval
        self.m_bins = bins
        
        # for quantization
        self.m_base_cent = self.hz2cent(self.m_base_hz)
        self.m_top_cent = self.hz2cent(self.m_top_hz)
        
        # bottom bin

        # for de-quantization
        # librosa.note_to_hz('C1')
        self.m_1st_cent = self.hz2cent(torch.tensor([cent_1st]))
        # librosa.note_to_hz('B6')
        self.m_last_cent = self.hz2cent(torch.tensor([cent_last]))
        
        self.m_quan_cent_dis = (self.m_last_cent - self.m_1st_cent)/self.m_bins
        self.m_dis_cent = torch_nn.Parameter(
            torch.linspace(self.m_1st_cent.numpy()[0], 
                           self.m_last_cent.numpy()[0], 
                           self.m_bins),
            requires_grad=False)

        # for viterbi decoding
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
        return 1200 * torch.log2(hz/self.m_hz_ref)

    def cent2hz(self, cent):
        return torch.pow(2, cent/1200) * self.m_hz_ref

    def quantize_hz(self, hz):
        cent = self.hz2cent(hz)
        q_bin = torch.round((cent - self.m_base_cent) * self.m_bins /\
                            (self.m_top_cent - self.m_base_cent)) 
        q_bin = torch.min([torch.max([0, self.m_bins]), self.m_bins - 1]) +1 
        return q_bin 

    def dequantize_hz(self, quantized_cent):
        cent = quantized_cent * self.m_quan_cent_dis + self.m_1st_cent
        return self.cent2hz(cent)    

    def recover_f0(self, bin_mat):
        
        if bin_mat.shape[0] != 1:
            print("F0 generation only support batchsize=1")
            sys.exit(1)

        # assume bin_mat (1, length, m_bins)
        if not self.m_viterbi_decode:
            # normal sum
            cent = torch.sum(bin_mat * self.m_dis_cent, axis=2) /\
                   torch.sum(bin_mat, axis=2)
            return self.cent2hz(cent)
        else:
            tmp_bin_mat = bin_mat.to('cpu')

            # viterbi decode:
            with torch.no_grad():
                prob_u = torch.ones_like(tmp_bin_mat) \
                         - torch.mean(tmp_bin_mat, axis=2, keepdim=True)
                tmp_bin_mat = torch.cat([tmp_bin_mat, prob_u],axis=2).squeeze(0)
                tmp_bin_mat = tmp_bin_mat.numpy()

                # viterbi decoding
                quantized_cent = nii_dy.viterbi_decode(
                    self.m_viterbi_init, self.m_viterbi_tran, tmp_bin_mat * 0.5)

                # convert state to F0
                u_idx = quantized_cent>=self.m_bins
                
                prob_m = torch.zeros_like(bin_mat)
                for idx, i in enumerate(quantized_cent):
                    s_idx = np.max([i - 4, 0])
                    e_idx = np.min([i+5, self.m_bins])
                    prob_m[0, idx, s_idx:e_idx] = bin_mat[0, idx, s_idx:e_idx]
                cent = torch.sum(prob_m * self.m_dis_cent, axis=2) /\
                       torch.sum(prob_m, axis=2)
                f0 = self.cent2hz(cent)
                f0[0, u_idx]=0
            return f0

if __name__ == "__main__":
    print("util_music")
