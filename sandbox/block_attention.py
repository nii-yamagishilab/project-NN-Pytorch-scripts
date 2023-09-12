##!/usr/bin/env python
"""
Blocks for attention mechanism

Implementation inspired by https://github.com/soobinseo/Transformer-TTS.git,
Code is re-facotrized and test against the original repo.

DotScaledAttention, MultiheadAttention, MultiheadAttentionWrapper are separated
1. DotScaledAttention: the core attention softmax(QK^T/sqrt(d))V. Masks on 
   attention matrix and output sequence can be provided as optional input
2. MultiheadAttention: it splits data into multiple tensors and calls the former
3. MultiheadAttentionWrapper: wrapper over multiheadAttention with 
   additional operations such aslinear transformation and normalization ...
 
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"

# ================================================
# DotScaledAttention & multi-head implementation
# ================================================

class DotScaledAttention(torch_nn.Module):
    """DotScaledAttention in Transformer
    
    O = q_mask * softmax( (Q K^\top / \sqrt(d)) k_mask ) V, where 
    Q: (batch, length1, dimension)
    K: (batch, length2, dimension)
    V: (batch, length2, dimension2)
    
    k_mask: (batch, length1, length2)
    q_mask: (batch, length1)
    
    Example:
        l_m1 = DotScaledAttention()

        q_data = t.rand([5, 100, 64])
        k_data2 = t.rand([5, 40, 64])
        v_data3 = t.rand([5, 40, 32])

        q_mask = t.ones([5, 100])
        q_mask[0, -4:] = 0
        q_mask[1, -5:] = 0
        q_mask_bin = q_mask.eq(0)

        k_mask = t.ones([5, 100, 40])
        k_mask[0, :, -4:] = 0
        k_mask[1, :, -5:] = 0
        k_mask_bin = k_mask.eq(0)

        o1, a1 = l_m1(q_data, k_data2, v_data3, q_mask_bin, k_mask_bin)
        
    Note: user should provide a k_mask for causal attention
    """
    def __init__(self, attn_dropout=0.0):
        """attention_module = DotScaledAttention(attn_dropout=0.0)

        args
        ----
          attn_dropout: float, dropout rate on attention matrix

        """
        super(DotScaledAttention, self).__init__()

        # dropout on attention weights
        if attn_dropout > .0:
            self.m_attn_dropout = torch_nn.Dropout(p=attn_dropout)
        else:
            self.m_attn_dropout = torch_nn.Identity()
            
        return
    
    def forward(self, Q, K, V, q_mask=None, k_mask=None):
        """O = DotScaledAttention(Q, K, V, q_mask=None, k_mask=None)

        O = q_mask * softmax( (Q K^\top / \sqrt(d)) k_mask ) V
        
        input:
        ------
          Q: tensor, (batch, length1, dimension)
          K: tensor, (batch, length2, dimension)
          V: tensor, (batch, length2, dimension2)
    
          k_mask: None or tensor, (batch, length1, length2)
          q_mask: None or tensor, (batch, length1)
        
        output
        ------
          O: tensor, (batch, length1, dimension2)
          attn: tensor, (batch, length1, length2), attention matrix
          
        k_mask[i] is a mask for the i-th data in the batch, 
          if k_mask[i, j, k]==True, attention[i, j, k] should be zero

        q_mask[i] is a mask for the i-the query, q_mask[i, j]==True
          indicates that output O[i][j, ...] should be masked
        """
        bsize = Q.shape[0]
        feat_dim = Q.shape[-1]
        q_len = Q.shape[1]
        k_len = K.shape[1]
        
        assert K.shape[-1] == Q.shape[-1], "Q and K differ in feat dimension"
        assert K.shape[1] == V.shape[1], "K and V differ in length"

        # Q K^\top
        # attn has shape (length1, length2)
        attn = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(feat_dim)        
        
        # mask the attn matrix if necessary
        if k_mask != None:
            attn = attn.masked_fill(k_mask, -2 ** 32 +1)
            
        # softmax, over length2 of the (batch, length1, length2)
        attn = torch_nn_func.softmax(attn, dim=-1)
        
        # apply q_mask
        if q_mask is not None:
            assert q_mask.ndim == 2 or q_mask.ndim == 3, \
                "q_mask should have 2 or 3 dimensions"
            if q_mask.ndim == 2:
                # (batch, length1) -> (batch, length1, length2)
                mask_tmp = q_mask.unsqueeze(-1).repeat(1, 1, k_len)
            else:
                mask_tmp = q_mask
                
            # mask query (row) that should be dummy
            attn = attn.masked_fill(mask_tmp, 0)
        
        # o = dropout(attn) * V
        O = torch.bmm(self.m_attn_dropout(attn), V)
        return O, attn



class MultiheadAttention(torch_nn.Module):
    """Multihead Attention in Transformer
    
    V, K, Q -> linear -> split -> DotScaledAttention -> concate -> linear ->
    
    Q: (batch, lengthQ, feat_dimK)
    K: (batch, lengthK, feat_dimK)
    V: (batch, lengthK, feat_dimV)
    
    k_mask: (batch, lengthK, lengthQ)
    q_mask: (batch, lengthQ)
    
    Example:
      q_data = torch.rand([5, 100, 64])
      k_data2 = torch.rand([5, 40, 64])
      v_data3 = torch.rand([5, 40, 32])

      q_mask = torch.ones([5, 100])
      q_mask[0, -4:] = 0
      q_mask[1, -5:] = 0
      q_mask_bin = q_mask.eq(0)

      k_mask = torch.ones([5, 100, 40])
      k_mask[0, :, -4:] = 0
      k_mask[1, :, -5:] = 0
      k_mask_bin = k_mask.eq(0)
      
      l_m = MultiheadAttention(64, 32, 4)
      
      data_out = l_m.forward(v_data3, k_data2, q_data, k_mask_bin, q_mask_bin)

    """
    def __init__(self, 
                 feat_dim_k, 
                 feat_dim_v, 
                 num_head=4, 
                 attn_dp = 0.0,
                 flag_cat_q = True):
        """MultiheadAttention(feat_dim_k, feat_dim_v, num_head=4, attn_dp= 0.0)
        
        Args
        ----
          feat_dim_k:    int, feat_dimension of Query and Key
          feat_dim_v:    int, feat_dimension of Value
          num_head:      int, number of heads
          attn_dp:       float, dropout on attention matrix
          flag_cat_q:    bool, concat query and output before linear output?
                         see https://github.com/soobinseo/Transformer-TTS.git
        """
        super(MultiheadAttention, self).__init__()
        
        self.num_head = num_head
        
        if feat_dim_k % self.num_head > 0 or feat_dim_v % self.num_head > 0:
            print("feat_dim_k cannot be divided by num_head")
            sys.exit(1)
        
        self.m_q_fc = torch_nn.Linear(feat_dim_k, feat_dim_k, bias=False)
        self.m_k_fc = torch_nn.Linear(feat_dim_k, feat_dim_k, bias=False)
        self.m_v_fc = torch_nn.Linear(feat_dim_v, feat_dim_v, bias=False)
        
        torch_nn.init.xavier_uniform_(
            self.m_q_fc.weight, gain=torch_nn.init.calculate_gain('linear'))
        torch_nn.init.xavier_uniform_(
            self.m_k_fc.weight, gain=torch_nn.init.calculate_gain('linear'))
        torch_nn.init.xavier_uniform_(
            self.m_v_fc.weight, gain=torch_nn.init.calculate_gain('linear'))
        
        self.m_attn = DotScaledAttention(attn_dp)
        
        # output layer
        self.flag_cat_q = flag_cat_q

        if self.flag_cat_q:
            self.m_output = torch_nn.Linear(feat_dim_k + feat_dim_v, feat_dim_v)
        else:
            self.m_output = torch_nn.Linear(feat_dim_v, feat_dim_v)

        torch_nn.init.xavier_uniform_(
            self.m_output.weight, gain=torch_nn.init.calculate_gain('linear'))

        return
        
    def forward(self, query, key, value, q_mask=None, k_mask=None):
        """O, attn = MultiheadAttention(query, key, value, q_mask, k_mask)
        
        input:
        ------
          Q: (batch, lengthQ, feat_dimK)
          K: (batch, lengthK, feat_dimK)
          V: (batch, lengthK, feat_dimV)

    
          k_mask: None or tensor, (batch, lengthQ, lengthK)
          q_mask: None or tensor, (batch, lengthQ)
        
        output
        ------
          O: tensor, (batch, lengthQ, ...)
          attn: tensor, (batch * head, lengthQ, lengthK), attention matrix
          
        k_mask[i] is a mask for the i-th key/value, k_mask[i, j, k]==True
          indicates that K[i][j, ...] and V[i][k, ...] should be ignored. 
          attention[i][j, k] should be zero

        q_mask[i] is a mask for the i-the query, q_mask[i, j]==True
          indicates that output O[i][j, ...] should be masked
        """

        bsize = value.size(0)
        k_len = key.size(1)
        q_len = query.size(1)
            
        # transform and split the input Q, K, V
        def _trans_split(data_mat, trans_func, head):
            bsize, length, dim = data_mat.shape
            
            # (batch, length, feat_dim) -> (batch, length, feat_dimV)
            # -> (batch, lengthK, num_head, feat_dimV / num_head)
            tmp_mat = trans_func(data_mat).view(bsize, length, head, -1)
            
            # -> ( num_head, batch, lengthK, feat_dimV / num_head)
            tmp_mat = tmp_mat.permute(2, 0, 1, 3).contiguous()
            
            # -> ( num_head * batch, lengthK, feat_dimV / num_head)
            tmp_mat = tmp_mat.view(-1, length, tmp_mat.shape[-1])
            return tmp_mat
        
        value_mul = _trans_split(value, self.m_v_fc, self.num_head)
        key_mul = _trans_split(key, self.m_k_fc, self.num_head)
        query_mul = _trans_split(query, self.m_q_fc, self.num_head)
        
        # duplicate masks to multi heads
        qm = q_mask.repeat(self.num_head, 1) if q_mask is not None else None
        km = k_mask.repeat(self.num_head, 1, 1) if k_mask is not None else None

        # attention and sum
        o_mul, attn = self.m_attn(query_mul, key_mul, value_mul, qm, km)
        
        # recover it back
        # ( num_head * batch, lengthQ, feat_dimV / num_head) ->
        # ( num_head, batch, lengthQ, feat_dimV / num_head) ->
        o_mul = o_mul.view(self.num_head, bsize, q_len, -1)

        # -> ( batch, lengthQ, feat_dimV) 
        o_mat = o_mul.permute(1, 2, 0, 3).contiguous().view(bsize, q_len, -1)
        
        # concatenate the input query and output of attention if necessary
        if self.flag_cat_q:
            # (batch, lengthQ, feat_dimQ + feat_dimV)
            o_mat = torch.cat([o_mat, query], dim=-1)
        
        # linear output
        o_mat = self.m_output(o_mat)
        
        return o_mat, attn


class MultiHeadAttWrapper(MultiheadAttention):
    """ MultiHeadAttWrapper
    Wrapper around multiHeadAttention, with layer normalization 
    and residual output. The residual and layernorm are considered
    as additional operations
    
    input -> MultiHeadAttention -> residual -> layernorm
    
    Same usage as MultiheadAttention.

    """
    def __init__(self, 
                 feat_dim_k, feat_dim_v, 
                 num_head=4, 
                 attn_dropout = 0.0, 
                 flag_cat_q = True, 
                 flag_layernorm_out = True, 
                 flag_residual = True):
        """MultiHeadAttWrapper()
        
        Args
        ----
          feat_dim:     int, feat_dimension
          num_head:     int, number of heads
          attn_dropout: float, attention dropout in Multihead attention

          flag_cat_q: bool, if true, concate(query, attention's output)
          flag_causal: bool, causal dependency in self-attention
          feat_dropout: dropout on feature after self-attention
        """
        super(MultiHeadAttWrapper, self).__init__(
            feat_dim_k, feat_dim_v, num_head, attn_dropout, flag_cat_q)
                
        # output layer norm
        self.flag_layernorm_out = flag_layernorm_out

        if self.flag_layernorm_out:
            self.m_layernorm = torch_nn.LayerNorm(feat_dim_v)

        # use residual connection?
        self.flag_residual = flag_residual

        return
        
    def forward(self, query, key, value, q_mask = None, k_mask = None):
        """O, attn = SelfMultiheadAttention(feat, mask)
        
        input:
        ------
          Q: (batch, lengthQ, feat_dimK)
          K: (batch, lengthK, feat_dimK)
          V: (batch, lengthK, feat_dimV)

    
          k_mask: None or tensor, (batch, lengthQ, lengthK)
          q_mask: None or tensor, (batch, lengthQ)
        
        output
        ------
          O: tensor, (batch, lengthQ, feat_dim)
          attn: tensor, (batch * head, lengthQ, lengthK), attention matrix
          
        """        
            
        # call multi-head attention
        o_mat, attn = super(MultiHeadAttWrapper, self).forward(
            query, key, value, q_mask, k_mask)

        # residual
        if self.flag_residual:
            o_mat = query + o_mat
        
        # layernorm
        if self.flag_layernorm_out:
            o_mat = self.m_layernorm(o_mat)
        
        return o_mat, attn



# ====================
# misc 
# ====================

def position_encoding(n_pos, n_dim, padding_idx=None):
    """data = position_encoding(n_pos, n_dim, padding_idx)
    Position encoding in Transformer
    
    input: 
    ------
      n_pos: int, pos, number of possible positions
      n_dim: int, n_dim//2 = i, number of hidden dimensions
    
    output:
    ------
      sin_tab: np.array, (n_pos, n_dim)
    
    sin_tab[n, 2i] = sin(n / 10000 ^ (2i / n_dim))
    sin_tab[n, 2i+1] = cos(n / 10000 ^ (2i / n_dim))
    
    Example:
      data = position_encoding(1024, 512, 0)
    """
    # make sure that n_dim is an even number
    if n_dim % 2 > 0:
        print("position_encoding: n_dim should be an even number") 
        sys.exit(1)
        
    # create the table
    sin_tab = np.zeros([n_pos, n_dim])
    for idx in np.arange(n_dim // 2):
        # period: 10000 ^ (2i / n_dim)
        pd = np.power(10000, 2 * idx / n_dim)
        # sin(n / 10000 ^ (2i / n_dim))
        sin_tab[:, 2 * idx] = np.sin( np.arange(n_pos) / pd)
        # cos(n / 10000 ^ ((2i+1) / n_dim))
        sin_tab[:, 2 * idx+1] = np.cos( np.arange(n_pos) / pd)
    
    # remove the dummy positioning encoding
    if padding_idx is not None:
        sin_tab[padding_idx] = 0

    return sin_tab

# ========
# Guided Attention mask
# ========

def GuidedAttentionPenelty(olen, ilen, sigma):
    """p_matrix = GuidedAttentionPenelty(olen, ilen, sigma)
    
    From paper 
    Tachibana, Hideyuki, Katsuya Uenoyama, and Shunsuke Aihara. 
    Efficiently trainable text-to-speech system based on deep 
    convolutional networks with guided attention. Proc. ICASSP 2018.
    https://arxiv.org/pdf/1710.08969.pdf
    
    W_{nt} = 1 − exp{−(n/N − t/T)^2 /(2g^2)}
    
    input
    -----
      ilen:   int, input sequence length
      olen:   int, output sequence length
      sigma:  float, smoothing parameter
      
    output
    ------
      p_matrix:   tensor, penelty matrix, (olen, ilen)
    """
    try:
        grid_o, grid_i = torch.meshgrid(torch.arange(olen), torch.arange(ilen), 
                                        indexing='ij')
    except TypeError:
        grid_o, grid_i = torch.meshgrid(torch.arange(olen), torch.arange(ilen))

    p_matrix = -((grid_o / olen - grid_i / ilen) ** 2) / (2 * (sigma ** 2))
    p_matrix = 1.0 - torch.exp(p_matrix)
    return p_matrix


def GuidedAttentionPeneltyBatch(olens, ilens, sigma=0.4):
    """p_matrices = GuidedAttentionPeneltyBatch(olens, ilens)
    Batch version of GuidedAttentionPenelty
    
    input
    -----
      ilen:   list of int, input sequence length
      olen:   list of int, output sequence length
      sigma:  float, smoothing parameter
      
    output
    ------
      p_matrices: tensor, penelty matrix, (len(olens), max(olens), max(ilens))
      
    Example
    -------
      GuidedAttentionPeneltyBatch([6, 5], [3, 5], 0.4)
      tensor([[[0.0000, 0.2934, 0.7506, 0.0000, 0.0000],
         [0.0831, 0.0831, 0.5422, 0.0000, 0.0000],
         [0.2934, 0.0000, 0.2934, 0.0000, 0.0000],
         [0.5422, 0.0831, 0.0831, 0.0000, 0.0000],
         [0.7506, 0.2934, 0.0000, 0.0000, 0.0000],
         [0.8858, 0.5422, 0.0831, 0.0000, 0.0000]],

        [[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
         [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
         [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
         [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
         [0.8647, 0.6753, 0.3935, 0.1175, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]]])
    """
    assert len(ilens) == len(olens), "len(ilens) != len(olens)"
    num_batch = len(ilens)
    pmats = torch.zeros([num_batch, max(olens), max(ilens)])
    for idx, olen, ilen in zip(range(num_batch), olens, ilens):
        pmats[idx, :olen, :ilen] = GuidedAttentionPenelty(olen, ilen, sigma)
    return pmats



def GuidedAttentionMask(olens, ilens):
    """masks = GuidedAttentionMask(olens, ilens)
    Generating masks. Attention matrix corresponding to padded zero should
    not be computed in the GuidedAttention Loss, then they should be masked.
    Note that, masks in Encoder and Decoder are computed in different ways.
    
    input
    -----
      ilen:   list of int, input sequence length
      olen:   list of int, output sequence length
      sigma:  float, smoothing parameter
      
    output
    ------
      masks: tensor, mask matrices, (len(olens), max(olens), max(ilens))
      
    Example
    -------
      GuidedAttentionMask([6, 5], [3, 5])
      tensor([[[ True,  True,  True, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True, False, False]],

        [[ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True],
         [ True,  True,  True,  True,  True],
         [False, False, False, False, False]]])
    
      It should be used like this:
      loss_all = attention_matrices * GuidedAttentionPeneltyBatch
      loss_mean = torch_mean(loss_all.masked_select(GuidedAttentionMask))
    """
    assert len(ilens) == len(olens), "len(ilens) != len(olens)"
    num_batch = len(ilens)
    
    masks = torch.zeros([num_batch, max(olens), max(ilens)])
    for idx, olen, ilen in zip(range(num_batch), olens, ilens):
        masks[idx, :olen, :ilen] += 1.0

    return masks.gt(0.0)


if __name__ == "__main__":
    print("block_attention.py")
