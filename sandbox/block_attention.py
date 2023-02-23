##!/usr/bin/env python
"""
Blocks for attention mechanism

Implementation is based on https://github.com/soobinseo/Transformer-TTS.git,
but code is re-facotrized:

DotScaledAttention and MultiheadAttention are separated.
The former is the core attention softmax(QK^T/sqrt(d))V, 
with optional mask to mask dummy query and dummy key-value 
that zero-padded due to the varied sequence length in batch

The former further includes the mask due to causal dependency
between output and input

The latter does split-> transform -> DotScaledAtt -> concat -> transform
 
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
    
    k_mask: (batch, length1)
    q_mask: (batch, length1)
    
    Example:
        l_m1 = DotScaledAttention()

        q_data = torch.rand([5, 100, 64])
        k_data2 = torch.rand([5, 40, 64])
        v_data3 = torch.rand([5, 40, 32])

        q_mask = torch.ones([5, 100])
        q_mask[0, -4:] = 0
        q_mask[1, -5:] = 0
        q_mask_bin = q_mask.eq(0)
        k_mask = torch.ones([5, 40])
        k_mask[0, -4:] = 0
        k_mask[1, -5:] = 0
        k_mask_bin = k_mask.eq(0)

        o1, a1 = l_m1(q_data, k_data2, v_data3, q_mask_bin, k_mask_bin)
        
        # causal
        l_m1 = DotScaledAttention(True)
        data = torch.rand([5, 100, 64])
        q_mask = torch.ones([5, 100])
        q_mask_bin = q_mask.eq(0)
        o1, a1 = l_m1(data, data, data, q_mask_bin, q_mask_bin)
        o1[0, 1] - a1[0, 1, 0] * data[0, 0] - a1[0, 1, 1] * data[0, 1]
    """
    def __init__(self, flag_causal=False, dropout=None):
        super(DotScaledAttention, self).__init__()
        self.flag_causal = flag_causal
        if dropout is not None:
            self.m_drop = torch_nn.Dropout(p=dropout)
        else:
            self.m_drop = None
        return
    
    def forward(self, Q, K, V, q_mask=None, k_mask=None):
        """O = DotScaledAttention(Q, K, V, q_mask=None, k_mask=None)
        O = q_mask * softmax( (Q K^\top / \sqrt(d)) k_mask ) V
        
        input:
        ------
          Q: tensor, (batch, length1, dimension)
          K: tensor, (batch, length2, dimension)
          V: tensor, (batch, length2, dimension2)
    
          k_mask: None or tensor, (batch, length2)
          q_mask: None or tensor, (batch, length1)
        
        output
        ------
          O: tensor, (batch, length1, dimension2)
          attn: tensor, (batch, length1, length2), attention matrix
          
        k_mask[i] is a mask for the i-th key/value, k_mask[i, j]==True
          indicates that K[i][j] and V[i][j] should be masked. 
          attention[i][:, j] should be zero
        q_mask[i] is a mask for the i-the query, q_mask[i, j]==True
          indicates that output O[i][j] should be masked
        """
        bsize = Q.shape[0]
        feat_dim = Q.shape[-1]
        q_len = Q.shape[1]
        k_len = K.shape[1]
        
        # Q K^\top
        # attn has shape (length1, length2)
        attn = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(feat_dim)
        
        # apply k_mask to mask dummy key/value (by setting attn to 0)
        if k_mask is not None:
            # (batch, length2) -> (batch, length1, length2) by duplicating
            mask_tmp = k_mask.unsqueeze(1).repeat(1, q_len, 1)
            
            # if causal dependency, add diagonal mask
            #  mask_tmp[:, i, >i] should be True
            if self.flag_causal and q_len == k_len:
                # length2 must be == length1
                # create upper triagle (length1, length1)
                tria_tmp = torch.triu(torch.ones_like(mask_tmp[0]), diagonal=1)
                # repeat to batch
                tria_tmp = tria_tmp.unsqueeze(0).repeat(bsize, 1, 1).gt(0)
                # overlap the upper-triangle matrix with the k_mask
                mask_tmp = torch.bitwise_or(mask_tmp, tria_tmp)
            
        elif self.flag_causal and q_len == k_len:
            # even if no need to mask dummy input, it is necessary to
            # mask for causal self-attention
            mask_tmp = torch.triu(torch.ones([k_len, k_len]), diagonal=1)
            # repeat to batch
            mask_tmp = mask_tmp.unsqueeze(0).repeat(bsize, 1, 1).gt(0)
            mask_tmp = mask_tmp.to(device=Q.device)
        else:
            # no k_mask provided, neither is k_mask provided
            mask_tmp = None
        
        
        # mask the attn matrix if necessary
        if mask_tmp != None:
            attn = attn.masked_fill(mask_tmp, -2 ** 32 +1)
        
            
        # softmax, over length2 of the (batch, length1, length2)
        attn = torch_nn_func.softmax(attn, dim=-1)
        
        # apply q_mask
        if q_mask is not None:
            # (batch, length1, 1) -> (batch, length1, length2)
            mask_tmp = q_mask.unsqueeze(-1).repeat(1, 1, k_len)
            # mask query (row) that should be dummy
            attn = attn.masked_fill(mask_tmp, 0)
        
        # apply dropout is necessary
        if self.m_drop is not None:
            attn = self.m_drop(attn)

        # o = attn * V
        O = torch.bmm(attn, V)
        return O, attn




class MultiheadAttention(torch_nn.Module):
    """Multihead Attention in Transformer
    
    V, K, Q -> linear -> split -> DotScaledAttention -> concate -> linear
    
    Q: (batch, lengthQ, feat_dimK)
    K: (batch, lengthK, feat_dimK)
    V: (batch, lengthK, feat_dimV)
    
    k_mask: (batch, lengthK)
    q_mask: (batch, lengthQ)
    
    Example:
      q_data = torch.rand([5, 100, 64])
      k_data2 = torch.rand([5, 40, 64])
      v_data3 = torch.rand([5, 40, 32])

      q_mask = torch.ones([5, 100])
      q_mask[0, -4:] = 0
      q_mask[1, -5:] = 0
      q_mask_bin = q_mask.eq(0)
      k_mask = torch.ones([5, 40])
      k_mask[0, -4:] = 0
      k_mask[1, -5:] = 0
      k_mask_bin = k_mask.eq(0)
      
      l_m = MultiheadAttention(64, 32, 4)
      
      data_out = l_m.forward(v_data3, k_data2, q_data, k_mask_bin, q_mask_bin)
    """
    def __init__(self, feat_dim_k, feat_dim_v, num_head=4, 
                 flag_cat_q=True, flag_causal=False, dropout=None,
                 with_bias=False, flag_norm_before=False):
        """MultiheadAttention(num_head=4, flag_cat_q=True)
        
        Args
        ----
          feat_dim_k: int, feat_dimension of Query and Key
          feat_dim_v: int, feat_dimension of Value
          num_head: int, number of heads
          flag_cat_q: bool, if true, concate(query, attention's output)
          flag_causal: bool, causal dependency in self-attention
          with_bias: bool, bias in feedforward layer for multi-head splitting?
                     (default False)
          dropout: float or None, dropout rate on attention matrix
                     (default None)
          flag_norm_before: bool, whether do layer normalize before attention
                     (default False). If true, the input q, k, and v should
                     be layerer normed before given to forward()

        When flag_causal is True, Q, K, V must have same temporal length
        """
        super(MultiheadAttention, self).__init__()
        
        # log information
        self.flag_causal = flag_causal
        self.num_head = num_head
        
        # feedforward layers
        if feat_dim_k % self.num_head > 0 or feat_dim_v % self.num_head > 0:
            print("feat_dim_k cannot be divided by num_head")
            sys.exit(1)
        
        self.m_q_fc = torch_nn.Linear(feat_dim_k, feat_dim_k, bias=with_bias)
        self.m_k_fc = torch_nn.Linear(feat_dim_k, feat_dim_k, bias=with_bias)
        self.m_v_fc = torch_nn.Linear(feat_dim_v, feat_dim_v, bias=with_bias)
        
        torch_nn.init.xavier_uniform_(
            self.m_q_fc.weight, gain=torch_nn.init.calculate_gain('linear'))
        torch_nn.init.xavier_uniform_(
            self.m_k_fc.weight, gain=torch_nn.init.calculate_gain('linear'))
        torch_nn.init.xavier_uniform_(
            self.m_v_fc.weight, gain=torch_nn.init.calculate_gain('linear'))
        
        # core attention 
        self.m_attn = DotScaledAttention(self.flag_causal, dropout)
        
        # dropout
        if dropout is not None:
            self.m_drop = torch_nn.Dropout(p=dropout)
        else:
            self.m_drop = None
        
        # output linear layer
        self.flag_cat_q = flag_cat_q
        if self.flag_cat_q:
            self.m_output = torch_nn.Linear(feat_dim_k+feat_dim_v, feat_dim_v)
        else:
            self.m_output = torch_nn.Linear(feat_dim_v, feat_dim_v)
        torch_nn.init.xavier_uniform_(
            self.m_output.weight, gain=torch_nn.init.calculate_gain('linear'))
        
        # 
        self.m_layernorm = torch_nn.LayerNorm(feat_dim_v)
        self.flag_norm_before = flag_norm_before

        if feat_dim_k != feat_dim_v:
            print("Warning: query/key and value differ in feature dimensions.")
            print("Residual connection will not be used")
        
        
        return
        
    def forward(self, value, key, query, k_mask=None, q_mask=None):
        """O, attn = MultiheadAttention(value, key, query, k_mask, q_mask)
        
        input:
        ------
          Q: (batch, lengthQ, feat_dimK)
          K: (batch, lengthK, feat_dimK)
          V: (batch, lengthK, feat_dimV)

    
          k_mask: None or tensor, (batch, length2)
          q_mask: None or tensor, (batch, length1)
        
        output
        ------
          O: tensor, (batch, length1, dimension2)
          attn: tensor, (batch, length1, length2), attention matrix
          
        k_mask[i] is a mask for the i-th key/value, k_mask[i, j]==True
          indicates that K[i][j] and V[i][j] should be masked. 
          attention[i][:, j] should be zero
        q_mask[i] is a mask for the i-the query, q_mask[i, j]==True
          indicates that output O[i][j] should be masked
        """
        bsize = value.size(0)
        k_len = key.size(1)
        q_len = query.size(1)
        
        if self.flag_causal and k_len != q_len:
            print("Causal Attention, Q,V,K must have same length in time")
            sys.exit(1)
            
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
        if q_mask is not None:
            q_mask_tmp = q_mask.repeat(self.num_head, 1)
        else:
            q_mask_tmp = None

        if k_mask is not None:
            k_mask_tmp = k_mask.repeat(self.num_head, 1)
        else:
            k_mask_tmp = None

        
        # attention and sum
        o_mul, attn = self.m_attn(query_mul, key_mul, value_mul, 
                                  q_mask_tmp, k_mask_tmp)
        
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
        
        # linear
        o_mat = self.m_output(o_mat)
        
        # dropout
        if self.m_drop:
            o_mat = self.m_drop(o_mat)

        # residual & layer norm
        if o_mat.shape[-1] == query.shape[-1]:
            o_mat = o_mat + query

        # layer normalize after 
        if not self.flag_norm_before:
            o_mat = self.m_layernorm(o_mat)

        return o_mat, attn


# ====================
# misc 
# ====================

def position_encoding(n_pos, n_dim, padding_idx=None):
    """Position encoding in Transformer
    
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

class FeedforwardBlock(torch_nn.Module):
    """Feedforward block in Transformer 
    """
    def __init__(self, feat_dim):
        super(FeedforwardBlock, self).__init__()
        
        self.m_block = torch_nn.Sequential(
            torch_nn.Linear(feat_dim, feat_dim * 4),
            torch_nn.ReLU(),
            torch_nn.Linear(feat_dim * 4, feat_dim)
            #torch_nn.Dropout(p=0.1)
        )
        self.m_layernorm = torch_nn.LayerNorm(feat_dim)
        
        # initialization
        torch_nn.init.xavier_uniform_(
            self.m_block[0].weight, gain=torch_nn.init.calculate_gain('relu'))
        torch_nn.init.xavier_uniform_(
            self.m_block[2].weight, gain=torch_nn.init.calculate_gain('linear'))
        return

    def forward(self, feat):
        """ out = FeedforwardBlock(feat)
        
        input
        -----
          feat: tensor, (batch, length, feat_dim)
          
        output
        ------
          output: tensor, (batch, length, feat_dim)
        """
        return self.m_layernorm(self.m_block(feat) + feat)


class FeedforwardBlockv2(torch_nn.Module):
    """Feedforward block in Transformer 
    """
    def __init__(self, feat_dim, dropout=0.0, flag_norm_before=False):
        super(FeedforwardBlockv2, self).__init__()
        
        self.m_block = torch_nn.Sequential(
            torch_nn.Linear(feat_dim, feat_dim * 4),
            torch_nn.ReLU(),
            torch_nn.Dropout(p=dropout),
            torch_nn.Linear(feat_dim * 4, feat_dim)
        )
        self.flag_norm_before = flag_norm_before
        self.m_layernorm = torch_nn.LayerNorm(feat_dim)
        
        # initialization
        torch_nn.init.xavier_uniform_(
            self.m_block[0].weight, gain=torch_nn.init.calculate_gain('relu'))
        torch_nn.init.xavier_uniform_(
            self.m_block[-1].weight, gain=torch_nn.init.calculate_gain('linear'))
        return

    def forward(self, feat):
        """ out = FeedforwardBlock(feat)
        
        input
        -----
          feat: tensor, (batch, length, feat_dim)
          
        output
        ------
          output: tensor, (batch, length, feat_dim)
        """
        if not self.flag_norm_before:
            return self.m_layernorm(self.m_block(feat) + feat)
        else:
            return self.m_block(feat) + feat


if __name__ == "__main__":
    print("block_attention.py")
