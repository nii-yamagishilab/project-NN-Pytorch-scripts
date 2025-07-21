#!/usr/bin/env python
"""
Transformer implementation

Based on https://github.com/soobinseo/Transformer-TTS.git
1. Components are re-facotrized
   DotScaledAttention and MultiheadAttention are separated
     The former is the core attention softmax(QK^T/sqrt(d))V, 
     The latter does split-> transform -> DotScaledAtt -> concat -> transform

   Mask for attention between query and key (k_mask) and 
   mask for output (q_mask) should be given as input to the two
   APIs 
   
2. stop-token is predicted. 
   This requires the input to be padded with some dummy values that
   indicate ending state. 

   See ModelExample for details
 
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
from logging import getLogger

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

from submodules import attention
from submodules import spk_emb

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"


logger = getLogger(__name__)

class EncoderPrenet(torch_nn.Module):
    """PreNet of Encoder    
    """
    def __init__(self, num_symbol, emb_dim, out_dim, enc_dp=0.2):
        """
        Args
        ----
          num_symbol: int, number of unique symbols that will be input
          emb_dim:    int, dimension of the input symbol embedding
          out_dim:    int, dimension of the output of the EncoderPreNet
          enc_dp:     float, dropout rate, default 0.2
        
        Example
        -------
          l_enc = EncoderPrenet(10, 32, 64)
          data1 = torch.randint(0, 10, [5, 10])
          dataout = l_enc(data1)

        """
        super(EncoderPrenet, self).__init__()
        
        # embedding layer
        if num_symbol < 1:
            # the input to EncoderPreNet will be embeddings
            self.m_emb = None
        else:
            self.m_emb = torch_nn.Embedding(num_symbol, emb_dim, padding_idx=0)
        
        self.m_emb_dim = emb_dim

        # convolution layers
        self.m_block = torch_nn.Sequential(
            torch_nn.Conv1d(emb_dim, out_dim, kernel_size=5, padding=2),
            torch_nn.BatchNorm1d(out_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(p=enc_dp),
            torch_nn.Conv1d(out_dim, out_dim, kernel_size=5, padding=2),
            torch_nn.BatchNorm1d(out_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(p=enc_dp),
            torch_nn.Conv1d(out_dim, out_dim, kernel_size=5, padding=2),
            torch_nn.BatchNorm1d(out_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(p=enc_dp),
        )
        self.m_linear = torch_nn.Linear(out_dim, out_dim)
        
        # weight initialization on conv
        for idx in [0, 4, 8]:
            torch_nn.init.xavier_uniform_(
                self.m_block[idx].weight, 
                gain=torch_nn.init.calculate_gain('relu'))

        # weight initialization on linear
        torch_nn.init.xavier_uniform_(
            self.m_linear.weight, 
            gain=torch_nn.init.calculate_gain('linear'))
        
        return

    def _forward_char_input(self, char_index):
        """
        input
        -----
          char: tensor, type.long, (batch, length)
        
        output
        ------
          output: tensor, (batch, length, feat_dim)
        """
        emb = self.m_emb(char_index)
        
        hid = self.m_block(emb.transpose(1, 2)).transpose(1, 2)
        return self.m_linear(hid)

    def _forward_emb_input(self, input_emb):
        """ 
        input
        -----
          input_emb: tensor, float, (batch, length, emb_dim)
        
        output
        ------
          output: tensor, (batch, length, feat_dim)
        """
        assert input_emb.shape[-1] == self.m_emb_dim, 'Input embed dim error'

        hid = self.m_block(input_emb.transpose(1, 2)).transpose(1, 2)
        return self.m_linear(hid)
    
    def forward(self, input_x):
        """ output = Encoder(input_x)
        
        input
        -----
          input_x: 
            if self.m_emb is not None
              tensor, type.long, (batch, length)
            else
              tensor, float, (batch, length, emb_dim)
        
        output
        ------
          output: tensor, (batch, length, feat_dim)
        """
        if self.m_emb is not None:
            return self._forward_char_input(input_x)
        else:
            return self._forward_emb_input(input_x)
    

class FeedforwardBlock(torch_nn.Module):
    """Feedforward block in Transformer 
    """
    def __init__(self, feat_dim, flag_layernorm_before = False):
        """
        Args
        ----
          feat_dim:               int, dimension of the input and output feature
          flag_layernorm_before:  bool, whether use layernorm before input 
                                  or after output, default False
        """
        super(FeedforwardBlock, self).__init__()
        
        self.m_block = torch_nn.Sequential(
            torch_nn.Linear(feat_dim, feat_dim * 4),
            torch_nn.ReLU(),
            torch_nn.Linear(feat_dim * 4, feat_dim),
        )
        self.m_lnorm = torch_nn.LayerNorm(feat_dim)
        self.flag_layernorm_before = flag_layernorm_before

        # initialization
        torch_nn.init.xavier_uniform_(
            self.m_block[0].weight, 
            gain=torch_nn.init.calculate_gain('relu'))

        torch_nn.init.xavier_uniform_(
            self.m_block[-1].weight, 
            gain=torch_nn.init.calculate_gain('linear'))

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
        # layernorm
        feat_ = self.m_lnorm(feat) if self.flag_layernorm_before else feat
        
        # process and residual
        feat_ = self.m_block(feat_) + feat
        
        # layernorm
        out = self.m_lnorm(feat_) if not self.flag_layernorm_before else feat_
        return out

    
class PostNet(torch_nn.Module):
    """Postnet added for Transformer TTS
    """
    def __init__(self, feat_dim, hid_dim, postnet_dp=0.5, num_block=4):
        """PostNet(feat_dim, hid_dim)
        Args:
        -----
          feat_dim:   int, input feature dimension (batch, length, feat_dim)
          hid_dim:    int, dimension of hidden features inside PostNet
          postnet_dp: float, dropout rate, default 0.5
          num_block:  int, number of conv-bn-tanh-dropout block
        """
        super(PostNet, self).__init__()
        # padding length for convolution
        self.pad = 4

        # blocks
        self.m_block_list = []
        for idx in range(num_block):
            in_dim = feat_dim if idx == 0 else hid_dim
            
            self.m_block_list.append(
                torch_nn.Sequential(
                    torch_nn.Conv1d(in_dim, hid_dim, 
                                    kernel_size=5, padding=self.pad),
                    torch_nn.BatchNorm1d(hid_dim),
                    torch_nn.Tanh(),
                    torch_nn.Dropout(p=postnet_dp)))
            
            # weight initialization 
            torch_nn.init.xavier_uniform_(
                self.m_block_list[-1][0].weight, 
                gain=torch_nn.init.calculate_gain('tanh'))

        self.m_block_list = torch_nn.ModuleList(self.m_block_list)

        # output
        self.m_output = torch_nn.Conv1d(hid_dim, feat_dim, 
                                        kernel_size=5, padding=self.pad)
        torch_nn.init.xavier_uniform_(
            self.m_output.weight, 
            gain=torch_nn.init.calculate_gain('linear'))
        
        return
    
    def forward(self, x):
        x = x.transpose(1, 2)
        # to make sure that the transformation is causal in time domain
        # see https://github.com/pytorch/pytorch/issues/1333
        for block in self.m_block_list:
            x = block(x)[:, :, :-self.pad]
        out = self.m_output(x)[:, :, :-self.pad]
        return out.transpose(1, 2)
    
    
class DecoderPrenet(torch_nn.Module):
    """Decoder prenet of transformer
    """
    def __init__(self, input_dim, hid_dim, out_dim, prenet_dp=0.5):
        """DecoderPrenet(input_dim, hid_dim, out_dim)
        Args:
        -----
          input_dim: int, input feature dimension (batch, length, input_dim)
          hid_dim:   int, hidden feature dimension
          out_dim:   int, output feature dimension (batch, length, output_dim)
          prenet_dp: float, dropout rate of DecoderPreNet, default 0.5
        
        """
        super(DecoderPrenet, self).__init__()

        self.m_block = torch_nn.Sequential(
            torch_nn.Linear(input_dim, hid_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(p=prenet_dp),
            torch_nn.Linear(hid_dim, out_dim),
            torch_nn.ReLU(),
            torch_nn.Dropout(p=prenet_dp)
        )
        for idx in [0, 3]:
            torch_nn.init.xavier_uniform_(
                self.m_block[idx].weight, 
                gain=torch_nn.init.calculate_gain('relu'))
        return
    
    def forward(self, x):
        return self.m_block(x)


class Encoder(torch_nn.Module):
    """Encoder of Transformer
    
    Example:
      l_encoder = Encoder(10, 0, 1024, 32, 32)
      data1 = torch.randint(0, 9, [5, 10])+1
      data1[0, -5:] = 0
      data1[1, -3:] = 0
      out, mask, attn = l_encoder(data1)
      print(attn[0][0])
    """
    def __init__(self, 
                 num_symbol, dummy_symbol, 
                 max_length, emb_dim, out_dim, spk_dim,
                 num_head = 4, block_num = 6,
                 prenet_dp = 0.2,
                 preattn_dp = 0.1, 
                 attn_dp = 0.1):
        """
        Args
        ----
          num_symbol:   int, number of input unique symbols
          dummy_symbol: str, symbol that indicates empty input
          max_length:   int, maximum length of input sequence
          emb_dim:      int, number of dimensions of input embedding
          out_dim:      int, number of dimensions of output feature
          num_head:     int, number of heads in multi-head attention, default 4
          block_num:    int, number of self-attention blocks, default 6
          prenet_dp:    float, dropout rate of prenet, default 0.2
          preattn_dp:   float, dropout rate before first att. block, default 0.1
          attn_dp:      float, dropout rate on self-att. matrix, default 0.1
        """
        super(Encoder, self).__init__()

        # 
        self.num_head = num_head
        self.spk_dim = spk_dim
        
        # weight for summing position codes and embeddings
        self.alpha = torch_nn.Parameter(torch.ones(1))
        
        # position code
        self.m_pos_emb = torch_nn.Embedding.from_pretrained(
            torch.tensor(attention.position_encoding(max_length, out_dim, padding_idx=0),
                         dtype=torch.float32),
            freeze=True)

        # prenet
        self.m_prenet = EncoderPrenet(num_symbol, emb_dim, out_dim, prenet_dp)

        # spk
        if spk_dim > 0:
            self.m_spk_merger = spk_emb.SpkEmbMerger(spk_dim, out_dim, out_dim)
        else:
            self.m_spk_merger = None

        
        # 
        self.m_dummy_symbol = dummy_symbol
        self.m_num_symbol = num_symbol

        # dropout before the first attention block
        self.m_dropout1 = torch_nn.Dropout(p=preattn_dp)
        
        # attention blocks (with dropout)
        attn_bag = []
        ffn_bag = []
        for idx in range(block_num):
            attn_bag.append(attention.MultiHeadAttWrapper(
                out_dim, out_dim, num_head, attn_dp))
            ffn_bag.append(FeedforwardBlock(out_dim))
        self.m_attns = torch_nn.ModuleList(attn_bag)
        self.m_ffns = torch_nn.ModuleList(ffn_bag)

        return

    def _create_mask(self, x):
        """ q_mask, k_mask = _create_mask(x)
        create the query and attention mask given input x

        input
        -----
          x:  tensor, input symbol sequence, 
                      (batch, length), or (batch, length, dim)
        output
        ------
          q_mask: tensor, mask for query, (batch, length)
          k_mask: tensor, mask for attention, (batch, length, length)
        """
        # if there is only one data in the batch, no need to mask
        if x.shape[0] == 1:
            q_mask, k_mask = None, None
        else:
            if self.m_num_symbol > 0:
                # the usual case where x is a string of characters
                bsize, maxlen = x.shape
                # mask if input x is equal to the special dummy symbol
                mask = x == self.m_dummy_symbol
            else:
                # unusual case where x is already an embedding sequence
                bsize, maxlen, _ = x.shape
                mask = x.sum(dim=2) == self.m_dummy_symbol

            q_mask = mask
            k_mask = mask.unsqueeze(1).repeat(1, x.size(1), 1)
        return q_mask, k_mask

    def _create_position_mat(self, x, mask=None):
        """pos_mat = _create_position_mat(x, mask)
        
        input
        -----
          x: tensor, input symbol sequence
        
        output
        ------
          pos_mat: position matrix
        """
        bsize, maxlen = x.shape
        # create position matrix
        pos_mat = (torch.arange(maxlen) + 1).unsqueeze(0).repeat(bsize, 1)
        pos_mat = pos_mat.to(dtype=torch.int64, device=x.device)
        
        if mask is not None:
            pos_mat = pos_mat.masked_fill_(mask, 0)
        return pos_mat


    def forward(self, x, spk_emb):
        """
        """
        # create mask and position index matrix
        q_mask, k_mask = self._create_mask(x)
        pos_mat = self._create_position_mat(x, q_mask)
            
        # prenet and position encoding
        x_ = self.m_prenet(x)
        x_ = x_ + self.alpha * self.m_pos_emb(pos_mat)

        # dropout
        x_ = self.m_dropout1(x_)
        
        attn_list = []
        # attention and transformation
        for l_attn, l_ffn in zip(self.m_attns, self.m_ffns):
            x_, att = l_attn(x_.clone(), x_.clone(), x_.clone(), q_mask, k_mask)
            x_ = l_ffn(x_)
            attn_list.append(att)

        # spk embedding (if available)
        if self.m_spk_merger is None:
            pass
        else:
            x_ = self.m_spk_merger(spk_emb, x_, [x_.shape[1] for y in x_])
            
        return x_, q_mask, attn_list
    

class Decoder(torch_nn.Module):
    """Decoder of Transformer
    
    Example:
        l_encoder = Encoder(10, 0, 1024, 32, 32)
        data1 = torch.randint(0, 9, [5, 10])+1
        data1[0, -5:] = 0
        data1[1, -3:] = 0
        enc_out, mask, attn = l_encoder(data1)
        #print(attn[0][0])

        l_decoder = Decoder(1024, 64, 1, 32)
        target = torch.rand([5, 20, 64])
        data_length = [20] * 5

        out, out_new, stop, sa_attns, da_attns = l_decoder(
             enc_out, target, mask, data_length)
    """
    def __init__(self, 
                 max_length, target_dim, reduction_factor, hid_dim,
                 spk_dim = 0,
                 num_head = 4,
                 num_block = 6, 
                 prenet_dp = 0.5,
                 preattn_dp = 0.1,
                 attn_dp = 0.1,
                 postnet_dp = 0.5,
                 num_block_postnet = 4):
        """
        Args
        ----
          max_length:       int, maximum length input sequence
          target_dim:       int, dimension of output feature
          reduction_factor: int, reduction factor (usually 1)
          hid_dim:          int, dimension of hidden feat. from encoder
          spk_dim:          int, dimension of speaker embedding, default 0
          num_head:         int, number of heads in attention, default 4
          num_block:        int, number of self-att&cross.att blocks, default 6
          prenet_dp:        float, dropout rate of Decoder PreNet, default 0.5
          preattn_dp:       float, dropout rate before attention, default 0.1
          attn_dp:          float, dropout rate on attn. mat, default 0.1
          postnet_dp:       float, dropout rate in Decodoer PostNet, default 0.5
          num_block_postnet: int, number of conv blocks in postnet, default 4
        """
        super(Decoder, self).__init__()
        
        # number of head
        self.num_head = num_head

        # weight for summing position codes and embeddings
        self.alpha = torch_nn.Parameter(torch.ones(1))
        
        # position code
        self.m_pos_emb = torch_nn.Embedding.from_pretrained(
            torch.tensor(attention.position_encoding(max_length, hid_dim, padding_idx=0),
                         dtype=torch.float32),
            freeze=True)
        
        # dropout before first attention block
        self.m_dropout1 = torch_nn.Dropout(p=preattn_dp)
        
        # prenet (use x2 as hidden feature dim in prenet)
        if spk_dim > 0:
            self.m_spk_merger = spk_emb.SpkEmbMerger(spk_dim, target_dim, target_dim)
        else:
            self.m_spk_merger = None
        
        self.m_prenet = DecoderPrenet(target_dim, hid_dim*2, hid_dim, prenet_dp)
        
        self.m_linear = torch_nn.Linear(hid_dim, hid_dim)
        torch_nn.init.xavier_uniform_(
            self.m_linear.weight, gain=torch_nn.init.calculate_gain('linear'))
        
        
        # self-attention & attention blocks
        l_selfatt = []
        l_dotatt = []
        l_ffn = []
        for idx in range(num_block):
            # self attention
            l_selfatt.append(
                attention.MultiHeadAttWrapper(
                    hid_dim, hid_dim, num_head, attn_dp))
            l_dotatt.append(
                attention.MultiHeadAttWrapper(
                    hid_dim, hid_dim, num_head, attn_dp))
            l_ffn.append(FeedforwardBlock(hid_dim))
        
        self.m_selfatts = torch_nn.ModuleList(l_selfatt)
        self.m_dotatts = torch_nn.ModuleList(l_dotatt)
        self.m_ffns = torch_nn.ModuleList(l_ffn)
            
        # output transformation
        self.m_tar = torch_nn.Linear(hid_dim, target_dim * reduction_factor)
        torch_nn.init.xavier_uniform_(
            self.m_tar.weight, gain=torch_nn.init.calculate_gain('linear'))
                
        # post net
        self.m_postnet = PostNet(target_dim * reduction_factor, hid_dim,
                                 postnet_dp, num_block_postnet)

        # a log for the speaker embedding
        self.spk_dim = spk_dim
        return

    def _create_mask(self, fb_input, fb_pos_mat, cond_mask):
        """ self_qm, self_km, cross_qm, cross_km = _create_mask(
               cond_feat, fb_input, cond_mask)
        
        input
        -----
          fb_input:   tensor, (batch, len_mel, dim), feedback feature 
          fb_pos_mat: tensor, (batch, len_mel), output from _create_position_mat
          cond_mask:  tensor, (batch, len_cond), query_mask on encoder output
        """
        
        bsize, dec_len, _ = fb_input.shape

        if bsize > 1:
            # create mask for decoder feature
            cross_km = cond_mask.unsqueeze(1).repeat(1, dec_len, 1)
            cross_qm = fb_pos_mat.eq(0)

            self_km = fb_pos_mat.eq(0).unsqueeze(1).repeat(1, dec_len, 1)
            self_qm = fb_pos_mat.eq(0)
        else:
            cross_km = None
            cross_qm = None
            self_km = None
            self_qm = None
    
        # apply causal to the self attention mask
            
        # triangle matrix with diagonal line to zero
        c_mask = torch.triu(torch.ones([dec_len, dec_len]), diagonal=1)
        # repeat to batch
        c_mask = c_mask.unsqueeze(0).repeat(bsize, 1, 1)
        c_mask = c_mask.to(device = fb_input.device, dtype = fb_input.dtype)
            
        if self_km is not None:
            self_km = torch.bitwise_or(self_km, c_mask.gt(0))
        else:
            self_km = c_mask.gt(0)

        return self_qm, self_km, cross_qm, cross_km
    

    def _create_position_mat(self, x, data_length):
        """pos_mat = _create_position_mat(x, mask)
        
        input: x, tensor, input mel data, (batch, lenth, dim)
        output: pos_mat: tensor, position matrix, (batch, length)
        """
        bsize, maxlen, dim = x.shape
        # create position matrix
        pos_mat = (torch.arange(maxlen) + 1).unsqueeze(0).repeat(bsize, 1)
        pos_mat = pos_mat.to(dtype=torch.int64, device=x.device)

        # mask padded pat
        for idx in range(bsize):
            pos_mat[idx, data_length[idx]:] = 0
            
        return pos_mat

    def forward(self, cond_feat, fb_input, cond_mask, data_length, spk_emb):
        """
        """
        
        # position index matrix
        pos_mat = self._create_position_mat(fb_input, data_length)
              
        # create masks
        self_qm, self_km, cross_qm, cross_km = self._create_mask(
            fb_input, pos_mat, cond_mask)
        
        # spk embedding (if available)
        if self.m_spk_merger is None:
            dec_input = fb_input
        else:
            dec_input = self.m_spk_merger(spk_emb, fb_input, data_length)
            
        # decoder prenet and positioning encoding
        x_ = self.m_linear(self.m_prenet(dec_input))
        
        # position encoding
        x_ = x_ + self.alpha * self.m_pos_emb(pos_mat)

        # dropout before attention blocks
        x_ = self.m_dropout1(x_)
        
        # self-attention and dot-attention
        sa_attns = []
        da_attns = []
        for sa, da, ffn in zip(self.m_selfatts, self.m_dotatts, self.m_ffns):
            # x_: (batch, length, dim)
            x_, sa_at = sa(x_.clone(), x_.clone(), x_.clone(), self_qm, self_km)
            x_, da_at = da(x_, cond_feat, cond_feat, cross_qm, cross_km)
            x_ = ffn(x_)
            sa_attns.append(sa_at)
            da_attns.append(da_at)
        

        # for stop token
        ffn_output = x_
        
        # for generated output
        target = self.m_tar(ffn_output)
        
        # post net
        target_post = target + self.m_postnet(target)
                
        return target, target_post, sa_attns, da_attns, ffn_output
        

class StopTokenNet(torch_nn.Module):
    """StopTokenNet
    
    A module to predict stop token    
    """
    def __init__(self, hid_dim):
        """
        Args
        ----
          hid_dim: int, number of input feature dimensions 
        
        """
        super(StopTokenNet, self).__init__()

        # padding length for conv
        self.pad = 4

        # simple conv modules
        self.m_l0 = torch_nn.LSTM(hid_dim, hid_dim, batch_first=True)
        self.m_l1 = torch_nn.Conv1d(hid_dim, 1, kernel_size=5, padding=self.pad)
        self.m_l1act = torch_nn.Sigmoid()
        self.m_l2 = torch_nn.Conv1d(1, 1, kernel_size=5, padding=self.pad)
        self.m_l2act = torch_nn.Sigmoid()

        
        torch_nn.init.xavier_uniform_(
            self.m_l1.weight, gain=torch_nn.init.calculate_gain('relu'))
        torch_nn.init.xavier_uniform_(
            self.m_l2.weight, gain=torch_nn.init.calculate_gain('linear'))

        return
    
    def forward(self, x):
        """ stop_token = forward(x)
        
        input
        -----
          x: tensor, (batch, length2, dim), decoder's output sequence
        
        output
        ------
          stop_token: tensor, (batch, length2, 1)
        """
        # use causal convolution
        # see https://github.com/pytorch/pytorch/issues/1333
        x, _ = self.m_l0(x)
        x = self.m_l1act(self.m_l1(x.transpose(1, 2))[:, :, :-self.pad])
        x = self.m_l2act(self.m_l2(x)[:, :, :-self.pad])
        return x.transpose(1, 2)


class TransformerTTS(torch_nn.Module):
    """TransformerTTS
    
    For example definition and usage of TransformerTTS, please check
    the class called TransformerWrapperExample. 
    
    Note that the input and feedback data for the forward() method
    should be prepared carefully. See comment in TransformerWrapperExample

    Args
    ----
      num_symbol:   int, the total number of all possible unique input symbols
                    or the size of the dictionary of the input symbols.
      dummy_symbol: int, among the num_symbol symbols, which one is used as 
                    the dummy value that indicates void input?
                    For example, input symbols in one minibatch may be 
                    [[1, 4, 1, 2, ..., 0, 0, 0],
                     [3, 4, 1, 2, ..., 10, 2, 9],
                     [1, 5, 7, 9, ..., 10, 2, 0]]
                    Because varied seq lengths in one mini-batch, we need to pad
                    the short sequences with a dummy value.
                    The 0 here is the padded dummy value. dummy_symbol = 0
      max_length:   int, maximum length of output sequences during inference
      emb_dim:      int, dimension of the embedding vector in Encoder 
      cond_dim:     int, dimension of the output feature from Encoder
      out_dim:      int, dimension of the target feature. 
                    Target feature has shape (batch, length, out_dim)

      enc_num_head: int, number of att. heads in encoder
      enc_block_num: int, number of att. blocks in encoder
      enc_prenet_dp:    float, dropout rate of prenet, default 0.2
      enc_preattn_dp:   float, dropout rate before first att. block, default 0.1
      enc_attn_dp:      float, dropout rate on self-att. matrix, default 0.1

      dec_num_head:     int, number of heads in attention, default 4
      dec_num_block:    int, number of self-att&cross.att blocks, default 6
      dec_prenet_dp:    float, dropout rate of Decoder PreNet, default 0.5
      dec_preattn_dp:   float, dropout rate before attention, default 0.1
      dec_attn_dp:      float, dropout rate on attn. mat, default 0.1
      dec_postnet_dp:   float, dropout rate in Decodoer PostNet, default 0.5

      reduction_factor, int, reduction_factor (Default: 1)
                        Reduction factor > 1 has not be testified
      training_phase:   int or None
                        None: train enc-dec and stop-token predictor together
                        1: training stage 1, only train enc-dec, not stop-token
                        2: training stage 2, fix enc-dec, only train stop-token
                        (Default: None)
      endframe_len:     int, if stop_token[-endframe_len:].mean() > 0.5, 
                        the inference will stop. (Defaut: 1)

      guided_att_loss_layer: int, number of cross-att layer for guided attn loss
                             default 2.
      guided_att_loss_head:  int, number of cross-att heads for guided attn loss
                             default 2.
                 
      stop_based_on_input_length: bool, stop inference if the generated sequence
           length reaches the length of input sequence (Default False)
    """
    def __init__(self, 
                 num_symbol,
                 dummy_symbol,
                 max_length,
                 emb_dim,
                 cond_dim,
                 out_dim,
                 spk_emb = 0,
                 enc_num_head = 4, 
                 enc_block_num = 6, 
                 enc_prenet_dp = 0.2, 
                 enc_preattn_dp = 0.1, 
                 enc_attn_dp = 0.1,
                 dec_num_head = 4, 
                 dec_block_num = 6,
                 dec_prenet_dp = 0.5, 
                 dec_preattn_dp = 0.1, 
                 dec_attn_dp = 0.1,
                 dec_postnet_dp = 0.5,
                 reduction_factor=1, 
                 train_phase=None,                  
                 endframe_len=1,
                 guided_att_loss_layer = 2,
                 guided_att_loss_head = 2,
                 stop_based_on_input_length = False):
        super(TransformerTTS, self).__init__()

        # configuration
        #num_symbol = config['num_symbol']
        #dummy_symbol = config['dummy_symbol']
        #max_length = config['max_length']
        #emb_dim = config['emb_dim']
        #cond_dim = config['cond_dim']
        #out_dim = csonfig['out_dim']

        #if 'spk_emb' in config and config['spk_emb']['on']:
        #    spk_dim = config['spk_emb']['output_dim']
        #else:
        #    spk_dim = 0
        
        # training phase
        self.train_phase = train_phase
        
        # encoder
        self.l_encoder = Encoder(num_symbol, dummy_symbol, max_length, 
                                 emb_dim, cond_dim, spk_emb, enc_num_head, enc_block_num,
                                 enc_prenet_dp, enc_preattn_dp, enc_attn_dp)
        
        # decoder (hid_dim is same as feature dimension of encoder's output
        self.l_decoder = Decoder(max_length, out_dim, reduction_factor, 
                                 cond_dim, spk_emb, dec_num_head, dec_block_num,
                                 dec_prenet_dp, dec_preattn_dp, dec_attn_dp, 
                                 dec_postnet_dp)
        
        # stop token predictor
        self.l_stop = StopTokenNet(cond_dim)
        
        # other hyper-parameters
        self.maxlen = max_length
        self.tardim = out_dim
        self.rec_fac = reduction_factor
        self.endframe_len = endframe_len

        # loss for mel
        # self.loss_mel_1 = torch_nn.L1Loss()
        # self.loss_mel_2 = torch_nn.MSELoss()

        # loss for stop token
        #self.loss_stop = torch_nn.BCELoss(reduction='none')

        # stop based on input sequence?
        self.stop_based_on_input_length = stop_based_on_input_length

        # number of layers to apply guided attention loss
        self.num_layer_ga_loss = guided_att_loss_layer

        # number of heads to apply guided attention loss
        self.num_heads_ga_loss = guided_att_loss_head

        return

    def return_rec_fac(self):
        return self.rec_fac
    
    def _stop_gen(self, stop_token, input_length):
        """ whether we should stop inference
        
        input
        -----
          stop_token: tensor, (1, N)
          input_length: int, length of the input sequence
        
        output
        ------
          bool, True: stop generation
                False: continue generation
        """
        
        if self.stop_based_on_input_length:
            # stop based on input sequence length
            if stop_token.shape[1] >= input_length:
                return True
            else:
                return False
        else:
            if stop_token.shape[0] > 1:
                logger.info("Cannot do generation for batch size > 1")
                sys.exit(1)

            # stop based on stop token
            if stop_token.shape[1] > self.endframe_len:
                if stop_token[0][-self.endframe_len:].mean() > 0.5:
                    return True
                else: 
                    return False
            else:
                return False
        
    
    def forward(self, input_str, fd_target, data_length, spk_emb):
        """ d_out, d_out_post, stop, d_sa_attns, d_da_attns, e_attns = 
        forward(input_str, target, data_length)
        
        Forward method for training
        
        input
        -----
          input_str: tensor, batch of input character index sequence 
                     (batch, length)
          fd_target: tensor, target fed back to decoder pre-net
                     (batch, length2, dim)
                     fd_target must have been shifted in time
                     This API will directly send fd_target to pre-net
          data_length: list of int, length of target sequence, (batch)
                     Since the actual length of the target sequences in a batch
                     may be different, we need to know its desired length so
                     that dummy values can be masked out
        output
        ------
          d_out: tensor, (batch, length2, dim), generated target sequences
                 before the post-net, i.e., input to the post-net
          d_out_post: tensor, (batch, length2, dim), generated target sequences
                 from the post-net
          stop: tensor, (batch, length2), values for stop-token
          d_sa_attns: list of tensor, each tensor is (batch, length2, length2)
                      list of self-attention matrix in decoder
                      length of list is equal to the number of blocks in decoder
          d_da_attns: list of tensor, each tensor is (batch, length, length2)
                      list of alignment attention matrix in decoder
          e_attns: list of tensor, each tensor is (batch, length, length)
                   list of self-eattention matrix in encoder
        
        Note that length and length2 denote the length of the input and target
        sequences, respectively.
        """
        bsize = input_str.shape[0]
        mlen = fd_target.shape[1]
        dim = fd_target.shape[-1]
        
        # adjust input if reduction factor is on
        if self.rec_fac > 0:
            mlen_tmp = mlen // self.rec_fac
            target_tmp = fd_target[:, :mlen_tmp].view(bsize,-1,dim*self.rec_fac)
        else:
            target_tmp = fd_target        
            
        if self.train_phase is None or self.train_phase == 'freeze_stoptoken':
            # encoding
            e_out, e_mask, e_attns = self.l_encoder(input_str, spk_emb)
            # decoding
            d_out, d_out_post, d_sa_attns, d_da_attns, ffn_o = self.l_decoder(
                e_out, target_tmp, e_mask, data_length, spk_emb)
            
            if self.train_phase == 'freeze_stoptoken':
                # for separate training stage 1, stop_token loss is set to 0
                stop = self.l_stop(ffn_o.detach()) * 0 + 0.5
            else:
                # for joint training, compute the stop-token loss as usual
                stop = self.l_stop(ffn_o.detach())

        elif self.train_phase == 'tune_stoptoken':
            # fix the encoder and decoder part
            self.l_encoder.eval()
            self.l_decoder.eval()
            # no grad on main part
            with torch.no_grad():
                e_out, e_mask, e_attns = self.l_encoder(input_str, spk_emb)
                d_out, d_out_post, d_sa_attns, d_da_attns, ffn_o = self.l_decoder(
                    e_out, target_tmp, e_mask, data_length, spk_emb)
            # only update stop token
            stop = self.l_stop(ffn_o)
        
        else:
            # fix the encoder and decoder part
            self.l_encoder.eval()
            self.l_decoder.eval()
            self.l_stop.eval()
            # no grad on main part
            with torch.no_grad():
                e_out, e_mask, e_attns = self.l_encoder(input_str, spk_emb)
                d_out, d_out_post, d_sa_attns, d_da_attns, ffn_o = self.l_decoder(
                    e_out, target_tmp, e_mask, data_length, spk_emb)
                stop = self.l_stop(ffn_o)
            
        # unfold if reduction factor > 1
        if self.rec_fac > 0:
            d_out_post = d_out_post.view(bsize, -1, self.tardim)
        
        return d_out, d_out_post, stop, d_sa_attns, d_da_attns, e_attns

    def setflag_tune_stoptoken(self):
        self.train_phase = 'tune_stoptoken'
        return

    def setflag_freeze_stoptoken(self):
        self.train_phase = 'freeze_stoptoken'
        return

    def setflag_freeze_all(self):
        self.train_phase = 'freeze_all'
        return
    
        
    def inference(self, input_str, max_length=None, spk_emb=None):
        """output = inference(input_str)

        Method for inference
        
        input
        -----
          input_str: tensor, batch of input character index sequence 
                     (batch, length)
        
        output
        ------
          output: tensor, predicted target sequence
                  (batch, length2, dim)
        """
        # encoding
        e_out, e_mask, e_attns = self.l_encoder(input_str, spk_emb)
        
        # decoding
        bsize = input_str.shape[0]
        feedback = torch.zeros([bsize, self.rec_fac, self.tardim], 
                               device=input_str.device)

        # maximum number of generation
        if max_length is None:
            max_length = self.maxlen
        # default is self.maxlen - 1 steps
        run_steps = min([max_length, self.maxlen - 1])
        for idx in range(run_steps):

            if idx % 50 == 0: print('.', end='', flush=True)
            
            d_out, d_out_post, d_sa_attns, d_da_attns, ffn_o = self.l_decoder(
                e_out, feedback, e_mask, [idx+1] * bsize, spk_emb)

            stop = self.l_stop(ffn_o)
            
            # if the model produces the dummy frame, the loop should end
            if self._stop_gen(stop, input_str.shape[1] + self.endframe_len):
                break
            
            feedback = torch.cat([feedback, d_out[:, -1:, :]], dim=1)

        # unfold if reduction factor > 1
        if self.rec_fac > 0:
            d_out_post = d_out_post.view(bsize, -1, self.tardim)
        
        # remove the dummy padded end frames
        if self.endframe_len > 0:
            d_out_post = d_out_post[:, :-self.endframe_len]

        return d_out_post, d_da_attns, stop

    @staticmethod
    def compute_mel_loss(d_out, d_out_post, target, olens=None):
        """ m1, m2 = compute_mel_loss(d_out, d_out_post, target, olens)
        
        Compute the L1 loss between predicted and natural target sequences

        input
        -----
          d_out: tensor, (batch, length2, dim), generated target sequences
                 before the post-net, i.e., input to the post-net
          d_out_post: tensor, (batch, length2, dim), generated target sequences
                 from the post-net
          target: tensor, (batch, length2, dim), natural target sequences
          olens:  list of int, (batch), length of each utterance

        output
        ------
          m1: scalar, loss between d_out and target
          m2: scalar, loss between d_out_post and target
        """
        # in case the length does not match
        assert d_out.shape[1] == target.shape[1], "Length mismatch"
        
        if olens:
            # 0 for dummy frames
            select_mask = torch.zeros_like(d_out)
            for idx, olen in enumerate(olens):
                select_mask[idx, :olen] = 1.0
            # conver to binary
            select_mask = select_mask.gt(0.0)
        
            d_out_ = d_out.masked_select(select_mask)
            target_ = target.masked_select(select_mask)
            d_out_post_ = d_out_post.masked_select(select_mask)
        else:
            d_out_ = d_out
            target_ = target
            d_out_post_ = d_out_post

        # m1 loss between decoder and target
        m1 = torch_nn_func.l1_loss(d_out_, target_) 
        m2 = torch_nn_func.l1_loss(d_out_post_, target_)
        
        return m1, m2

    @staticmethod    
    def compute_stop_loss(stop_token, data_length):
        """ m = compute_stop_loss(stop_token, data_length)
        
        Compute the stop-token prediction loss

        input
        -----
          stop_token: tensor, (batch, length2), values for stop-token
          data_length: list of int, (batch,), data_length[i] is the desired 
                       length for the i-th sequence in the batch

        output
        ------
          m: scalar, loss 
        """
        weight = torch.ones_like(stop_token)
        target = torch.zeros_like(stop_token)

        for idx in range(len(data_length)):
            # weight at the end of the valid data sequence is increased
            weight[idx, data_length[idx]-1:data_length[idx]] = 10
            # target 000...1111
            target[idx, data_length[idx]-1:] = 1
            
        # BCELoss manully computed
        m = torch_nn_func.binary_cross_entropy(stop_token, target) * weight
        # Do average over each utterance first
        m = m.mean()
        return m
    
    @staticmethod
    def compute_guided_att_loss(attns_mats, olens, ilens, num_heads_ga_loss=2, num_layer_ga_loss=2):
        """compute_guided_attention_loss(self, attns_mats, olens, ilens)
        
        Computed Guided Attention Loss
        """
        # batch
        batch_size = len(olens)

        # device and dtype
        device, dtype = attns_mats[0].device, attns_mats[0].dtype

        # create matrices for penalty and masking
        #  we need to duplicate for multiheads
        weight = attention.GuidedAttentionPeneltyBatch(
            olens * num_heads_ga_loss, 
            ilens * num_heads_ga_loss)

        masks = attention.GuidedAttentionMask(
            olens * num_heads_ga_loss, 
            ilens * num_heads_ga_loss)
        
        # to device
        weight = weight.to(dtype = dtype, device = device)
        masks = masks.to(device = device)
        
        # loss
        loss = 0
        for idx, attn_mat in enumerate(attns_mats[::-1]):
            # only compute for the last a few layers
            if num_layer_ga_loss and idx == num_layer_ga_loss:
                break
            # only compute for the first a few heads
            loss_tmp = weight * attn_mat[: batch_size * num_heads_ga_loss]
            loss += torch.mean(loss_tmp.masked_select(masks))

            
        return loss
        

if __name__ == "__main__":
    print("Definition of TransformerTTS")
