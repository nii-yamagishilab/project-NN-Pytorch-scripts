#!/usr/bin/env python
"""
 
"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"


class HighwayGate(torch_nn.Module):
    """
    """
    def __init__(self, num_dims, 
                 act=torch_nn_func.relu):
        """
        Args:
            num_dims: int, dimension of hidden unit
            act:   function, activation function for transformation
                   default torch.nn.functional.relu
        """
        super(HighwayGate, self).__init__()
        self.m_trans = torch_nn.Linear(num_dims, num_dims)
        self.m_gate = torch_nn.Linear(num_dims, num_dims)
        self.m_trans_act = act
        
        torch_nn.init.xavier_uniform_(
            self.m_trans.weight,
            gain=torch_nn.init.calculate_gain('linear'))
        torch_nn.init.xavier_uniform_(
            self.m_gate.weight,
            gain=torch_nn.init.calculate_gain('linear'))
    
        return
    
    def forward(self, x):
        gate_output = torch.sigmoid(self.m_gate(x))
        trans_output = self.m_trans_act(self.m_trans(x))
        return trans_output * gate_output + x * (1. - gate_output)
    
        
class HighwayNet(torch_nn.Module):
    """
    HighwayNet
    """
    def __init__(self, num_dims, 
                 num_layers=4, 
                 act=torch_nn_func.relu):
        """
        Args:
            num_dims: int, dimension of hidden unit
            num_layers: int, number of highway layers (default 4)
        """
        super(HighwayNet, self).__init__()
        self.m_core = torch_nn.Sequential(
            *[HighwayGate(num_dims, act) for x in range(num_layers)])
        return

    def forward(self, input_):
        return self.m_core(input_)



  
class CBHG(torch_nn.Module):
    """
    CBHG Module
    """
    def __init__(self, 
                 input_dim, 
                 hidden_dim = 256, 
                 num_conv=16, 
                 num_gru_layers=2, 
                 max_pool_kernel_size=2):
        """
        Args:
            input_dim:  int, number of input dimension
            hidden_dim: int, number of dimensions for hidden 
            num_conv: # of convolution banks
            projection_size: dimension of projection unit
            num_gru_layers: # of layers of GRUcell
            max_pool_kernel_size: max pooling kernel size
            is_post: bool, whether post processing or not
        """
        super(CBHG, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # convolution layers
        self.convbank_list = torch_nn.ModuleList()
        self.convbank_list.append(
            torch_nn.Conv1d(in_channels=input_dim,
                            out_channels=hidden_dim,
                            kernel_size=1))

        for i in range(2, num_conv+1):
            self.convbank_list.append(
                torch_nn.Conv1d(in_channels = hidden_dim,
                                out_channels = hidden_dim,
                                kernel_size = i // 2 * 2 + 1,
                                padding=int(np.floor(i/2))))

        # batchnorm layers
        self.batchnorm_list = torch_nn.ModuleList()
        for i in range(1, num_conv+1):
            self.batchnorm_list.append(torch_nn.BatchNorm1d(hidden_dim))

        # total size 
        convbank_outdim = hidden_dim * num_conv
        
        # 
        self.out_proj = torch_nn.Sequential(
            torch_nn.Conv1d(in_channels=convbank_outdim,
                            out_channels=hidden_dim,
                            kernel_size=3, padding=int(np.floor(3 / 2))),
            torch_nn.BatchNorm1d(hidden_dim),
            torch_nn.ReLU(),
            torch_nn.Conv1d(in_channels=hidden_dim,
                            out_channels=input_dim,
                            kernel_size=3,
                            padding=int(np.floor(3 / 2))),
            torch_nn.BatchNorm1d(input_dim)
            
        )
        
        self.max_pool = torch_nn.MaxPool1d(max_pool_kernel_size, 
                                           stride=1, padding=1)
        
        self.highway = HighwayNet(input_dim)
        
        self.gru = torch_nn.GRU(input_dim, hidden_dim // 2, 
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bidirectional=True)
        return


    def forward(self, input_):

        input_ = input_.transpose(1,2).contiguous()
        
        batch_size = input_.size(0)
        total_length = input_.size(-1)

        convbank_list = list()
        convbank_input = input_

        # Convolution bank filters
        for k, (conv, bn) in enumerate(
                zip(self.convbank_list, self.batchnorm_list)):
            convbank_input = torch.relu(bn(conv(convbank_input).contiguous()))
            convbank_list.append(convbank_input)

        # Concatenate all features
        conv_cat = torch.cat(convbank_list, dim=1)

        # Max pooling
        conv_cat = self.max_pool(conv_cat)[:,:,:-1]

        # Projection
        prj_out = self.out_proj(conv_cat) + input_
        
        # Highway networks
        highway = self.highway(prj_out.transpose(1,2))

        # Bidirectional GRU
        self.gru.flatten_parameters()
        out, _ = self.gru(highway)

        return out



class PostNet(torch_nn.Module):
    def __init__(self, config, hid_dim=256): 
        super(PostNet, self).__init__()

        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        
        self.m_core = torch_nn.Sequential(
            torch_nn.Linear(self.input_dim, hid_dim),
            CBHG(hid_dim),
            torch_nn.Linear(hid_dim, self.output_dim))

        #self.loss = torch_nn.L1Loss()
        return

    @staticmethod
    def compute_spec_loss(d_out, target):
        """ m1, m2 = compute_spec_loss(d_out, d_out_post, target)
        
        Compute the L1 loss between predicted and natural target sequences

        input
        -----
          d_out: tensor, (batch, length2, dim), generated target sequences
                 before the post-net, i.e., input to the post-net
          target: tensor, (batch, length2, dim), natural target sequences
        
        output
        ------
          m1: scalar, loss between d_out and target
          m2: scalar, loss between d_out_post and target
        """
        minlen = min([d_out.shape[1], target.shape[1]])
        
        # loss between decoder and target
        m1 = torch_nn_func.l1_loss(d_out[:, :minlen], target[:, :minlen])

        return m1

    def forward(self, x):
        return self.m_core(x)

if __name__ == "__main__":
    print("postnet")
