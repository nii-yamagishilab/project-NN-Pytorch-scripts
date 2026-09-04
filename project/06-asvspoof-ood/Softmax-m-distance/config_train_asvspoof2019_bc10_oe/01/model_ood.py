#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import core_modules.p2sgrad as nii_p2sgrad
import sandbox.eval_asvspoof as nii_asvspoof 
import importlib
import core_scripts.math_tools.stats as nii_stats_tk

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##############
## FOR MODEL
##############

class ClassParser():
    def __init__(self, protocol_filepaths, sep=' '):
        data_buffer = {}
        if type(protocol_filepaths) is str:
            tmp = [protocol_filepaths]
        else:
            tmp = protocol_filepaths
        for protocol_filepath in tmp:
            with open(protocol_filepath, 'r') as file_ptr:
                for line in file_ptr:
                    line = line.rstrip('\n')
                    cols = line.split(sep)
                    data_buffer[cols[1]] = cols[-2]
        possible_values = list(set(data_buffer.values()))
        self.data_buffer = data_buffer
        self.mapper = possible_values
        self.mapper.sort()
        return

    def retrieve_ID(self, filename):
        if filename in self.data_buffer:
            return self.mapper.index(self.data_buffer[filename])
        else:
            print("Cannot find {:s}".format(filename))
            return None

    def get_num_class(self):
        return len(self.mapper)


class Model(torch_nn.Module):
    """ Model definition
    """
    def __init__(self, in_dim, out_dim, args, prj_conf, mean_std=None):
        super(Model, self).__init__()

        ##### required part, no need to change #####

        # mean std of input and output
        in_m, in_s, out_m, out_s = self.prepare_mean_std(
            in_dim,out_dim, args, prj_conf, mean_std)
        self.input_mean = torch_nn.Parameter(in_m, requires_grad=False)
        self.input_std = torch_nn.Parameter(in_s, requires_grad=False)
        self.output_mean = torch_nn.Parameter(out_m, requires_grad=False)
        self.output_std = torch_nn.Parameter(out_s, requires_grad=False)        
        #####
        
        ####
        # on input waveform and output target
        ####
        # A flag to log validation set
        self.flag_validation = None
        
        # Load protocol and prepare the target data for network training
        protocol_file = prj_conf.optional_argument
        self.class_parser = ClassParser(protocol_file)

        # Backend-model
        # The model should be define
        # model.v_emd_dim: dimension of the embedding
        # model.get_embedding(x): compute the embedding given the input x
        model = importlib.import_module("model")
        self.m_model = model.Model(in_dim, out_dim, args, prj_conf, mean_std)
        self.g_pretrained_model_path = 'trained_network_oe.pt'
        # loading from OE trained model by model_oe, needs no prefix
        self.g_pretrained_model_prefix = ''
        
        # manually set a number of class number
        # this number will be saved as the 
        self.num_class = 1024
        
        embdim = self.m_model.v_emd_dim
        self.m_mean_vec = torch_nn.Parameter(
            torch.zeros([self.num_class, embdim]), requires_grad=False)
        self.m_cov_mat = torch_nn.Parameter(
            torch.zeros([self.num_class, embdim, embdim]), 
            requires_grad=False)
        self.m_counter = torch_nn.Parameter(
            torch.tensor([0 for x in range(self.num_class)], 
                         dtype=torch.int32),
            requires_grad=False)
        self.m_inv_mat = None

        self.m_dataprob = nii_debug.data_probe()
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, prj_conf, 
                         data_mean_std=None):
        """ prepare mean and std for data processing
        This is required for the Pytorch project, but not relevant to this code
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
        This is required for the Pytorch project, but not relevant to this code
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but not relevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but not relevant to this code
        """
        return y * self.output_std + self.output_mean


    def forward(self, input, fileinfo):
        if self.flag_validation:
            return torch.zeros([1])

        self.m_model.eval()
        # compute mean and cov for each class
        with torch.no_grad():
            self.m_model.eval()
            filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
            emb_vecs = self.m_model.get_embedding(input, fileinfo)
            class_ids = [self.class_parser.retrieve_ID(x) for x in filenames]

            # do the computation in an online mode
            for emb_vec, class_id in zip(emb_vecs, class_ids):
                m, cov, cnt = nii_stats_tk.f_online_mean_cov_tensor(
                    emb_vec.unsqueeze(0).unsqueeze(0),
                    self.m_mean_vec[class_id:class_id+1],
                    self.m_cov_mat[class_id:class_id+1],
                    self.m_counter[class_id].item())
                
                self.m_mean_vec[class_id:class_id+1] = m
                self.m_cov_mat[class_id:class_id+1] = cov
                self.m_counter[class_id] = cnt

        # fake a zero loss which will not update anything by back-progpatation
        return torch.zeros([1])

    

    def inference(self, input, fileinfo):

        # [batch, class, dim]
        valid_idx = self.m_counter > 0
        
        if self.m_inv_mat is None:
            # for the first time
            # filter out unfull rank cov
            edim = self.m_mean_vec.shape[-1]
            for idx in range(len(valid_idx)):
                flag = valid_idx[idx]
                if flag and torch.matrix_rank(self.m_cov_mat[idx]) == edim:
                    valid_idx[idx] = True
                else:
                    valid_idx[idx] = False
                    # set it to 0, avoid invalid matrix in the future
                    self.m_counter[idx] = 0
                
        
        num_class = torch.sum(valid_idx)
        mean_vec = self.m_mean_vec[valid_idx]
        cov_mat = self.m_cov_mat[valid_idx]

        if self.m_inv_mat is None:
            # do the inverse for all classes
            self.m_inv_mat = torch.inverse(cov_mat)
            
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        emb_vecs = self.m_model.get_embedding(input, fileinfo)
        class_ids = [self.class_parser.retrieve_ID(x) for x in filenames]
        
        # [batch, class, dim]
        score_vecs = emb_vecs.unsqueeze(1).repeat(1, num_class, 1)
        score_vecs = score_vecs - mean_vec
        # [batch, class, 1, dim] * cov * [batch, class, dim, 1]
        score_vecs = torch.matmul(
            torch.matmul(score_vecs.unsqueeze(2), self.m_inv_mat),
            score_vecs.unsqueeze(-1))
        
        for filename, score_vec, idx in zip(filenames, score_vecs, class_ids):
            print("Output, %s, %d, %f" % (filename, idx, score_vec.min().item()))
        
        # prepare to dump the statistics
        for filename, emb, score_vec in zip(filenames, emb_vecs, score_vecs):
            self.m_dataprob.add_data(filename)
            self.m_dataprob.add_data(emb)
            self.m_dataprob.add_data(score_vec)

        return None
    
    def finish_up_inference(self):
        filename = os.getenv("TEMP_SAVE_FILE_NAME")
        if filename == '':
            filename = 'dump'
        self.m_dataprob.dump(output_path = 'debug/{:s}_data'.format(filename),
                             add_time_tag = False)
        return


class Loss():
    """ Wrapper to define loss function 
    """
    def __init__(self, args):
        """
        """
        return


    def compute(self, outputs, target):
        """ 
        """
        loss = outputs
        return loss

    
if __name__ == "__main__":
    print("Definition of model")

    
