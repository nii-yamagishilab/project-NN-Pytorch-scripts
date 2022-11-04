#!/usr/bin/env python
"""
model.py for Active learning model

This model.py consists of two parts:
1. A CM with SSL-based front-end and linear back-end. 
   The same model as 07-asvspoof-ssl/model-W2V-XLSR-ft-GF/model.py,
   but the code is revised and simplified.
2. A function al_retrieve_data to scoring the pool set data.

al_retrieve_data scores the pool set data and returns a list of data index.
The returned data index will be used to retrieve the data from pool.

al_retrieve_data is called in core_scripts/nn_manager/nn_manager_AL.py.
Please check the training algorithm there.
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
import sandbox.eval_asvspoof as nii_asvspoof 
import sandbox.util_bayesian as nii_bayesian
import sandbox.util_loss_metric as nii_loss_util


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

############################
## FOR pre-trained MODEL
############################
import fairseq as fq

class SSLModel(torch_nn.Module):
    def __init__(self, mpath, ssl_orig_output_dim):
        """ SSLModel(cp_path, ssl_orig_output_dim)
        
        Args
        ----
          mpath: string, path to the pre-trained SSL model
          ssl_orig_output_dim: int, dimension of the SSL model output feature
        """
        super(SSLModel, self).__init__()
        md, _, _ = fq.checkpoint_utils.load_model_ensemble_and_task([mpath])

        self.model = md[0]
        # this should be loaded from md
        self.out_dim = ssl_orig_output_dim
        return

    def extract_feat(self, input_data):
        """ output = extract_feat(input_data)
        
        input:
        ------
          input_data,tensor, (batch, length, 1) or (batch, length)
          datalength: list of int, length of wav in the mini-batch

        output:
        -------
          output: tensor, (batch, frame_num, frame_feat_dim)
        """
        
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        
        # emb has shape [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        
        return emb


##############
## FOR MODEL
##############
class FrontEnd(torch_nn.Module):
    """ Front end wrapper
    """
    def __init__(self, output_dim, mpath, ssl_out_dim, fix_ssl=False):
        super(FrontEnd, self).__init__()
        
        # dimension of output feature
        self.out_dim = output_dim
        
        # whether fix SSL or not
        self.flag_fix_ssl = fix_ssl
        
        # ssl part
        self.ssl_model = SSLModel(mpath, ssl_out_dim)
        
        # post transformation part
        self.m_front_end_process = torch_nn.Linear(
            self.ssl_model.out_dim, self.out_dim)
        
        return

    def set_flag_fix_ssl(self, fix_ssl):
        self.flag_fix_ssl = fix_ssl
        return
    
    def forward(self, wav):
        """ output = front_end(wav)
        
        input:
        ------
          wav: tensor, (batch, length, 1)

        output:
        -------
          output: tensor, (batch, frame_num, frame_feat_dim)
        """
        if self.flag_fix_ssl:
            self.ssl_model.eval()
            with torch.no_grad():
                x_ssl_feat = self.ssl_model.extract_feat(wav)
        else:
            x_ssl_feat = self.ssl_model.extract_feat(wav)
        output = self.m_front_end_process(x_ssl_feat)
        return output

            
class BackEnd(torch_nn.Module):
    """Back End Wrapper
    """
    def __init__(self, input_dim, out_dim, num_classes, 
                 dropout_rate, dropout_flag=True, dropout_trials=[1]):
        super(BackEnd, self).__init__()

        # input feature dimension
        self.in_dim = input_dim
        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        self.m_mcdp_num  = dropout_trials
        

        # linear linear to produce output logits 
        self.m_utt_level = torch_nn.Linear(self.out_dim, self.num_class)
        
        return

    def forward(self, feat):
        """ logits, emb_vec = back_end_emb(feat)

        input:
        ------
          feat: tensor, (batch, frame_num, feat_feat_dim)

        output:
        -------
          logits: tensor, (batch, num_output_class)
          emb_vec: tensor, (batch, emb_dim)
        """
        # through the frame-level network
        # (batch, frame_num, self.out_dim)
        
        # average pooling -> (batch, self.out_dim)
        feat_utt = feat.mean(1)
        
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt

    def inference(self, feat):
        """scores, emb_vec, energy = inference(feat)
        
        This is used for inference, output includes the logits and 
        confidence scores.

        input:
        ------
          feat: tensor, (batch, frame_num, feat_feat_dim)

        output:
        -------
          scores: tensor, (batch, 1)
          emb_vec: tensor, (batch, emb_dim)
          energy:   tensor,   (batch, 1)
        """
        # logits
        logits, feat_utt = self.forward(feat)
        
        # logits -> score
        scores = logits[:, 1] - logits[:, 0]

        # compute confidence using negative energy
        energy = nii_loss_util.neg_energy(logits)
        
        return scores, feat_utt, energy
        


class MainLossModule(torch_nn.Module):
    """ Loss wrapper
    """
    def __init__(self):
        super(MainLossModule, self).__init__()
        self.m_loss = torch_nn.CrossEntropyLoss()
        return
    
    def forward(self, logits, target):
        return self.m_loss(logits, target)


class FeatLossModule(torch_nn.Module):
    """ Loss wrapper over features
    Not used here
    """
    def __init__(self):
        super(FeatLossModule, self).__init__()
        return

    def forward(self, data, target):
        """
        """
        return 0


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
        
        # a flag for debugging (by default False)
        #self.model_debug = False
        #self.validation = False
        ############################################
        
        ####
        # auxililary 
        ####
        # flag of current training stage
        #  this variable will be overwritten 
        self.temp_flag = args.temp_flag

        ####
        # Load protocol and prepare the target data for network training
        ####
        protocol_f = prj_conf.optional_argument
        self.protocol_parser = nii_asvspoof.protocol_parse_general(protocol_f)
        

        ####
        # Bayesian parameter
        ####
        self.m_mcdp_rate = None
        self.m_mcdp_flag = True
        # if [1], we will only do one inference
        self.m_mcdropout_num = [1]

        ####
        # Model definition
        ####    
        # front-end 
        #  dimension of compressed front-end feature
        self.v_feat_dim = 128
        self.m_front_end = FrontEnd(self.v_feat_dim,
                                    prj_conf.ssl_front_end_path,
                                    prj_conf.ssl_front_end_out_dim)
        
        # back-end
        # dimension of utterance-level embedding vectors
        self.v_emd_dim = self.v_feat_dim
        # number of output classes
        self.v_out_class = 2
        self.m_back_end = BackEnd(self.v_feat_dim, 
                                  self.v_emd_dim, 
                                  self.v_out_class,
                                  self.m_mcdp_rate,
                                  self.m_mcdp_flag,
                                  self.m_mcdropout_num)
            
        #####
        # Loss function
        #####
        self.m_ce_loss = MainLossModule()
        self.m_cr_loss = FeatLossModule()
        # weight for the feature loss
        self.m_feat = 0.0

                
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, 
                         prj_conf, data_mean_std=None):
        """ prepare mean and std for data processing
        This is required for the Pytorch project, but irrelevant to this code
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
        This is required for the Pytorch project, but irrelevant to this code
        """
        return (x - self.input_mean) / self.input_std

    def normalize_target(self, y):
        """ normalizing the target data
        This is required for the Pytorch project, but irrelevant to this code
        """
        return (y - self.output_mean) / self.output_std

    def denormalize_output(self, y):
        """ denormalizing the generated output from network
        This is required for the Pytorch project, but irrelevant to this code
        """
        return y * self.output_std + self.output_mean


    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def _get_target_vec(self, num_sys, num_aug, bs, device, dtype):
        target = [1] * num_aug + [0 for x in range((num_sys-1) * num_aug)]
        target = np.tile(target, bs)
        target = torch.tensor(target, device=device, dtype=dtype)
        return target
            

    def __inference(self, x, fileinfo):
        """
        """
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        
        # too short sentences, skip it
        if not self.training and x.shape[1] < 3000:
            targets = self._get_target(filenames)
            for filename, target in zip(filenames, targets):
                print("Output, {:s}, {:d}, {:f}, {:f}, {:f}".format(
                    filename, target, 0.0, 0.0, 0.0))
                return None

        # front-end
        feat_vec = self.m_front_end(x)
        
        # back-end
        scores, _, energy = self.m_back_end.inference(feat_vec)
        

        # print output
        targets = self._get_target(filenames)
        for filename, target, score, eps in \
            zip(filenames, targets, scores, energy):
            print("Output, {:s}, {:d}, {:f}, {:f}".format(
                filename, target, score.item(), eps.item()))
            
        # don't write output score as a single file
        return None

    def __forward_single_view(self, x, fileinfo):
        """
        """
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        
        # front-end & back-end
        feat_vec = self.m_front_end(x)
        logits, emb_vec = self.m_back_end(feat_vec)
            
        target = self._get_target(filenames)
        target_ = torch.tensor(target, device=x.device, dtype=torch.long)
            
        # loss
        loss = self.m_ce_loss(logits, target_)
        return loss

    def __forward_multi_view(self, x, fileinfo):
        """
        """
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        
        # input will be (batchsize, length, 1+num_spoofed, num_aug)
        bat_siz = x.shape[0]
        pad_len = x.shape[1]
        num_sys = x.shape[2]
        num_aug = x.shape[3]

        # to (batchsize * (1+num_spoofed) * num_aug, length)
        x_new = x.permute(0, 2, 3, 1).contiguous().view(-1, pad_len)
        datalen_tmp = np.repeat(datalength, num_sys * num_aug)

        # target vector
        # this is for CE loss
        # [1, 0, 0, ..., 1, 0, 0 ...]
        target = self._get_target_vec(num_sys, num_aug, bat_siz, 
                                      x.device, torch.long)
        # this is for contrasitive loss (ignore the augmentation)
        target_feat = self._get_target_vec(num_sys, 1, bat_siz, 
                                           x.device, torch.long)
            

        # front-end & back-end
        feat_vec = self.m_front_end(x_new)
        logits, emb_vec = self.m_back_end(feat_vec)
                        
        # CE loss
        loss_ce = self.m_ce_loss(logits, target)
            

        if self.m_feat:
            # feat loss
            loss_cr_1 = 0 
            loss_cr_2 = 0 
            # reshape to multi-view format 
            #   (batch, (1+num_spoof), nview, dimension...)
            feat_vec_ = feat_vec.view(bat_siz, num_sys, num_aug, -1,
                                      feat_vec.shape[-1])
            emb_vec_ = emb_vec.view(bat_siz, num_sys, num_aug, -1)
            for bat_idx in range(bat_siz):
                loss_cr_1 += self.m_feat  / bat_siz * self.m_cr_loss(
                    feat_vec_[bat_idx],
                    target_feat[bat_idx * num_sys :(bat_idx+1) * num_sys])
                loss_cr_2 += self.m_feat  / bat_siz * self.m_cr_loss(
                    emb_vec_[bat_idx],
                    target_feat[bat_idx * num_sys :(bat_idx+1) * num_sys])
                
            return [[loss_ce, loss_cr_1, loss_cr_2], 
                    [True, True, True]]
        else:
            return loss_ce

        
    def forward(self, x, fileinfo):
        """
        """
        if self.training and x.shape[2] > 1:
            # if training with multi-view data
            return self.__forward_multi_view(x, fileinfo)
        elif self.training:
            return self.__forward_single_view(x, fileinfo)
        else:
            return self.__inference(x, fileinfo)


    def get_embedding(self, x, fileinfo):
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        feature_vec = self._compute_embedding(x, datalength)
        return feature_vec


    def al_retrieve_data(self, data_loader, num_sample):
        """idx = al_retrieve_data(data_loader, num_sample)
        
        Data retrival function for active learning

        Args:
        -----
          data_loader: Pytorch DataLoader for the pool data set
          num_sample: int, number of samples to be selected
        
        Return
        ------
          idx: list of index

        """
        
        # buffer
        # note that data_loader.dataset.__len__() returns the number of 
        # individual samples, not the number of mini-batches
        idx_list = np.zeros([data_loader.dataset.__len__()])
        conf_list = np.zeros([data_loader.dataset.__len__()])
        # 
        counter = 0

        # loop over the pool set
        with torch.no_grad():
            for data_idx, (x, y, data_info, idx_orig) in \
                enumerate(data_loader):
                    
                # feedforward pass
                filenames = [nii_seq_tk.parse_filename(y) for y in data_info]
                datalength = [nii_seq_tk.parse_length(y) for y in data_info]
                
                if isinstance(x, torch.Tensor):
                    x = x.to(self.input_std.device, 
                             dtype=self.input_std.dtype)
                else:
                    nii_display.f_die("data input is not a tensor")
                    
                
                # front-end
                feat_vec = self.m_front_end(x)
        
                # back-end
                scores, _, energy = self.m_back_end.inference(feat_vec)
                
                # add the energy (confidence score) and data index to the buffer
                conf_list[counter:counter+x.shape[0]] = np.array(
                    [x.item() for x in energy])
                idx_list[counter:counter+x.shape[0]] = np.array(
                    idx_orig)
                counter += x.shape[0]

        # select the least useful data (those with low enerngy, high-confidence)
        sorted_idx = np.argsort(conf_list)
        # retrieve the data index
        return_idx = [idx_list[x] for x in sorted_idx[:num_sample]]
        
        # return the data index, 
        # the corresponding samples will be added to training set
        return return_idx

class Loss():
    """ Wrapper for scripts, ignore it
    """
    def __init__(self, args):
        """
        """
        
    def compute(self, outputs, target):
        """ 
        """
        return outputs

    
if __name__ == "__main__":
    print("Definition of model")

    
