#!/usr/bin/env python
"""
model.py

Self defined model definition.
Usage:

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
from torch import linalg as torch_la

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import sandbox.eval_asvspoof as nii_asvspoof 
import sandbox.util_bayesian as nii_bayesian
import sandbox.util_loss_metric as nii_loss_metric

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

############################
## FOR pre-trained MODEL
############################
from s3prl.nn import S3PRLUpstream

class SSLModel(torch_nn.Module):
    def __init__(self, mpath, ssl_orig_output_dim, output_layer_idx=None):
        """ SSLModel(cp_path, ssl_orig_output_dim, output_layer_idx=None)
        
        Args
        ----
          mpath: string, path to the pre-trained SSL model
          ssl_orig_output_dim: int, dimension of the SSL model output feature
          output_layer_idx: list of int, layer index from which to get output
        """
        super(SSLModel, self).__init__()

        self.model = S3PRLUpstream("wav2vec2_custom", path_or_url = mpath)

        # this should be loaded from model
        self.out_dim = ssl_orig_output_dim

        # layer index
        self.layer_indices = output_layer_idx

        return

    def extract_feat(self, input_data):
        """ output = extract_feat(input_data)
        
        input:
        ------
          input_data,tensor, (batch, length, 1) or (batch, length)
          datalength: list of int, length of wav in the mini-batch

        output:
        -------
          output: tensor, 
                  if self.layer_indices is None, it has shape
                     (batch, frame_num, frame_feat_dim) 
                  else
                     (batch, frame_num, frame_feat_dim),
                     (batch, frame_num, frame_feat_dim, N)
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
        
        input_tmp_len = torch.tensor([x.shape[0] for x in input_data], 
                                     dtype=torch.long, 
                                     device = input_data.device)
        
        all_hs, all_hs_len = self.model(input_tmp, input_tmp_len)
        
        emb = all_hs[-1]
        hid = torch.stack(all_hs[:-1], dim=-1)
        if self.layer_indices is None:
            return emb
        else:
            return emb, hid


##############
## FOR MODEL
##############
class FrontEnd(torch_nn.Module):
    """ Front end wrapper
    """
    def __init__(self, mpath, ssl_out_dim, 
                 fix_ssl=False,
                 ssl_layer_indices=None):
        super(FrontEnd, self).__init__()
        
        # whether fix SSL or not
        self.flag_fix_ssl = fix_ssl
        
        # ssl part
        self.ssl_layer_indices = ssl_layer_indices
        self.ssl_model = SSLModel(mpath, ssl_out_dim, ssl_layer_indices)
                
        return

    def set_flag_fix_ssl(self, fix_ssl):
        self.flag_fix_ssl = fix_ssl
        return
    

    def __forward_w_hid(self, wav):
        """ forward with hidden features from SSL
        """
        # SSL part
        if self.flag_fix_ssl:
            self.ssl_model.eval()
            with torch.no_grad():
                x_ssl_feat, hid_feat = self.ssl_model.extract_feat(wav)
        else:
            x_ssl_feat, hid_feat  = self.ssl_model.extract_feat(wav)
            
        return x_ssl_feat, hid_feat

    def __forward_wo_hid(self, wav):
        """ forward without hidden feature from SSL
        """
        # SSL part
        if self.flag_fix_ssl:
            self.ssl_model.eval()
            with torch.no_grad():
                x_ssl_feat = self.ssl_model.extract_feat(wav)
        else:
            x_ssl_feat = self.ssl_model.extract_feat(wav)
            
        return x_ssl_feat
    
    def forward(self, wav):
        """ output = front_end(wav)
        
        input:
        ------
          wav: tensor, (batch, length, 1)

        output:
        -------
          output: tensor, (batch, frame_num, frame_feat_dim)
                  or list of tensor 
                  [(batch, frame_num, frame_feat_dim)
                   (N, batch, frame_num, frame_feat_dim)]
                  
        """
        if self.ssl_layer_indices is None:
            return self.__forward_wo_hid(wav)
        else:
            return self.__forward_w_hid(wav)

class MCDropFFResBlock(torch_nn.Module):
    """
    """
    def __init__(self, input_dim, out_dim, dropout_rate, dropout_flag):
        super(MCDropFFResBlock, self).__init__()
        # dimension
        self.in_dim = input_dim
        self.out_dim = out_dim
        # MC dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        
        self.m_block = torch_nn.Sequential(
            torch_nn.Linear(self.in_dim, self.out_dim),
            torch_nn.LeakyReLU(),
            nii_nn.DropoutForMC(self.m_mcdp_rate,self.m_mcdp_flag))
        return

    def forward(self, x):
        return self.m_block(x)
        
        

            
class BackEnd(torch_nn.Module):
    """Back End Wrapper
    
    Argument
    --------
      input_dim: int, input feature dimension
      out_dim: int, feature dimension before softmax
      num_classes: int, number of classes in softmax
      dropout_rate: float, dropout rate
      dropout_flag: bool, whether apply Droptout in inference
      frontend_hidfeat_num: int or None, if the front-end returns
                            multiple feature vector, how many?
      frontend_hidfeat_compressed_dim: int, dimension of feature
                            after compression.
      num_layer: int, number of FF layer blocks

    
    input -> feature merge -> feature compress -> feedforward -> softmax
            frontend_hidfeat_num
                            frontend_hidfeat_compressed_dim
                                                  out_dim        num_classes 

    """
    def __init__(self, input_dim, out_dim, num_classes, 
                 dropout_rate, dropout_flag=True,
                 frontend_hidfeat_num = None, 
                 frontend_hidfeat_compressed_dim = None,
                 num_layer = 3):
        super(BackEnd, self).__init__()

        # input feature dimension
        if frontend_hidfeat_compressed_dim:
            self.in_dim = frontend_hidfeat_compressed_dim
        else:
            self.in_dim = input_dim

        # output embedding dimension
        self.out_dim = out_dim
        # number of output classes
        self.num_class = num_classes
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        
        if frontend_hidfeat_num and frontend_hidfeat_compressed_dim:
            self.m_feat_merger = torch_nn.Linear(frontend_hidfeat_num, 1, 
                                                 bias=False)
            self.m_feat_compress = torch_nn.Sequential(
                torch.nn.LayerNorm(input_dim, 
                                   elementwise_affine=False),
                torch_nn.Linear(input_dim, frontend_hidfeat_compressed_dim))
            self.flag_frontend_hid = True

        elif frontend_hidfeat_compressed_dim:
            self.m_feat_merger = None
            self.m_feat_compress = torch_nn.Linear(
                input_dim, frontend_hidfeat_compressed_dim)
            self.flag_frontend_hid = False

        else:
            self.m_feat_merger = None
            self.m_feat_compress = torch_nn.Identity()
            self.flag_frontend_hid = False

        # a simple full-connected network for frame-level feature processing
        tmp = []
        for idx in range(num_layer-1):
            tmp.append(MCDropFFResBlock(self.in_dim, self.in_dim, 
                                        self.m_mcdp_rate, self.m_mcdp_flag))
        tmp.append(MCDropFFResBlock(self.in_dim, self.out_dim, 
                                    self.m_mcdp_rate, self.m_mcdp_flag))
        self.m_frame_level = torch_nn.Sequential(*tmp)

        # linear layer to produce output logits 
        self.m_utt_level = torch_nn.Linear(self.out_dim, self.num_class)


        
        return

    def duplicate(self, feat, num):
        """
        for many times of sampling, we need to do it multiple times.
        we can duplicate input features, pass them to dropout
        and will get different dropout results
        """
        if self.flag_frontend_hid:
            return [feat[0].repeat(num, 1, 1),
                    feat[1].repeat(num, 1, 1, 1)]
        else:
            # input feat is in shape (batch, frame_num, feat_dim)
            return feat.repeat(num, 1, 1)
    
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
        
        if self.flag_frontend_hid:
            _, hid_feat = feat[0], feat[1]
            # (batch, time, feat_dim, N) -> (batch, time, feat_dim)
            feat = self.m_feat_merger(hid_feat).squeeze(-1)
            
        # (batch, time, feat_dim) -> (batch, time, compressed_dim)
        feat = self.m_feat_compress(feat)

        
        # through the frame-level network
        # (batch, frame_num, self.out_dim)
        feat_ = self.m_frame_level(feat)
        
        # average pooling -> (batch, self.out_dim)
        feat_utt = feat_.mean(1)
        
        # output linear 
        logits = self.m_utt_level(feat_utt)
        return logits, feat_utt

class FeatLossModule(torch_nn.Module):
    """ Loss wrapper over features
    Here we use supervised contrastive feature loss
    """
    def __init__(self):
        super(FeatLossModule, self).__init__()

        # similarity metric for two sequences
        self.sim_metric_seq = lambda mat1, mat2: torch.bmm(
            mat1.permute(1, 0, 2), mat2.permute(1, 2, 0)).mean(0)

        return

    def forward(self, data, target):
        """
        """
        if data.ndim > 3:
            # loss between frame-level feature sequences
            return nii_loss_metric.supcon_loss(
                data, target, sim_metric=self.sim_metric_seq, length_norm=True)
        else:
            # loss between embeddings
            return nii_loss_metric.supcon_loss(data, target, length_norm=True)
    

class RankLossModuleAngle(torch_nn.Module):
    """ Rank Loss wrapper over features
    """
    def __init__(self):
        super(RankLossModuleAngle, self).__init__()

        # consistency for two vectors:
        # x: (D, ), y: (D, ) -> (1,)
        self.c_vec = lambda x, y: -torch_nn_func.cosine_similarity(x,y,0)

        # x: (L, D), y: (L, D) -> (L, 1) -> (1, )
        # consistency for two sequences
        self.c_seq = lambda x, y: -torch_nn_func.cosine_similarity(x,y,1).mean()
        
        return

    def forward(self, data, target):
        """
        input
        -----
          data: tensor, (batch, view, length, dim) or (batch, view, dim)
        """
        loss = 0.0
        for i in np.arange(data.shape[1]):
            if data.ndim > 3:
                # create an artificial anchor all ones vector
                anchor = torch.ones_like(data[0, i])
                # loss between frame-level feature sequences
                loss += nii_loss_metric.rank_consistency(
                    data[:, i], self.c_seq, anchor)
            else:
                # create an artificial anchor all ones vector
                anchor = torch.ones_like(data[0, i])
                # loss between embeddings
                loss += nii_loss_metric.rank_consistency(
                    data[:, i], self.c_vec, anchor)
        return loss
    

class RankLossModuleMag(torch_nn.Module):
    """ Rank Loss wrapper over features
    """
    def __init__(self, margin=0.1):
        super(RankLossModuleMag, self).__init__()

        # consistency for two vectors:
        # x: (D, ), y: (D, ) -> (1,)
        self.c_vec = lambda x1, x2: torch_nn_func.margin_ranking_loss(
            torch_la.vector_norm(x1, ord=2, keepdim=True), 
            torch_la.vector_norm(x2, ord=2, keepdim=True), 
            -torch.ones_like(x1[0:1]), margin)
        
        return

    def forward(self, data, target):
        """
        input
        -----
          data: tensor, (batch, view, length, dim) or (batch, view, dim)
        """
        loss = 0.0
        for i in np.arange(data.shape[1]):
            if data.ndim > 3:
                # loss between frame-level feature sequences
                loss += nii_loss_metric.rank_consistency_v2(
                    data[:, i].mean(dim=1), self.c_vec)
            else:
                # loss between embeddings
                loss += nii_loss_metric.rank_consistency_v2(
                    data[:, i], self.c_vec)
        return loss


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
        # Bayesian Dropout parameter
        # This is not for contrastive loss or vocoded data
        # This is for uncertainty estimation.
        ####
        # dropout rate
        self.m_mcdp_rate = 0.5
        # whether use Bayesian dropout during inference
        self.m_mcdp_flag = True
        # How many samples to draw for Bayesian uncertainty estimation.
        # Naively, we duplicate one data 144 times and create a mini-batch
        # By doing so, we simultanesouly do sampling and get 144 samples for
        # each input data.
        # But the mini-batch may be too large in size.
        # Hence, we do the sampling three times, each time draw 48 samples
        # 
        # If your GPU memorgy is insufficient, try to reduce this number
        self.m_mcdropout_num = [48, 48, 48]

        ####
        # Model definition
        ####    
        # front-end 

        self.flag_fix_ssl = False
        self.v_ssl_layer_indices = None
        self.m_front_end = FrontEnd(prj_conf.ssl_front_end_path,
                                    prj_conf.ssl_front_end_out_dim,
                                    self.flag_fix_ssl,
                                    self.v_ssl_layer_indices)
        
        # back-end
        # dimension of utterance-level embedding vectors
        self.v_feat_dim = 128
        self.v_emd_dim = self.v_feat_dim
        # number of output classes
        self.v_out_class = 2
        # back end
        self.m_back_end = BackEnd(prj_conf.ssl_front_end_out_dim, 
                                  self.v_emd_dim, 
                                  self.v_out_class,
                                  self.m_mcdp_rate,
                                  self.m_mcdp_flag,
                                  None,
                                  self.v_feat_dim,
                                  num_layer = 3)
            
        #####
        # Loss function
        #####
        # we do not use mixup loss for experiments.
        # here, it is used for compatability. 
        self.m_ce_loss = nii_loss_metric.MixUpCE()
        
        # feature loss module
        self.m_cr_loss = FeatLossModule()
        
        # weight for the contrastive feature loss
        self.m_feat = 1.0

        # rank loss based on direction
        self.m_rank_a = 0.0
        self.m_rk_loss_a = RankLossModuleAngle()
        
        # rank loss based on magnitude
        self.m_rank_m = 0.0
        self.m_rk_loss_m = RankLossModuleMag()

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

        # front end
        self.m_front_end.set_flag_fix_ssl(True)
        feat_vec = self.m_front_end(x)

        # back end
        # not relevant to the vocoded training data and contrastive loss,
        # this is for another project of uncertainty estimation.
        # we will use code as it is
        # 
        # Bayesian dropout sampling
        logits_bag = []
        for dropnum in self.m_mcdropout_num:
            # Simulate Bayesian sampling in inference
            # for many times of sampling, we need to do it multiple times.
            # we can duplicate SSL output from SSL, pass them to dropout
            # and will get different dropout results
              
            # duplicate the feature N times
            # (dropnum * batch, frame, frame_dim)
            feat_tmp = self.m_back_end.duplicate(feat_vec, dropnum)
            # (dropnum * batch, 2)
            logi_tmp, _ = self.m_back_end(feat_tmp)
            logits_bag.append(logi_tmp)
            
        # (total_dropnum * batch, 2)
        bs = x.shape[0]
        total_samp_num = sum(self.m_mcdropout_num)
        logits = torch.cat(logits_bag, dim=0).view(total_samp_num, bs, -1)

        # compute CM scores, model uncertainty, data uncertainty
        # we need to use CM scores to compute EER 
        scores, epss, ales = nii_bayesian.compute_llr_eps_ale(logits)
            
        targets = self._get_target(filenames)
        for filename, target, score, eps, ale in \
            zip(filenames, targets, scores, epss, ales):
            print("Output, {:s}, {:d}, {:f}, {:f}, {:f}".format(
                filename, target, score.item(), eps.item(), ale.item()))
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

    def __forward_multi_view(self, x_tuple, fileinfo):
        """
        """
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]

        # input will be [input_tensor, target_1, target_2, gamma_list]
        
        x = x_tuple[0]
        tar1 = x_tuple[1]
        tar2 = x_tuple[2]
        gamma = x_tuple[3]
        feat_class = x_tuple[4]

        # input_tensor will be (batchsize, length, batch size in data_augment)
        bat_siz = x.shape[0]
        pad_len = x.shape[1]
        num_sys = x.shape[2]
        num_view = x.shape[3]

        # to (batchsize * (1+num_spoofed) * num_aug, length)
        x_new = x.permute(0, 2, 3, 1).contiguous().view(-1, pad_len)

        # target vector loaded from data_augment, we don'y use _get_target_vec
        # this is for CE loss
        # [1, 0, 0, ..., 1, 0, 0 ...]
        target1 = tar1.flatten().to(torch.long)
        target2 = tar2.flatten().to(torch.long)
        gamma = gamma.flatten()
            
        # front-end & back-end
        feat_vec = self.m_front_end(x_new)
        logits, emb_vec = self.m_back_end(feat_vec)
                        
        # CE loss
        loss_ce = self.m_ce_loss(logits, target1, target2, gamma)
            
        if self.m_feat or self.m_rank:
            # feat loss
            loss_cr_1, loss_cr_2 = 0, 0 
            
            if type(feat_vec) is tuple:
                # if front-end returns hidden features, use the final output
                feat_vec = feat_vec[0]
            
            # reshape to multi-view format 
            #   (batch, (1+num_spoof), nview, dimension...)
            feat_vec_ = feat_vec.view(
                bat_siz, num_sys, num_view, -1, feat_vec.shape[-1])
            emb_vec_ = emb_vec.view(bat_siz, num_sys, num_view, -1)

            for bat_idx in range(bat_siz):
                loss_cr_1 += self.m_feat  / bat_siz * self.m_cr_loss(
                    feat_vec_[bat_idx], feat_class[bat_idx])

                loss_cr_2 += self.m_feat  / bat_siz * self.m_cr_loss(
                    emb_vec_[bat_idx], feat_class[bat_idx])

            return [[loss_ce, loss_cr_1, loss_cr_2], 
                    [True, True, True]]
        else:
            return loss_ce

        
    def forward(self, x, fileinfo):
        """
        """
        if self.training and type(x) is list and x[0].shape[2] > 1:
            # if training with multi-view data
            # contrastive feature loss requires multi-view of data
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

    
