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
        
        if mpath == 'lv':
            # if path to the customized or continually trained wav2vec is given
            self.model = S3PRLUpstream("wav2vec2_large_lv60_cv_swbd_fsh")
        else:
            # otherwise, load the pre-trained xlsr_53
            self.model = S3PRLUpstream("xlsr_53")

        # we use the output before the last projection layer
        self.out_dim = self.model.upstream.model.final_proj.in_features

        # layer index (from which to get hidden features for downstream tasks)
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
        # this is useful when SSLModel is not in model.parameters
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        
        # input data length
        input_tmp_len = torch.tensor([x.shape[0] for x in input_data], 
                                     dtype=torch.long, 
                                     device = input_data.device)
        
        # compute features using sp3rl API
        all_hs, all_hs_len = self.model(input_tmp, input_tmp_len)
        
        # the last Transformer blocks' output after layer norm
        # emb = all_hs[:-2] with layer normalization (w/ affine trans.)
        emb = all_hs[-1]

        # hidden fetures
        hid = torch.stack(all_hs[:-1], dim=-1)

        if self.layer_indices is None:
            return emb
        else:
            return emb, hid


##############
## FOR MODEL
##############

# Front end using a single SSL for CM
class FrontEnd(torch_nn.Module):
    """ Front end wrapper
    """
    def __init__(self, mpath, ssl_out_dim, 
                 fix_ssl=False,
                 ssl_layer_indices=None):
        super(FrontEnd, self).__init__()
        
        # whether fix SSL or not
        self.flag_fix_ssl = fix_ssl
        
        # from which layers do we get the hidden features?
        # if it is None, we only use the last transformer encoders' output
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


class FrontEndTeacher():
    """ Front end teacher that uses two SSLs
    Here, we don't save it as a torch Module so that
    it will not be treated as a part of Model.parameters
    """
    def __init__(self, mpath, ssl_out_dim, 
                 fix_ssl=False,
                 ssl_layer_indices=None):
        
        # whether fix SSL or not
        self.flag_fix_ssl = fix_ssl
        
        # ssl part
        self.ssl_layer_indices = ssl_layer_indices
        # ssl 1 pre-traiend xlsr_53 (see SSLModel)
        self.ssl_model = SSLModel('', ssl_out_dim, ssl_layer_indices)
        # ssl 2 pre-trained lv
        self.ssl_model2 = SSLModel('lv', ssl_out_dim, ssl_layer_indices)

        return

    def set_flag_fix_ssl(self, fix_ssl):
        self.flag_fix_ssl = fix_ssl
        return
    
    def __sent_to_device(self, device, dtype):
        """send the model to device in specific dtype

        FrontEndTeacher is not part of Model(), hence it will not
        be pushed to the device with Model. 
        This must be done manually before training
        """
        # ssl1
        if next(self.ssl_model.parameters()).device != device \
           or next(self.ssl_model.parameters()).dtype != dtype:
            self.ssl_model.to(device, dtype=dtype)
            self.ssl_model.eval()

        # ssl2
        if next(self.ssl_model2.parameters()).device != device \
           or next(self.ssl_model2.parameters()).dtype != dtype:
            self.ssl_model2.to(device, dtype=dtype)
            self.ssl_model2.eval()

        return
        
    def __forward_w_hid(self, wav):
        """ forward with hidden features from SSL
        """
        # SSL part
        if self.flag_fix_ssl:
            self.ssl_model.eval()
            self.ssl_model2.eval()
            with torch.no_grad():
                x_ssl_feat, hid_feat = self.ssl_model.extract_feat(wav)
                x_ssl_feat2, hid_feat2 = self.ssl_model2.extract_feat(wav)
        else:
            x_ssl_feat, hid_feat  = self.ssl_model.extract_feat(wav)
            x_ssl_feat2, hid_feat2 = self.ssl_model2.extract_feat(wav)
            
            
        return x_ssl_feat - x_ssl_feat2, hid_feat - hid_feat2, \
            [x_ssl_feat, x_ssl_feat2, hid_feat, hid_feat2]
    
    def __forward_wo_hid(self, wav):
        """ forward without hidden feature from SSL
        """
        # SSL part
        if self.flag_fix_ssl:
            self.ssl_model.eval()
            self.ssl_model2.eval()
            with torch.no_grad():
                x_ssl_feat = self.ssl_model.extract_feat(wav)
                x_ssl_feat2 = self.ssl_model2.extract_feat(wav)
        else:
            x_ssl_feat = self.ssl_model.extract_feat(wav) 
            x_ssl_feat2 = self.ssl_model2.extract_feat(wav)           
            
        return x_ssl_feat - x_ssl_feat2, [x_ssl_feat, x_ssl_feat2]
    
    def forward(self, wav):
        """ output = front_end(wav)
        
        input:
        ------
          wav: tensor, (batch, length, 1)

        output:
        -------
          output: tuple of tensor
                  if self.ssl_layer_indices is specified
                  [(batch, frame_num, frame_feat_dim)
                   (N, batch, frame_num, frame_feat_dim),

                   [(batch, frame_num, frame_feat_dim),
                    (batch, frame_num, frame_feat_dim),
                    (N, batch, frame_num, frame_feat_dim),
                    (N, batch, frame_num, frame_feat_dim)]]

                    where N = len(self.ssl_layer_indices)
                  else
                   [(batch, frame_num, frame_feat_dim)

                   [(batch, frame_num, frame_feat_dim),
                    (batch, frame_num, frame_feat_dim)]]
        """
        # send models to device if necessary
        self.__sent_to_device(wav.device, wav.dtype)
        
        if self.ssl_layer_indices is None:
            return self.__forward_wo_hid(wav)
        else:
            return self.__forward_w_hid(wav)



class MCDropFFResBlock(torch_nn.Module):
    """MCDropFFResBlock 
    Linear layer + LeakyReLU + MCDropout

    MCDropFFResBlock(input_dim, out_dim, dropout_rate, dropout_flag)
    
    args
    ----
      input_dim:      int, input feature dimension
      out_dim:        int, output feature dimension
      dropout_rate:   float, dropout rate
      dropout_flag:   bool, whether use dropout during training & inference
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
        """ y = MCDropFFResBlock(x)
        
        input
        -----
          x: tensor, (batch, ..., input_dim)
        
        output
        ------
          y: tensor, (batch, ..., out_dim)
        """
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

        # input feature dimension to the feedforward part
        if frontend_hidfeat_compressed_dim:
            self.in_dim = frontend_hidfeat_compressed_dim
        else:
            self.in_dim = input_dim

        # output embedding dimension fro the feedforward part
        self.out_dim = out_dim

        # number of output classes
        self.num_class = num_classes
        
        # dropout rate
        self.m_mcdp_rate = dropout_rate
        self.m_mcdp_flag = dropout_flag
        
        if frontend_hidfeat_num and frontend_hidfeat_compressed_dim:
            # if the input feature is to be merged or compressed
            
            # a trainable layer to merge the features from multiple layers
            self.m_feat_merger = torch_nn.Linear(frontend_hidfeat_num, 1, 
                                                 bias=False)

            # a trainable layer to compress feature dimension
            self.m_feat_compress = torch_nn.Sequential(
                torch.nn.LayerNorm(input_dim, elementwise_affine=False),
                torch_nn.Linear(input_dim, frontend_hidfeat_compressed_dim))
            
            # set the flag
            self.flag_frontend_hid = True

        elif frontend_hidfeat_compressed_dim:
            # if the input feature is to be compressed
            self.m_feat_merger = None
            self.m_feat_compress = torch_nn.Linear(
                input_dim, frontend_hidfeat_compressed_dim)
            # no hidden features
            self.flag_frontend_hid = False

        else:
            # no compression, no hidden features
            self.m_feat_merger = None
            self.m_feat_compress = torch_nn.Identity()
            self.flag_frontend_hid = False

            
        # feedforward part based on full-connected layers 
        #  for frame-level feature processing
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
        for MCDropout sampling during inference, 
        we can duplicate input features, pass them to dropout
        and will get different results
        """
        if self.flag_frontend_hid:
            # if input contains both the last and hidden features
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
            # if we need to merge the features from multiple hidden layers
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
        loss = FeatLossModule(data, target)
    
        input
        -----
          data:   tensor,   input data for feature loss
                  (#sample, #view, feature_dimension_1, ...)

          target: tensor,   target class labels
                  (#sample, #view)
        """
        if data.ndim > 3:
            # loss between frame-level feature sequences, we need to use the
            # similarity metrics defined for two sequences
            return nii_loss_metric.supcon_loss(
                data, target, sim_metric=self.sim_metric_seq, length_norm=True)
        else:
            # loss between embeddings
            return nii_loss_metric.supcon_loss(data, target, length_norm=True)
    



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

        # front-end teacher
        self.ssl_front_end_teacher_path = ''
        # create teacher during training, it is None here
        #  but it will be loaded during the 1st data minibatch
        self.m_front_end_teacher = None
        self.front_end_teacher_out_dim = prj_conf.ssl_front_end_out_dim

        
        # front-end 
        #  we initialize the model to be xlsr_53
        ssl_front_end_path = ''
        # not fixing the front end
        self.flag_fix_ssl = False
        # not using hidden features but using the last layer output
        self.v_ssl_layer_indices = None
        # 
        self.m_front_end = FrontEnd(ssl_front_end_path,
                                    prj_conf.ssl_front_end_out_dim,
                                    self.flag_fix_ssl,
                                    self.v_ssl_layer_indices)

        # back-end
        #  dimension of compressed front-end frame-level features
        self.v_feat_dim = 128
        #  dimension of utterance-level embedding vectors (before softmax)
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

        # weight for distilling
        self.m_feat_dis = 100.0

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


    def _load_front_end_teacher(self):
        """ wrapper to load a FrontEndTeacher during training
        """
        return FrontEndTeacher(
            self.ssl_front_end_teacher_path,
            self.front_end_teacher_out_dim,
            True,
            self.v_ssl_layer_indices)

    def _get_target(self, filenames):
        """ load cross-entropy target from protocol
        """
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def _get_target_vec(self, num_sys, num_aug, bs, device, dtype):
        """ not used anymore
        """
        target = [1] * num_aug + [0 for x in range((num_sys-1) * num_aug)]
        target = np.tile(target, bs)
        target = torch.tensor(target, device=device, dtype=dtype)
        return target
            

    def __inference(self, x, fileinfo):
        """
        __inference(x, fileinfo)
        
        Run CM inference over input data x
        
        input
        -----
          x:           tensor, input waveform data, (batch, length, 1)
          fileinfo:    file information for each data in x
                       fileinfo will be provided by the code
                       if main.py is called with --forward-with-file-name option
        
        output
        ------
          None, the score will be printed 
        """

        # get the file name and file length 
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
        # set the flag to fix the SSL (not necessary, just in case)
        self.m_front_end.set_flag_fix_ssl(True)
        feat_vec = self.m_front_end(x)

        # back end
        #  here we use the Bayesian dropout sampling approach 
        #  simply: we turn on dropout in inference, we do inference
        #  multiple times on each data, then average the output
        #  this allows us to estimate uncertainty
        # 
        #  see http://proceedings.mlr.press/v48/gal16.pdf
        logits_bag = []
        for dropnum in self.m_mcdropout_num:
            # Ideally, we can duplicate the data as a large mini-batch
            # and pass them through dropout layer.
            # To save GPU memory, we use a small "mini-batch" size and
            # do the inference over multiple "mini-batches"
              
            # duplicate the feature N times
            # (dropnum * batch, frame, frame_dim)
            feat_tmp = self.m_back_end.duplicate(feat_vec, dropnum)
            # (dropnum * batch, 2)
            logi_tmp, _ = self.m_back_end(feat_tmp)
            logits_bag.append(logi_tmp)
            
        # original batch size (which is usually 1)
        bs = x.shape[0]
        # total number of dropout instances
        total_samp_num = sum(self.m_mcdropout_num)
        # reshape to (total_dropnum, batch, ...)
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
        loss = __forward_single_view(x, fileinfo)
        
        Run CM forward pass over input data x, which has a single view.
        This is the normal case, and it is used on development set data
        
        input
        -----
          x:           tensor, input waveform data, (batch, length, 1)
          fileinfo:    file information for each data in x
                       fileinfo will be provided by the code
                       if main.py is called with --forward-with-file-name option
        
        output
        ------
          loss:        scalar, los value

        """
        # get the file name and file length 
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        
        # front-end & back-end
        feat_vec = self.m_front_end(x)
        logits, emb_vec = self.m_back_end(feat_vec)
            
        # get the target label from protocol
        target = self._get_target(filenames)
        target_ = torch.tensor(target, device=x.device, dtype=torch.long)
            
        # loss
        loss = self.m_ce_loss(logits, target_)
        return loss

    def __forward_multi_view(self, x_tuple, fileinfo):
        """ 
        loss = __forward_multi_view(x, fileinfo)
        
        Run CM forward pass over input data x_tuple that contains
        bona fide, vocoded data, and their augmented version.

        This is for contrastive learning on the training set.
        
        input
        -----
          x_tuple:     a tuple contains many things
                       It is returned by data_augment.py wav_aug_wrapper.
          fileinfo:    file information for each data in x
                       fileinfo will be provided by the code
                       if main.py is called with --forward-with-file-name option
        
        output
        ------
          losses:      tuple of losses, 

        """

        # get the file name and file length 
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]

        # waveform data
        x = x_tuple[0]

        # MixUP related placeholders
        # tar1, tar2, gamma are place holders for mixup CE loss
        # we don't use mixup in this model, but for compatibility, we 
        # use the interface of MixUP CE
        tar1 = x_tuple[1]
        tar2 = x_tuple[2]
        gamma = x_tuple[3]

        # label for contrastive feature loss
        feat_class = x_tuple[4]

        # batch size (in conventional meaning)
        bat_siz = x.shape[0]
        # waveform length (padded to the longest in the minibatch)
        pad_len = x.shape[1]
        # bona fide + number of vocoded data
        num_sys = x.shape[2]
        # original view + number of augmented views
        num_view = x.shape[3]

        # to (batchsize * (1+num_spoofed) * num_aug, length)
        x_new = x.permute(0, 2, 3, 1).contiguous().view(-1, pad_len)
            
        # front-end & back-end
        feat_vec = self.m_front_end(x_new)
        logits, emb_vec = self.m_back_end(feat_vec)
                        
        # CE loss
        # target vector loaded from data_augment, we don't use _get_target_vec
        # although we use MixUP API, this is just the normal CE loss
        #   [1, 0, 0, ..., 1, 0, 0 ...]
        target1 = tar1.flatten().to(torch.long)
        target2 = tar2.flatten().to(torch.long)
        gamma = gamma.flatten()

        loss_ce = self.m_ce_loss(logits, target1, target2, gamma)
        
            
        # load front-end teacher
        if self.m_front_end_teacher is None:
            self.m_front_end_teacher = self._load_front_end_teacher()
        feat_vec_teacher = self.m_front_end_teacher.forward(x_new)
        # ignore the hidden features
        feat_vec_teacher = feat_vec_teacher[0]
        
        # front-end feature-related loss
        if self.m_feat or self.m_feat_dis:
            
            # buffer for contrastive feature loss
            loss_cr_1, loss_cr_2 = 0, 0
            
            # take only the front end's last layer output
            if type(feat_vec) is tuple:
                feat_vec = feat_vec[0]
            
            # reshape frame-level feature embedding to multi-view format 
            #   (batch, (1+num_spoof), nview, dimension...)
            feat_vec_ = feat_vec.view(
                bat_siz, num_sys, num_view, -1, feat_vec.shape[-1])
            # reshape utterance-level feature embedding
            emb_vec_ = emb_vec.view(bat_siz, num_sys, num_view, -1)

            # compute over each utterance group in the mini-batch
            for bat_idx in range(bat_siz):
                loss_cr_1 += self.m_feat  / bat_siz * self.m_cr_loss(
                    feat_vec_[bat_idx], feat_class[bat_idx])

                loss_cr_2 += self.m_feat  / bat_siz * self.m_cr_loss(
                    emb_vec_[bat_idx], feat_class[bat_idx])
                
            # distilling loss
            loss_dis = self.m_feat_dis * \
                       torch_nn_func.l1_loss(feat_vec, feat_vec_teacher)
            
            return [[loss_ce, loss_cr_1, loss_cr_2, loss_dis], 
                    [True, True, True, True]]
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


class Loss():
    """ Wrapper for scripts, leave as it is
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

    
