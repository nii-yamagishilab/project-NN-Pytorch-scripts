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
import torchaudio
import torch.nn.functional as torch_nn_func

import sandbox.block_nn as nii_nn
import sandbox.util_frontend as nii_front_end
import core_scripts.other_tools.debug as nii_debug
import core_scripts.data_io.seq_info as nii_seq_tk
import sandbox.eval_asvspoof as nii_asvspoof 

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


##############
## FOR MODEL
##############

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
        #####
        
        ####
        # on input waveform and output target
        ####
        # Load protocol and prepare the target data for network training
        protocol_f = prj_conf.optional_argument
        self.protocol_parser = nii_asvspoof.protocol_parse_general(protocol_f)
        
        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        self.m_target_sr = 16000

        ####
        # optional configs (not used)
        ####                
        # re-sampling (optional)
        #self.m_resampler = torchaudio.transforms.Resample(
        #    prj_conf.wav_samp_rate, self.m_target_sr)

        # vad (optional)
        #self.m_vad = torchaudio.transforms.Vad(sample_rate = self.m_target_sr)
        
        # flag for balanced class (temporary use)
        #self.v_flag = 1

        ####
        # front-end configuration
        #  multiple front-end configurations may be used
        #  by default, use a single front-end
        ####    
        # frame shift (number of waveform points)
        self.frame_hops = [160]
        # frame length
        self.frame_lens = [320]
        # FFT length
        self.fft_n = [512]

        # LFCC dim (base component)
        self.lfcc_dim = [20]
        self.lfcc_with_delta = True

        # window type
        self.win = torch.hann_window
        # floor in log-spectrum-amplitude calculating (not used)
        self.amp_floor = 0.00001
        
        # number of frames to be kept for each trial
        # no truncation
        self.v_truncate_lens = [None for x in self.frame_hops]


        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.frame_lens)        

        # dimension of embedding vectors
        # here, the embedding is just the activation before sigmoid()
        self.v_emd_dim = None
        
        self.v_out_class = 2

        ####
        # create network
        ####
        # 1st part of the classifier
        self.m_transform = []
        # 
        self.m_before_pooling = []
        # 2nd part of the classifier
        self.m_output_act = []
        # front-end
        self.m_frontend = []
        
        # confidence predictor
        self.m_conf = []

        # it can handle models with multiple front-end configuration
        # by default, only a single front-end
        for idx, (trunc_len, fft_n, lfcc_dim) in enumerate(zip(
                self.v_truncate_lens, self.fft_n, self.lfcc_dim)):
            
            fft_n_bins = fft_n // 2 + 1
            if self.lfcc_with_delta:
                lfcc_dim = lfcc_dim * 3
            
            self.m_transform.append(
                torch_nn.Sequential(
                    torch_nn.Conv2d(1, 64, [5, 5], 1, padding=[2, 2]),
                    nii_nn.MaxFeatureMap2D(),
                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),
                    torch_nn.Conv2d(32, 96, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),
                    torch_nn.BatchNorm2d(48, affine=False),

                    torch_nn.Conv2d(48, 96, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(48, affine=False),
                    torch_nn.Conv2d(48, 128, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),

                    torch.nn.MaxPool2d([2, 2], [2, 2]),

                    torch_nn.Conv2d(64, 128, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(64, affine=False),
                    torch_nn.Conv2d(64, 64, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),

                    torch_nn.Conv2d(32, 64, [1, 1], 1, padding=[0, 0]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.BatchNorm2d(32, affine=False),
                    torch_nn.Conv2d(32, 64, [3, 3], 1, padding=[1, 1]),
                    nii_nn.MaxFeatureMap2D(),
                    torch_nn.MaxPool2d([2, 2], [2, 2]),
                    
                    torch_nn.Dropout(0.7)
                )
            )

            self.m_before_pooling.append(
                torch_nn.Sequential(
                    nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32),
                    nii_nn.BLSTMLayer((lfcc_dim//16) * 32, (lfcc_dim//16) * 32)
                )
            )

            if self.v_emd_dim is None:
                self.v_emd_dim = (lfcc_dim // 16) * 32
            else:
                assert self.v_emd_dim == (lfcc_dim//16) * 32, "v_emd_dim error"

            self.m_output_act.append(
                torch_nn.Linear((lfcc_dim // 16) * 32, self.v_out_class)
            )
            
            self.m_conf.append(
                torch_nn.Sequential(
                    torch_nn.Linear((lfcc_dim // 16) * 32, 128),
                    torch_nn.Tanh(),
                    torch_nn.Linear(128, 1),
                    torch_nn.Sigmoid()
                    )
            )

            self.m_frontend.append(
                nii_front_end.LFCC(self.frame_lens[idx],
                                   self.frame_hops[idx],
                                   self.fft_n[idx],
                                   self.m_target_sr,
                                   self.lfcc_dim[idx],
                                   with_energy=True)
            )

        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)
        self.m_before_pooling = torch_nn.ModuleList(self.m_before_pooling)
        self.m_conf = torch_nn.ModuleList(self.m_conf)
        # output 
        

        self.m_loss = torch_nn.NLLLoss()
        self.m_lambda = torch_nn.Parameter(torch.tensor([0.1]), 
                                           requires_grad=False)
        self.m_budget = 0.5
        # done
        return
    
    def prepare_mean_std(self, in_dim, out_dim, args, 
                         prj_conf, data_mean_std=None):
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


    def _front_end(self, wav, idx, trunc_len, datalength):
        """ simple fixed front-end to extract features
        
        input:
        ------
          wav: waveform
          idx: idx of the trial in mini-batch
          trunc_len: number of frames to be kept after truncation
          datalength: list of data length in mini-batch

        output:
        -------
          x_sp_amp: front-end featues, (batch, frame_num, frame_feat_dim)
        """
        
        with torch.no_grad():
            x_sp_amp = self.m_frontend[idx](wav.squeeze(-1))

        # return
        return x_sp_amp

    def _compute_embedding(self, x, datalength):
        """ definition of forward method 
        Assume x (batchsize, length, dim)
        Output x (batchsize * number_filter, output_dim)
        """
        # resample if necessary
        #x = self.m_resampler(x.squeeze(-1)).unsqueeze(-1)
        
        # number of sub models
        batch_size = x.shape[0]

        # buffer to store output scores from sub-models
        output_emb = torch.zeros([batch_size * self.v_submodels, 
                                  self.v_emd_dim], 
                                  device=x.device, dtype=x.dtype)
        
        # compute scores for each sub-models
        for idx, (fs, fl, fn, trunc_len, m_trans, m_be_pool, m_output) in \
            enumerate(
                zip(self.frame_hops, self.frame_lens, self.fft_n, 
                    self.v_truncate_lens, self.m_transform, 
                    self.m_before_pooling, self.m_output_act)):
            
            # extract front-end feature
            x_sp_amp = self._front_end(x, idx, trunc_len, datalength)

            # compute scores
            #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
            #  2. compute hidden features
            hidden_features = m_trans(x_sp_amp.unsqueeze(1))

            #  3. (batch, channel, frame//N, feat_dim//N) ->
            #     (batch, frame//N, channel * feat_dim//N)
            #     where N is caused by conv with stride
            hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
            frame_num = hidden_features.shape[1]
            hidden_features = hidden_features.view(batch_size, frame_num, -1)
            
            #  4. pooling
            #  4. pass through LSTM then summing
            hidden_features_lstm = m_be_pool(hidden_features)

            #  5. pass through the output layer
            tmp_emb = (hidden_features_lstm + hidden_features).mean(1)
            
            output_emb[idx * batch_size : (idx+1) * batch_size] = tmp_emb

        return output_emb

    def _compute_logit(self, feature_vec, inference=False):
        """
        """
        # number of sub models
        batch_size = feature_vec.shape[0]

        # buffer to store output scores from sub-models
        output_act = torch.zeros(
            [batch_size * self.v_submodels, self.v_out_class], 
            device=feature_vec.device, dtype=feature_vec.dtype)

        # compute scores for each sub-models
        for idx, m_output in enumerate(self.m_output_act):
            tmp_emb = feature_vec[idx*batch_size : (idx+1)*batch_size]
            output_act[idx*batch_size : (idx+1)*batch_size] = m_output(tmp_emb)
        
        # feature_vec is [batch * submodel, output-class]
        return output_act

    def _compute_score(self, logits):
        """
        """
        # [batch * submodel, output-class], logits
        # [:, 1] denotes being bonafide
        if logits.shape[1] == 2:
            return logits[:, 1] - logits[:, 0]
        else:
            return logits[:, -1]

    def _compute_conf(self, feature_vec, inference=False):
        """
        """
        # number of sub models
        batch_size = feature_vec.shape[0]

        # buffer to store output scores from sub-models
        conf_score = torch.zeros(
            [batch_size * self.v_submodels, 1], 
            device=feature_vec.device, dtype=feature_vec.dtype)

        # compute scores for each sub-models
        for idx, m_output in enumerate(self.m_conf):
            tmp_emb = feature_vec[idx*batch_size : (idx+1)*batch_size]
            conf_score[idx*batch_size : (idx+1)*batch_size] = m_output(tmp_emb)
        
        return conf_score.squeeze(1)

    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def _clamp_prob(self, input_prob, clamp_val=1e-12):
        return torch.clamp(input_prob, 0.0 + clamp_val, 1.0 - clamp_val)
    
    def _loss(self, logits, targets, conf_scores):
        """
        """
        with torch.no_grad():
            index = torch.zeros_like(logits)
            index.scatter_(1, targets.data.view(-1, 1), 1)
        
        
        # clamp the probablity and conf scores
        prob = self._clamp_prob(torch_nn_func.softmax(logits, dim=1))
        conf_tmp = self._clamp_prob(conf_scores)
        
        # mixed log probablity
        log_prob = torch.log(
            prob * conf_tmp.view(-1, 1) +  (1 - conf_tmp.view(-1, 1)) * index)
            
        loss = self.m_loss(log_prob, targets)
        loss2 = -torch.log(conf_scores).mean()
        loss = loss + self.m_lambda * loss2
        
        if self.m_budget > loss2:
            self.m_lambda.data = self.m_lambda / 1.01 
        else:
            self.m_lambda.data = self.m_lambda / 0.99

        return loss.mean()

    def forward(self, x, fileinfo):
        
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]

        # too short sentences
        if not self.training and x.shape[1] < 3000:
            targets = self._get_target(filenames)
            for filename, target in zip(filenames, targets):
                print("Output, %s, %d, %f, %f" % (
                    filename, target, 0.0, 0.0))
            return None
        
        # for training, do balanced batch
        if self.training:
            target = self._get_target(filenames)
            bona_indx = np.argwhere(np.array(target) > 0)[:, 0]
            spoof_indx = np.argwhere(np.array(target) == 0)[:, 0]
            num = np.min([len(bona_indx), len(spoof_indx)])
            trial_indx = np.concatenate([bona_indx[0:num], spoof_indx[0:num]])

            if len(trial_indx) == 0:
                x_ = x
                flag_no_data = True
            else:
                filenames = [filenames[x] for x in trial_indx]
                datalength = [datalength[x] for x in trial_indx]
                x_ = x[trial_indx]
                flag_no_data = False
        else:
            x_ = x

        feature_vec = self._compute_embedding(x_, datalength)
        logits = self._compute_logit(feature_vec)
        conf_scores = self._compute_conf(feature_vec)
        
        if self.training:
            # target
            target = self._get_target(filenames)
            target_vec = torch.tensor(target, device=x.device, dtype=torch.long)
            target_vec = target_vec.repeat(self.v_submodels)
            
            # randomly set half of the conf data to 1
            b = torch.bernoulli(torch.zeros_like(conf_scores).uniform_(0, 1))
            conf_scores = conf_scores * b + (1 - b)

            loss = self._loss(logits, target_vec, conf_scores)

            if flag_no_data:
                return loss * 0
            else:
                return loss

        else:
            scores = self._compute_score(logits)
            targets = self._get_target(filenames)
            for filename, target, score, conf in \
                zip(filenames, targets, scores, conf_scores):
                print("Output, %s, %d, %f, %f" % (
                    filename, target, score.item(), conf.item()))
            # don't write output score as a single file
            return None

    def get_embedding(self, x, fileinfo):
        filenames = [nii_seq_tk.parse_filename(y) for y in fileinfo]
        datalength = [nii_seq_tk.parse_length(y) for y in fileinfo]
        feature_vec = self._compute_embedding(x, datalength)
        return feature_vec


class Loss():
    """ Wrapper to define loss function 
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

    
