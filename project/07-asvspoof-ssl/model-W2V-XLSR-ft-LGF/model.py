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
import sandbox.eval_asvspoof as nii_asvspoof 

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

############################
## FOR pre-trained MODEL
############################
import fairseq

class SSLModel(torch_nn.Module):    
    def __init__(self, cp_path, ssl_orig_output_dim):
        """ SSLModel(cp_path, ssl_orig_output_dim)
        
        Args:
          cp_path: string, path to the pre-trained SSL model
          ssl_orig_output_dim: int, dimension of the SSL model output feature
        """
        super(SSLModel, self).__init__()
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        # dimension of output from SSL model. This is fixed
        self.out_dim = ssl_orig_output_dim
        return

    def extract_feat(self, input_data):
        """ feature = extract_feat(input_data)
        Args:
          input_data: tensor, waveform, (batch, length)
        
        Return:
          feature: tensor, feature, (batch, frame_num, feat_dim)
        """
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            #self.model.eval()
            
        #with torch.no_grad():
        if True:   
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']
        return emb


#################
## Misc functions
#################
# A function to load in/out label for OOD detection. This is just a place holder
# in this project
g_attack_map = {}
def protocol_parse_general(protocol_filepaths, g_map, sep=' ', target_row=-1):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    The format is:
      SPEAKER  TRIAL_NAME  - SPOOF_TYPE TAG
      LA_0031 LA_E_5932896 - A13        spoof
      LA_0030 LA_E_5849185 - -          bonafide
    ...  
    input:
    -----
      protocol_filepath: string, path to the protocol file
      target_row: int, default -1, use line[-1] as the target label
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """
    data_buffer = nii_asvspoof.CustomDict(missing_value=True)
    
    if type(protocol_filepaths) is str:
        tmp = [protocol_filepaths]
    else:
        tmp = protocol_filepaths
        
    for protocol_filepath in tmp:
        if len(protocol_filepath) and os.path.isfile(protocol_filepath):
            with open(protocol_filepath, 'r') as file_ptr:
                for line in file_ptr:
                    line = line.rstrip('\n')
                    cols = line.split(sep)

                    if g_map:
                        try:
                            data_buffer[cols[1]] = g_map[cols[target_row]]
                        except KeyError:                    
                            data_buffer[cols[1]] = False
                    else:
                        data_buffer[cols[1]] = True
                    
    return data_buffer
    

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

        # load protocol for CM (if available)
        self.protocol_parser = nii_asvspoof.protocol_parse_general(protocol_f)

        # Load protocol for OOD (if available)
        self.in_out_parser = protocol_parse_general(protocol_f, g_attack_map, 
                                                    ' ', -2)
        # Working sampling rate
        #  torchaudio may be used to change sampling rate
        #self.m_target_sr = 16000
        
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
        
        # 
        self.v_feat_dim = [128]

        # number of sub-models (by default, a single model)
        self.v_submodels = len(self.v_feat_dim)

        # dimension of embedding vectors
        # here, the embedding is just the activation before sigmoid()
        self.v_emd_dim = None
        
        self.v_out_class = 2

        ####
        # create network
        ####
        # path to the pre-trained SSL model 
        ssl_path = os.path.dirname(__file__) + '/../../../SSL_pretrained/xlsr_53_56k.pt'
        ssl_orig_output_dim = 1024 
        self.m_ssl = SSLModel(ssl_path, ssl_orig_output_dim)
        
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
        for idx, v_feat_dim in enumerate(self.v_feat_dim):
                        
            # just a place holder
            self.m_transform.append(
                torch_nn.Sequential(
                    torch_nn.Identity()
                )
            )

            self.m_before_pooling.append(
                torch_nn.Sequential(
                    nii_nn.BLSTMLayer(v_feat_dim, v_feat_dim),
                    nii_nn.BLSTMLayer(v_feat_dim, v_feat_dim)
                )
            )

            if self.v_emd_dim is None:
                self.v_emd_dim = v_feat_dim
            else:
                assert self.v_emd_dim == v_feat_dim, "v_emd_dim error"

            self.m_output_act.append(
                torch_nn.Linear(v_feat_dim, self.v_out_class)
            )
            
            self.m_frontend.append(
                torch_nn.Linear(self.m_ssl.out_dim, v_feat_dim)
            )

        self.m_frontend = torch_nn.ModuleList(self.m_frontend)
        self.m_transform = torch_nn.ModuleList(self.m_transform)
        self.m_output_act = torch_nn.ModuleList(self.m_output_act)
        self.m_before_pooling = torch_nn.ModuleList(self.m_before_pooling)
        # output 
        

        self.m_loss = torch_nn.CrossEntropyLoss()
        self.m_temp = 1
        self.m_lambda = 0.
        self.m_e_m_in = -25.0
        self.m_e_m_out = -7.0
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


    def _front_end(self, wav, idx, datalength):
        """ simple fixed front-end to extract features
        
        input:
        ------
          wav: waveform
          idx: idx of the trial in mini-batch
          datalength: list of data length in mini-batch

        output:
        -------
          x_sp_amp: front-end featues, (batch, frame_num, frame_feat_dim)
        """
        
        #with torch.no_grad():
        x_ssl_feat = self.m_ssl.extract_feat(wav.squeeze(-1))

        # return
        return self.m_frontend[idx](x_ssl_feat)

    def _pretransform(self, x_sp_amp, m_trans):
        """ A wrapper on the self.m_transform part
        """
        # compute scores
        #  1. unsqueeze to (batch, 1, frame_length, fft_bin)
        #  2. compute hidden features
        #hidden_features = m_trans(x_sp_amp.unsqueeze(1))

        #  3. (batch, channel, frame//N, feat_dim//N) ->
        #     (batch, frame//N, channel * feat_dim//N)
        #     where N is caused by conv with stride
        #hidden_features = hidden_features.permute(0, 2, 1, 3).contiguous()
        #frame_num = hidden_features.shape[1]
        #hidden_features = hidden_features.view(batch_size, frame_num, -1)
            
        hidden_features = m_trans(x_sp_amp)
        return hidden_features

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
        output_emb = torch.zeros(
            [batch_size * self.v_submodels, self.v_emd_dim], 
            device=x.device, dtype=x.dtype)
        
        # compute scores for each sub-models
        for idx, (m_trans, m_be_pool, m_output) in \
            enumerate(
                zip(self.m_transform, self.m_before_pooling, 
                    self.m_output_act)):
            
            # extract front-end feature
            x_sp_amp = self._front_end(x, idx, datalength)

            # 1. 2. 3. steps in transform
            hidden_features = self._pretransform(x_sp_amp, m_trans)

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

    def _get_target(self, filenames):
        try:
            return [self.protocol_parser[x] for x in filenames]
        except KeyError:
            print("Cannot find target data for %s" % (str(filenames)))
            sys.exit(1)

    def _clamp_prob(self, input_prob, clamp_val=1e-12):
        return torch.clamp(input_prob, 0.0 + clamp_val, 1.0 - clamp_val)


    def _get_in_out_indx(self, filenames):
        in_indx = []
        out_indx = []
        for x, y in enumerate(filenames):
            if self.in_out_parser[y]:
                in_indx.append(x)
            else:
                out_indx.append(x)
        return np.array(in_indx), np.array(out_indx)

    def _energy(self, logits):
        """
        """
        # - T \log \sum_y \exp (logits[x, y] / T)
        eng = - self.m_temp * torch.logsumexp(logits / self.m_temp, dim=1)
        return eng
    
    def _loss(self, logits, targets, energy, in_indx, out_indx):
        """
        """
        
        # loss over cross-entropy on in-dist. data
        if len(in_indx):
            loss = self.m_loss(logits[in_indx], targets[in_indx])
        else:
            loss = 0

        # loss on energy of in-dist.data
        if len(in_indx):
            loss += self.m_lambda * torch.pow(
                torch_nn_func.relu(energy[in_indx] - self.m_e_m_in), 2).mean()
        
        # loss on energy of out-dist. data
        if len(out_indx):
            loss += self.m_lambda * torch.pow(
                torch_nn_func.relu(self.m_e_m_out - energy[out_indx]), 2).mean()

        return loss

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
            
        feature_vec = self._compute_embedding(x, datalength)
        logits = self._compute_logit(feature_vec)
        energy = self._energy(logits)
        in_indx, out_indx = self._get_in_out_indx(filenames)

        if self.training:
            # target
            target = self._get_target(filenames)
            target_vec = torch.tensor(target, device=x.device, dtype=torch.long)
            target_vec = target_vec.repeat(self.v_submodels)
            
            # loss
            loss = self._loss(logits, target_vec, energy, in_indx, out_indx)
            return loss

        else:
            scores = self._compute_score(logits)
            targets = self._get_target(filenames)
            for filename, target, score, energytmp in \
                zip(filenames, targets, scores, energy):
                print("Output, %s, %d, %f, %f" % (
                    filename, target, score.item(), -energytmp.item()))
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

    
