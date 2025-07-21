#!/usr/bin/env python
"""wrapper for speaker embedding extractors
"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
from logging import getLogger

import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.augment import time_domain

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, Xin Wang"


logger = getLogger(__name__)


class Pretrained_speechbrain(torch_nn.Module):
    """Pretrained speaker embedding API from speechbrain

    See usage in https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
    """
    def __init__(self, config, run_opts):
        super(Pretrained_speechbrain, self).__init__()

        #
        name = config['name']
        if name == 'ecapa_tdnn':
            src = "speechbrain/spkrec-ecapa-voxceleb"
            # accepted sampling rate of this model
            self.tar_sr = 16000
            self.l2_norm = True
        else:
            logger.info('Unsupported speaker embedding {:s}'.format(name))
            sys.exit(1)

        # wrap run_opts from main into a dictionary
        # speechbrain/inference/interfaces.py: 539 -> kawargs -> cls (EncoderClassifier/Pretrained)
        # speechbrain/inference/interfaces.py: 214 -> kawargs will be resolved into run_opts
        #  run_opts here will overwrite the run_opt_defaults in interfaces
        # without run_opts, pretrained encoder will be put into cpu the default 
        kawrgs = {'run_opts': run_opts}
        self.classifier = EncoderClassifier.from_hparams(source=src, **kawrgs)
        self.classifier.eval()

        # dimension of output vector
        self.emb_dim = config['output_dim']

        # add re-sample
        if config['input_sr'] != self.tar_sr:
            self.resampler = time_domain.Resample(
                orig_freq=config['input_sr'], new_freq=self.tar_sr)
        else:
            self.resampler = torch_nn.Identity()

        # max duration (in case out-of-mem)
        if 'max_wav_dur' in config:
            self.max_len = config['max_wav_dur'] * self.tar_sr
        else:
            self.max_len = -1
            
        return

    def encode_batch(self, signal):
        with torch.no_grad():
            # resample and set length
            if self.max_len == -1:
                signal_ = self.resampler(signal)
            else:
                signal_ = self.resampler(signal)[:, :self.max_len]
            emb = self.classifier.encode_batch(signal_)

            # length norm
            if self.l2_norm:
                emb = torch.nn.functional.normalize(emb, dim=-1)
            
        return emb



class SpkEmbMerger(torch_nn.Module):
    """A wrapper to merge speaker embedding with other input features
    """
    def __init__(self, spk_dim, feat_dim, out_dim, merge_method = 'sum'):
        """
        Args
        ----
          spk_dim:          int, dimension of speaker embedding
          feat_dim:         int, dimension of other features
          out_dim:          int, dimension of output feature
          merge_method:     str, method of merging the speaker and other feats
                            it can be concat: concat the two
                            sum: transform spk_dim to feat_dim then sum
                            default: concat
        """
        super(SpkEmbMerger, self).__init__()

        self.spk_dim = spk_dim
        self.feat_dim = feat_dim
        self.out_dim = out_dim
        self.method = merge_method

        if self.method == 'sum':
            self.m_linear = torch_nn.Linear(spk_dim, feat_dim)
            torch_nn.init.xavier_uniform_(
                self.m_linear.weight,
                gain=torch_nn.init.calculate_gain('linear'))
        elif self.method == 'concat':
            self.m_linear = torch_nn.Identity()
        else:
            logger.info("Unknown merging method {:s}".format(merge_method))
            sys.exit(1)
        return

    def _forward_sum(self, spk_emb, feat, data_length):

        # transform spk_emb
        spk_emb_ = self.m_linear(spk_emb)

        # make dimension to be compatible
        data_len = feat.shape[1]
        if spk_emb_.ndim == 3:
            # (batch, 1, dim)
            # duplicate speaker embedding across time
            spk_emb_ = spk_emb_.repeat(1, data_len, 1)
        elif spk_emb_.ndim == 2:
            # (batch, dim))
            spk_emb_ = spk_emb_.unsqueeze(1).repeat(1, data_len, 1)
        else:
            logger.info("Spk_emb has {:d} dimensions".format(spk_emb.ndim))
            sys.exit(1)

        # mask spk_emb
        for idx, data_len in enumerate(data_length):
            spk_emb_[idx, data_len:] *= 0

        return feat + spk_emb_
        
    
    def _forward_concat(self, spk_emb, feat, data_length):
        
        data_len = feat.shape[1]

        if spk_emb.ndim == 3:
            # (batch, 1, dim)
            # duplicate speaker embedding across time
            spk_emb_ = spk_emb.repeat(1, data_len, 1)
        elif spk_emb.ndim == 2:
            # (batch, dim))
            spk_emb_ = spk_emb.unsqueeze(1).repeat(1, data_len, 1)
        else:
            logger.info("Spk_emb has {:d} dimensions".format(spk_emb.ndim))
            sys.exit(1)

        # mask spk_emb
        for idx, data_len in enumerate(data_length):
            spk_emb_[idx, data_len:] *= 0
                
        # concate with feedback input
        output = torch.concat([feat, spk_emb_], dim=-1)

        return output

    def forward(self, spk_emb, feat, data_length):
        if self.method == 'sum':
            return self._forward_sum(spk_emb, feat, data_length)
        elif self.method == 'concat':
            return self._forward_concat(spk_emb, feat, data_length)
        else:
            logger.info("Unknown merging method {:s}".format(merge_method))
            sys.exit(1)

        
if __name__ == "__main__":
    print("speaker embedding")
