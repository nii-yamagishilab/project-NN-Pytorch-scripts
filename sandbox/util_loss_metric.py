#!/usr/bin/env python
"""
util_loss_metric

Loss functions or metrics


References
[1] Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. 
    Energy-Based Out-of-Distribution Detection. 
    In Proc. NIPS, 33:21464–21475. 2020.
[2] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, 
    Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan 
    Supervised Contrastive Learning. Proc.NIPS. 2020.
[3] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 
    Mixup: Beyond Empirical Risk Minimization. In Proc. ICLR. 2018.
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
__copyright__ = "Copyright 2022, Xin Wang"


#####################
# negative energy [1]
#####################

def neg_energy(logits, temperature=1):
    """ neg_eng = neg_energy(logits, temperature=1)

    neg_eng[x] =  -T \log \sum_y \exp (logits[x, y] / T)

    See [1]

    input
    -----
      logits: tensor, shape (batch, dim)
      temperature: float, temperature hyperparameter
    
    output
    ------
      neg_eng: tensor, shape (batch,)
    """
    eng = - temperature * torch.logsumexp(logits / temperature, dim=1)
    return eng


def neg_energy_reg_loss(energy, margin_in, margin_out, flag_in):
    """ loss = neg_energy_reg_loss(energy, margin_in, margin_out, flag_in)

    See [1] eqs.(8-9)
    
    input
    -----
      energy: tensor, any shape is OK
      margin_in: float, margin for the in-dist. data
      margin_out: float, margin for the out-dist. data
      flag_in: bool, if the input data is in-dist. data

    output
    ------
      loss: scalar
    """
    if flag_in:
        loss = torch.pow(torch_nn_func.relu(energy - margin_in), 2).mean()
    else:
        loss = torch.pow(torch_nn_func.relu(margin_out - energy), 2).mean()
    return loss


#####################
# supervised contrastive loss [2]
#####################

def supcon_loss(input_feat, 
               labels = None, mask = None, sim_metric = None, 
               t=0.07, contra_mode='all', length_norm=False):
    """
    loss = SupConLoss(feat, 
                      labels = None, mask = None, sim_metric = None, 
                      t=0.07, contra_mode='all')
    input
    -----
      feat: tensor, feature vectors z [bsz, n_views, ...].
      labels: ground truth of shape [bsz].
      mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
      sim_metric: func, function to measure the similarity between two 
            feature vectors
      t: float, temperature
      contra_mode: str, default 'all'
         'all': use all data in class i as anchors
         'one': use 1st data in class i as anchors
      length_norm: bool, default False
          if True, l2 normalize feat along the last dimension

    output
    ------
      A loss scalar.
        
    Based on https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.

    Example:
      feature = torch.rand([16, 2, 1000], dtype=torch.float32)
      feature = torch_nn_func.normalize(feature, dim=-1)
      label = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 1, 1, 1, 1, 1], 
               dtype=torch.long)
      loss = supcon_loss(feature, labels=label)
    """
    if length_norm:
        feat = torch_nn_func.normalize(input_feat, dim=-1)
    else:
        feat = input_feat
        
    # batch size
    bs = feat.shape[0]
    # device
    dc = feat.device
    # dtype
    dt = feat.dtype
    # number of view
    nv = feat.shape[1]
    
    # get the mask
    # mask[i][:] indicates the data that has the same class label as data i
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(bs, dtype=dt, device=dc)
    elif labels is not None:
        labels = labels.view(-1, 1)
        if labels.shape[0] != bs:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).type(dt).to(dc)
    else:
        mask = mask.type(dt).to(dc)
    
    # prepare feature matrix
    # -> (num_view * batch, feature_dim, ...)
    contrast_feature = torch.cat(torch.unbind(feat, dim=1), dim=0)
    
    # 
    if contra_mode == 'one':
        # (batch, feat_dim, ...)
        anchor_feature = feat[:, 0]
        anchor_count = 1
    elif contra_mode == 'all':
        anchor_feature = contrast_feature
        anchor_count = nv
    else:
        raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
    
    # compute logits
    # logits_mat is a matrix of size [num_view * batch, num_view * batch]
    # or [batch, num_view * batch]
    if sim_metric is not None:
        logits_mat = torch.div(
            sim_metric(anchor_feature, contrast_feature), t)
    else:
        logits_mat = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), t)
    
    # mask based on the label
    # -> same shape as logits_mat 
    mask_ = mask.repeat(anchor_count, nv)
    # mask on each data itself (
    self_mask = torch.scatter(
        torch.ones_like(mask_), 1, 
        torch.arange(bs * anchor_count).view(-1, 1).to(dc), 
        0)
    
    # 
    mask_ = mask_ * self_mask
    
    # for numerical stability, remove the max from logits
    # see https://en.wikipedia.org/wiki/LogSumExp trick
    # for numerical stability
    logits_max, _ = torch.max(logits_mat * self_mask, dim=1, keepdim=True)
    logits_mat_ = logits_mat - logits_max.detach()

    # compute log_prob
    exp_logits = torch.exp(logits_mat_ * self_mask) * self_mask
    log_prob = logits_mat_ - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask_ * log_prob).sum(1) / mask_.sum(1)

    # loss
    loss = - mean_log_prob_pos
    loss = loss.view(anchor_count, bs).mean()

    return loss


################
# Mixup
################

class MixUpCE(torch_nn.Module):
    def __init__(self, weight = None):
        super(MixUpCE, self).__init__()
        self.m_loss1 = torch_nn.CrossEntropyLoss(weight=weight,reduction='none')
        self.m_loss2 = torch_nn.CrossEntropyLoss(weight=weight,reduction='none')
        return

    def forward(self, logits, y1, y2=None, gammas=None):
        """ loss = MixUpCE.forward(logits, y1, y2, gammas)

        This API computes the mixup cross-entropy. 
        Logits is assumed to be f( gammas * x1 + (1-gammas) * x2).
        Thus, this API only compute the CE:
          gammas * Loss(logits, y1) + (1 - gammas) * Loss(logits, y2)
        
        Note that if y2 and gammas are None, it uses common CE

        input
        -----
          logits: tensor, (batch, dim)
          y1: tensor, (batch, )
          y2: tensor, (batch, )
          gammas: tensor, (batch, )
        

        output
        ------
          loss: scalar  
        """
        if y2 is None and gammas is None:
            loss_val = self.m_loss1(logits, y1)
        else:
            loss_val = gammas * self.m_loss1(logits, y1) 
            loss_val += (1-gammas) * self.m_loss2(logits, y2) 
        return loss_val.mean()



#####################
# Distillation related
#####################

def kld_distill(logits_s, logits_t, temp=20):
    """ KLD-based distillation loss

    input
    -----
      logits_s: tensor, (batch, ..., dim), student output logits
                where dim is #. output categories
      logits_t: tensor, (batch, ..., dim), teacher output logits
      temp: float, temperature, default=20

    output
    ------
      loss: scalar
    """
    
    KD_loss = torch_nn_func.kl_div(
        torch_nn_func.log_softmax(logits_s / temp, dim = -1),
        torch_nn_func.log_softmax(logits_t / temp, dim = -1),
        reduction = 'batchmean',
        log_target = True) * temp * temp

    return KD_loss

#####################
# Rank consistency 
#####################
def rank_consistency(x, metric = None, anchor = None, diff_mat = None):
    """loss = rank_consistency(x, metric)
    
    input
    -----
      x: tensor, (batch, D1, D2 ...)
      metric: a function or None
      
              This function must be f(x1, x2) -> scalar
              where x1 and x2 are in shape (D1, D2 ...)
              
              if None, negative cosine similarity for
              x1 and x2 of shape (D1, )
      anchor: tensor, (batch, D1, D2, ...), as anchor
              or None

              If None, one of difference vector in the
              matrix will be selected as anchor
    
      diff_mat: tensor, (batch, batch, D1, D2 ...)
                of None
                if diff_mat is provided, x will be ignored
    output
    ------
      loss: scalar, loss value
      
    Example
    -------
      >> x = torch.randn(4, 10)
      >> x[1] = x[0] + 1.0
      >> x[2] = x[0] + 2.0
      >> x[3] = x[0] + 3.0
      >> rank_consistency(x)
      tensor(-1.)
    """
    
    if diff_mat is None:
        # (batch, batch, dim)
        # diff_mat[i, j]  = x[j] - x[i]
        diff_mat = x - x.unsqueeze(1)
        
    # batch size
    bs = diff_mat.shape[0]

    # loss to be accumulated
    loss = 0.0
    
    # metric
    if metric is None:
        # default use negative cosine_similarity
        metric = lambda x1, x2: -torch_nn_func.cosine_similarity(x1, x2, dim=0)
    
    #
    if bs < 3:
        return loss

    # get anchor
    if anchor is None:
        # choose the diff_mat[1, 0] as the anchor
        anchor_row_idx = 1
        anchor_col_idx = 0
        anchor = diff_mat[anchor_row_idx, anchor_col_idx]
    else:
        # anchor is provided externally
        anchor_row_idx = -1
        anchor_col_idx = -1


    # loop over the matrix, compare the off-diagnoal elements 
    # with the anchor
    count = 0.0
    for col_idx in np.arange(bs-1):
        for row_idx in np.arange(col_idx+1, bs):
            if col_idx == anchor_col_idx and anchor_row_idx == row_idx:
                continue
            loss += metric(anchor, diff_mat[row_idx, col_idx])
            count += 1
            
    loss = loss / count    
    return loss


def rank_consistency_v2(x, metric = None, diff_mat = None):
    """loss = rank_consistency_v2(x, metric)
    
    input
    -----
      x: tensor, (batch, D1, D2 ...)
      metric: a function or None
      
              This function must be f(x1, x2) -> scalar
              where x1 and x2 are in shape (D1, D2 ...)
              
              if None, negative cosine similarity for
              x1 and x2 of shape (D1, )
      
      diff_mat: tensor, (batch, batch, D1, D2 ...)
                of None
                if diff_mat is provided, x will be ignored
    output
    ------
      loss: scalar, loss value
      
    Example
    -------
      >> x = torch.randn(4, 10)
      >> x[1] = x[0] + 1.0
      >> x[2] = x[0] + 2.0
      >> x[3] = x[0] + 3.0
      >> metric = lambda x1, x2: \
         torch_nn_func.margin_ranking_loss(x1, x2, torch.ones_like(x1), 0.1)
      >> rank_consistencyV2(x, metric)
      tensor(.0)
    """
    
    if diff_mat is None:
        # (batch, batch, dim)
        # diff_mat[i, j]  = x[j] - x[i]
        diff_mat = x - x.unsqueeze(1)
    
    # batch size
    bs = diff_mat.shape[0]

    # loss to be accumulated
    loss = 0.0
    
    # metric
    if metric is None:
        # default use margin_ranking_loss
        metric = lambda x1, x2: torch_nn_func.margin_ranking_loss(
            x1, x2, torch.ones_like(x1), 0.1)
    
    #
    if bs < 3:
        return loss

    count = 0.0
    # loop over the matrix, column first
    for col_idx in np.arange(bs-2):
        for row_idx in np.arange(col_idx+2, bs):
            # assume diff[i, j] should be diff[i-1, j]
            loss += metric(diff_mat[row_idx-1, col_idx], 
                           diff_mat[row_idx, col_idx])
            count += 1
            
    # loop over the matrix, column first
    for row_idx in np.arange(2, bs):
        for col_idx in np.arange(1, row_idx):
            # assume diff[i, j] should be diff[i, j-1]
            loss += metric(diff_mat[row_idx, col_idx], 
                           diff_mat[row_idx, col_idx-1])
            count += 1
            
    loss = loss / count    
    return loss


def rank_consistency_v3(x, metric = None):
    """loss = rank_consistency_v3(x, metric)
    
    input
    -----
      x: tensor, (batch, D1, D2 ...)
      metric: a function or None
      
              This function must be f(x1, x2) -> scalar
              where x1 and x2 are in shape (D1, D2 ...)
              
              if None, negative cosine similarity for
              x1 and x2 of shape (D1, )

    output
    ------
      loss: scalar, loss value
      
    Example
    -------
      >> x = torch.randn(4, 10)
      >> x[1] = x[0] + 1.0
      >> x[2] = x[0] + 2.0
      >> x[3] = x[0] + 3.0
      >> metric = lambda x1, x2: \
         torch_nn_func.margin_ranking_loss(x1, x2, torch.ones_like(x1), 0.1)
      >> rank_consistency_v3(x, metric)
      tensor(.0)
    """
    # batch size
    bs = x.shape[0]

    # loss to be accumulated
    loss = 0.0
    
    # metric
    if metric is None:
        # default use margin_ranking_loss
        # x1 should be ranked higher
        metric = lambda x1, x2: torch_nn_func.margin_ranking_loss(
            x1, x2, torch.ones_like(x1), 0.1)
    
    #
    if bs < 2:
        return loss

    count = 0.0
    # loop over the rows
    for row_idx1 in np.arange(1, bs):
        for row_idx2 in np.arange(0, row_idx1):
            loss += metric(x[row_idx1], 
                           x[row_idx2])
            count += 1
                        
    loss = loss / count    
    return loss

if __name__ == "__main__":
    print("loss and metric")
