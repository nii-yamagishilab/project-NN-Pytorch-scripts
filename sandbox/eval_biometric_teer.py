#!/usr/bin/env python
"""
This is the implementation of t-EER, 
Just wrap over https://github.com/TakHemlata/T-EER

@ARTICLE {Kinnunen2023-tEER,
author = {T. H. Kinnunen and K. Lee and H. Tak and N. Evans and A. Nautsch},
journal = {{IEEE} Transactions on Pattern Analysis and Machine Intelligence},
title = {t-EER: Parameter-Free Tandem Evaluation of Countermeasures and
Biometric Comparators (to appear)},
doi = {10.1109/TPAMI.2023.3313648},
year = {2023},
publisher = {IEEE Computer Society},
}

MIT License
Copyright (c) 2023 Hemlata
"""

import sys
import os
import numpy as np
import scipy.stats as stats
from scipy import special


def compute_Pmiss_Pfa_Pspoof_curves(tar_scores, non_scores, spf_scores):

    # Concatenate all scores and designate arbitrary labels 1=target, 0=nontarget, -1=spoof
    all_scores = np.concatenate((tar_scores, non_scores, spf_scores))
    labels = np.concatenate((np.ones(tar_scores.size), np.zeros(non_scores.size), -1*np.ones(spf_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Cumulative sums
    tar_sums    = np.cumsum(labels==1)
    non_sums    = np.cumsum(labels==0)
    spoof_sums  = np.cumsum(labels==-1)

    Pmiss       = np.concatenate((np.atleast_1d(0), tar_sums / tar_scores.size))
    Pfa_non     = np.concatenate((np.atleast_1d(1), 1 - (non_sums / non_scores.size)))
    Pfa_spoof   = np.concatenate((np.atleast_1d(1), 1 - (spoof_sums / spf_scores.size)))
    thresholds  = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return Pmiss, Pfa_non, Pfa_spoof, thresholds


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds



def compute_t_eer(target_scores, nontarget_scores, spoof_scores, rho):
    """
    Args
    ----
      target_scores:    np.array, (L_t, 2), where L_t is 
                        the number of target bona fide trial scores
                        target_score[:, 0] is the ASV score, 
                        target_score[:, 1] is the CM score.
      nontarget_scores: np.array, (L_n, 2), where L_n is 
                        the number of nontarget bona fide trial scores
      spoof_scores:     np.array, (L_s, 2), where L_s is 
                        the number of spoofed fide trial scores
      rho:              float, value of prior rho.    
                        
    Output
    ------
      teer_path:        np.array, (L_t + L_n + L_s + 1, 2),
                        teer_path[i, 0] is the ASV threshold
                        teer_path[i, 1] is the CM threshold for t-EER
      teer_val:         np.array, (L_t + L_n + L_s + 1),
                        t-eer values
                        
      con_teer:         scalar, value of the concurrent t-EER
      con_teer_tau:     np.array(2), the ASV and CM thresholds for concurrent t-EER
    """
    assert rho >= 0.0 and rho <= 1.0, "Value of rho should be in [0, 1]"
    
    # compute ASV P_miss, P_fa_non, P_fa_spoof
    Pmiss_ASV, Pfa_non_ASV, Pfa_spoof_ASV, tau_ASV_array = compute_Pmiss_Pfa_Pspoof_curves(
        target_scores[:, 0], nontarget_scores[:, 0], spoof_scores[:, 0])
    
    # compte CM P_miss and P_fa
    Pmiss_CM, Pfa_CM, tau_CM_array = compute_det_curve(
        np.concatenate([target_scores[:,1], nontarget_scores[:,1]]), spoof_scores[:,1])
    
    # create output buffer
    teer_path = np.zeros([tau_ASV_array.shape[0], 2])
    teer_val = np.zeros_like(tau_ASV_array)
    #
    con_teer_gap = np.inf
    con_teer_ASV_idx = None
    con_teer_CM_idx = None
    
    for tau_ASV_idx, tau_ASV in enumerate(tau_ASV_array):
        
        # P_tdm_miss(\tau_cm), given \tau_ASV
        P_tdm_miss = Pmiss_ASV[tau_ASV_idx] + Pmiss_CM - Pmiss_ASV[tau_ASV_idx] * Pmiss_CM
        
        # P_tdm_non(\tau_cm), given \tau_asv and \rho
        P_tdm_fa_rho = ( (1.0 - rho) * Pfa_non_ASV[tau_ASV_idx] * (1.0 - Pmiss_CM) 
                        + rho * Pfa_spoof_ASV[tau_ASV_idx] * Pfa_CM
                       )
        
        # check whether it satisfy the criterion (eq.12)
        check_stat = (1.0 - rho) * Pfa_non_ASV[tau_ASV_idx] + rho * Pfa_spoof_ASV[tau_ASV_idx]
        
        if check_stat < Pmiss_ASV[tau_ASV_idx]:
            # not possible to find t-EER
            teer_path[tau_ASV_idx] = np.array([np.nan, np.nan])
            teer_val[tau_ASV_idx] = np.nan
            
        else:
            # find the \tau_CM for t-EER
            tau_CM_idx = np.argmin(np.abs(P_tdm_miss - P_tdm_fa_rho))
            teer_path[tau_ASV_idx] = np.array([tau_ASV, tau_CM_array[tau_CM_idx]])
            teer_val[tau_ASV_idx] = (P_tdm_miss[tau_CM_idx] + P_tdm_fa_rho[tau_CM_idx])/2.0
            
            # find the concurrent t-EER
            gap = np.abs(Pfa_non_ASV[tau_ASV_idx] * (1.0 - Pmiss_CM[tau_CM_idx]) 
                         - Pfa_spoof_ASV[tau_ASV_idx] * Pfa_CM[tau_CM_idx])
            if gap < con_teer_gap:
                con_teer_gap = gap
                con_teer_CM_idx = tau_CM_idx
                con_teer_ASV_idx = tau_ASV_idx
                
    if con_teer_CM_idx and con_teer_ASV_idx:
        con_teer = teer_val[con_teer_ASV_idx]
        con_teer_tau = teer_path[con_teer_ASV_idx]
    else:
        con_teer = np.nan
        con_teer_tau = np.array([np.nan, np.nan])
        
    return teer_path, teer_val, con_teer, con_teer_tau


if __name__ == "__main__":
    print(__doc__)
