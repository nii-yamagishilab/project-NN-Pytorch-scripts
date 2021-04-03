#!/usr/bin/env python
"""
Functions for evaluation - asvspoof and related binary classification tasks

Python Function from min tDCF on asvspoof.org

All functions before tDCF_wrapper are licensed by Creative Commons.

----- License ----
This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc-sa/4.0/
or send a letter to
Creative Commons, 444 Castro Street, Suite 900,
Mountain View, California, 94041, USA.
------------------

"""

from __future__ import print_function
import os
import sys
import numpy as np
import core_scripts.data_io.io_tools as nii_io

def protocol_parse_asvspoof2019(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    The format is:
      SPEAKER  TRIAL_NAME  - SPOOF_TYPE TAG
      LA_0031 LA_E_5932896 - A13        spoof
      LA_0030 LA_E_5849185 - -          bonafide
    ...

    input:
    -----
      protocol_filepath: string, path to the protocol file
    
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> 1 (bonafide), 0 (spoof)
    """
    data_buffer = {}
    temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    for row in temp_buffer:
        if row[-1] == 'bonafide':
            data_buffer[row[1]] = 1
        else:
            data_buffer[row[1]] = 0
    return data_buffer


def protocol_parse_attack_label_asvspoof2019(protocol_filepath):
    """ Parse protocol of ASVspoof2019 and get bonafide/spoof for each trial
    The format is:
      SPEAKER  TRIAL_NAME  - SPOOF_TYPE TAG
      LA_0031 LA_E_5932896 - A13        spoof
      LA_0030 LA_E_5849185 - -          bonafide
    ...

    input:
    -----
      protocol_filepath: string, path to the protocol file
    
    output:
    -------
      data_buffer: dic, data_bufer[filename] -> attack type
    """
    data_buffer = {}
    temp_buffer = np.loadtxt(protocol_filepath, dtype='str')
    for row in temp_buffer:
        if row[-2] == '-':
            data_buffer[row[1]] = 'bonafide'
        else:
            data_buffer[row[1]] = row[-2]
    return data_buffer


def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), 
                             np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = (nontarget_scores.size - 
                            (np.arange(1, n_scores + 1) - tar_trial_sums))

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums/target_scores.size))
    # false rejection rates
    far = np.concatenate((np.atleast_1d(1), 
                          nontarget_trial_sums / nontarget_scores.size))  
    # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), 
                                 all_scores[indices]))  
    # Thresholds are the sorted scores
    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def compute_tDCF_legacy(
        bonafide_score_cm, spoof_score_cm, 
        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, print_cost=False):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker 
                                      (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss_asv   Cost of ASV falsely rejecting target.
                          Cfa_asv     Cost of ASV falsely accepting nontarget.
                          Cmiss_cm    Cost of CM falsely rejecting target.
                          Cfa_cm      Cost of CM falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, K.-A. Lee, H. Delgado, N. Evans, M. Todisco,
          M. Sahidullah, J. Yamagishi, D.A. Reynolds: "t-DCF: a Detection
          Cost Function for the Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification", Proc. Odyssey 2018: the
          Speaker and Language Recognition Workshop, pp. 312--319, 
          Les Sables d'Olonne,
          France, June 2018 
          https://www.isca-speech.org/archive/Odyssey_2018/pdfs/68.pdf)

      [2] ASVspoof 2019 challenge evaluation plan
          TODO: <add link>
    """


    # Sanity check of cost parameters
    if cost_model['Cfa_asv'] < 0 or cost_model['Cmiss_asv'] < 0 or \
            cost_model['Cfa_cm'] < 0 or cost_model['Cmiss_cm'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or \
       cost_model['Pspoof'] < 0 or \
       np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pmiss_spoof_asv is None:
        sys.exit('ERROR: you should provide miss rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan
    C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
         cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
    C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

    # Sanity check of the weights
    if C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

    # Normalized t-DCF
    tDCF_norm = tDCF / np.minimum(C1, C2)

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa_asv      = {:8.5f} (Cost of ASV falsely accepting a nontarget)'.format(cost_model['Cfa_asv']))
        print('   Cmiss_asv    = {:8.5f} (Cost of ASV falsely rejecting target speaker)'.format(cost_model['Cmiss_asv']))
        print('   Cfa_cm       = {:8.5f} (Cost of CM falsely passing a spoof to ASV system)'.format(cost_model['Cfa_cm']))
        print('   Cmiss_cm     = {:8.5f} (Cost of CM falsely blocking target utterance which never reaches ASV)'.format(cost_model['Cmiss_cm']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), s=CM threshold)')

        if C2 == np.minimum(C1, C2):
            print('   tDCF_norm(s) = {:8.5f} x Pmiss_cm(s) + Pfa_cm(s)\n'.format(C1 / C2))
        else:
            print('   tDCF_norm(s) = Pmiss_cm(s) + {:8.5f} x Pfa_cm(s)\n'.format(C2 / C1))

    return tDCF_norm, CM_thresholds



def compute_tDCF(
        bonafide_score_cm, spoof_score_cm, 
        Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,

      Speech waveform -> [CM] -> [ASV] -> decision

    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.

    INPUTS:

      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.

                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss       Cost of tandem system falsely rejecting target speaker.
                          Cfa         Cost of tandem system falsely accepting nontarget speaker.
                          Cfa_spoof   Cost of tandem system falsely accepting spoof.

      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?

    OUTPUTS:

      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).

    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """


    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon']*cost_model['Cfa']*Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;


    # Sanity check of the weights
    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(C0/tDCF_default, C1/tDCF_default, C2/tDCF_default))
        print('     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

    return tDCF_norm, CM_thresholds


def tDCF_wrapper(bonafide_cm_scores, spoof_cm_scores, 
                 tar_asv_scores=None, non_asv_scores=None, 
                 spoof_asv_scores=None, 
                 flag_verbose=False, flag_legacy=True):
    """ 
    mintDCF, eer, eer_thre = tDCF_wrapper(bonafide_cm_scores, spoof_cm_scores, 
                 tar_asv_scores=None, non_asv_scores=None, 
                 spoof_asv_scores=None, flag_verbose=False, flag_legacy=True)
    
    
    input
    -----
      bonafide_cm_scores: np.array of bona fide scores
      spoof_cm_scores: np.array of spoof scores
      tar_asv_scores: np.array of ASV target scores, or None
      non_asv_scores: np.array of ASV non-target scores, or None
      spoof_asv_scores: np.array of ASV spoof trial scores, or None,
      flag_verbose: print detailed messages
      flag_legacy: True: use legacy min-tDCF in ASVspoof2019
                   False: use min-tDCF revised

    output
    ------
      mintDCF: scalar,  value of min-tDCF
      eer: scalar, value of EER
      eer_thre: scalar, value of threshold corresponding to EER
    
    """
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }


    # read provided ASV scores
    if tar_asv_scores is None or non_asv_scores is None or \
       spoof_asv_scores is None:
        file_name = os.path.dirname(__file__)+ \
            '/data/asvspoof2019/ASVspoof2019.LA.asv.eval.gi.trl.scores.bin'
        data = nii_io.f_read_raw_mat(file_name, 2)
        tar_asv_scores = data[data[:, 1] == 2, 0]
        non_asv_scores = data[data[:, 1] == 1, 0]
        spoof_asv_scores = data[data[:, 1] == 0, 0]
    

    eer_asv, asv_threshold = compute_eer(tar_asv_scores, non_asv_scores)
    eer_cm, eer_threshold = compute_eer(bonafide_cm_scores, spoof_cm_scores)

    
    [Pfa_asv,Pmiss_asv,Pmiss_spoof_asv,Pfa_spoof_asv] = obtain_asv_error_rates(
        tar_asv_scores, non_asv_scores, spoof_asv_scores, asv_threshold)
    
    if flag_legacy:
        tDCF_curve, CM_thresholds = compute_tDCF_legacy(
            bonafide_cm_scores, spoof_cm_scores, 
            Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, flag_verbose)
    else:
        tDCF_curve, CM_thresholds = compute_tDCF(
            bonafide_cm_scores, spoof_cm_scores, 
            Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, flag_verbose)
    
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]
    
    return min_tDCF, eer_cm, eer_threshold


def ASVspoof2019_evaluate(bonafide_cm_scores, bonafide_cm_file_names,
                          spoof_cm_scores, spoof_cm_file_names, verbose=False,
                          protocol_alternative=None):
    """ Decompose scores for each attack. For ASVspoof2019
    
    ASVspoof2019_decompose(bonafide_cm_scores, bonafide_cm_file_names,
                           spoof_cm_scores, spoof_cm_file_names, verbose=False)
    input
    -----
      bonafide_cm_scores: np.array of bonafide scores
      bonafide_cm_file_names: file name list corresponding to bonafide_cm_scores
      spoof_cm_scores: np.array of spoofed scores (all attack types)
      spoof_cm_file_names: file name list corresponding to spoof_cm_scores

      verbose: print information from tDCF computation (default: False)
      protocol_alternative: alternative protocol to ASVspoof2019 (default: None)
    output
    ------
      min_tDCF: np.array of min tDCF for each attack
      eer_cm: np.array of EER for each attack
      eer_threshold: np.array of threshold for EER (not min tDCF threshod)
      spoof_attack_types: list of attack types
    """
    if protocol_alternative is not None:
        # if provided alternative procotol, use it. 
        # this is for protocol tweaking
        file_name = protocol_alternative
    else:
        # official protocol
        file_name = os.path.dirname(__file__)+ '/data/asvspoof2019/protocol.txt'
    
    protocol_data = np.genfromtxt(file_name, 
                                  dtype=[('spk', 'U10'), ('file', 'U20'),
                                         ('misc', 'U5'), ('spoof', 'U5'),
                                         ('type','U10')], delimiter=" ")
    spoof_type_dic = {protocol_data[x][1]:protocol_data[x][3] for x in \
                      range(protocol_data.shape[0])}

    spoof_attack_types = list(set([x[3] for x in protocol_data]))
    spoof_attack_types.sort()
    
    # default set to -1
    min_tDCF = np.zeros([len(spoof_attack_types) + 1]) - 1
    eer_cm = np.zeros([len(spoof_attack_types) + 1]) - 1
    eer_threshold = np.zeros([len(spoof_attack_types) + 1])
    
    # decompose results
    decomposed_spoof_scores = []
    for idx, spoof_attack_type in enumerate(spoof_attack_types):
        tmp_spoof_scores = [spoof_cm_scores[x] for x, y in \
                            enumerate(spoof_cm_file_names) \
                            if spoof_type_dic[y] == spoof_attack_type]
        tmp_spoof_scores = np.array(tmp_spoof_scores)
        decomposed_spoof_scores.append(tmp_spoof_scores.copy())
        if len(tmp_spoof_scores):
            x1, x2, x3 = tDCF_wrapper(bonafide_cm_scores, tmp_spoof_scores)
            min_tDCF[idx] = x1
            eer_cm[idx] = x2
            eer_threshold[idx] = x3

    # pooled results
    x1, x2, x3 = tDCF_wrapper(bonafide_cm_scores, spoof_cm_scores)
    min_tDCF[-1] = x1
    eer_cm[-1] = x2
    eer_threshold[-1] = x3
    spoof_attack_types.append("pooled")
    decomposed_spoof_scores.append(spoof_cm_scores)

    for idx in range(len(spoof_attack_types)):
        if verbose and eer_cm[idx] > -1:
            print("{:s}\tmin-tDCF: {:2.5f}\tEER: {:2.3f}%\t Thre:{:f}".format(
                spoof_attack_types[idx], min_tDCF[idx], eer_cm[idx] * 100, 
                eer_threshold[idx]))

    decomposed_spoof_scores = [decomposed_spoof_scores[x] \
                               for x, y in enumerate(min_tDCF) if y > -1]
    spoof_attack_types = [spoof_attack_types[x] \
                               for x, y in enumerate(min_tDCF) if y > -1]
    eer_threshold = [eer_threshold[x] \
                               for x, y in enumerate(min_tDCF) if y > -1]
    eer_cm = [eer_cm[x] for x, y in enumerate(min_tDCF) if y > -1]
    min_tDCF = [y for x, y in enumerate(min_tDCF) if y > -1]
        
    return min_tDCF, eer_cm, eer_threshold, spoof_attack_types, \
        decomposed_spoof_scores
    
##############
# for Pytorch models in this project
##############

def parse_pytorch_output_txt(score_file_path):
    """ parse_pytorch_output_txt(file_path)
    parse the score files generated by the pytorch models
    
    input
    -----
      file_path: path to the log file
    
    output
    ------
      bonafide: np.array, bonafide scores
      bonafide_names: list of file names corresponding to bonafide scores
      spoofed: np.array, spoofed scores
      spoofed_names: list of file names corresponding to spoofed scores
    
    """
    bonafide = []
    spoofed = []
    bonafide_names = []
    spoofed_names = []
    with open(score_file_path, 'r') as file_ptr:
        for line in file_ptr:
            if line.startswith('Output,'):
                temp = line.split(',')
                flag = int(temp[2])
                if np.isnan(float(temp[3])):
                    print(line)
                    continue
                if flag:
                    bonafide.append(float(temp[3]))
                    bonafide_names.append(temp[1].strip())
                else:
                    spoofed.append(float(temp[3]))
                    spoofed_names.append(temp[1].strip())
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, bonafide_names, spoofed, spoofed_names


def ASVspoof2019_decomposed_results(score_file_path, flag_return_results=False, 
                                    flag_verbose=True):
    """ Get the results from input score log file
    ASVspoof2019_decomposed_results(score_file_path, flag_return_results=False,
                                    flag_verbose=True)
    input
    -----
      score_file_path: path to the score file produced by the Pytorch code
      flag_return_results: whether return the results (default False)
      flag_verbose: print EERs and mintDCFs for each attack (default True)

    output
    ------
      if flag_return_results is True:
        mintDCFs: list of min tDCF, for each attack
        eers: list of EER, for each attack
        cm_thres: list of threshold for EER, for each attack
        spoof_types: list of spoof attack types
        spoof_scores: list of spoof file scores (np.array)
        bona: bonafide score
    """
    bona, b_names, spoofed, s_names = parse_pytorch_output_txt(score_file_path)
    
    mintDCFs, eers, cm_thres, spoof_types, spoof_scores = ASVspoof2019_evaluate(
        bona, b_names, spoofed, s_names, flag_verbose)
    
    if flag_return_results:
        return mintDCFs, eers, cm_thres, spoof_types, spoof_scores, bona 
    else:
        return

def ASVspoofNNN_decomposed_results(score_file_path, 
                                   flag_return_results=False,
                                   flag_verbose=True,
                                   protocol_alternative=None):
    """ Similar to ASVspoof2019_decomposed_results, but use alternative protocol
    """
    bona, b_names, spoofed, s_names = parse_pytorch_output_txt(score_file_path)

    mintDCFs, eers, cm_thres, spoof_types, spoof_scores = ASVspoof2019_evaluate(
        bona, b_names, spoofed, s_names, flag_verbose, protocol_alternative)
    
    if flag_return_results:
        return mintDCFs, eers, cm_thres, spoof_types, spoof_scores, bona 
    else:
        return

##############
# for testing using ./data/cm_dev.txt and asv_dev.txt
##############

def read_asv_txt_file(file_path):
    data = np.genfromtxt(
        file_path, dtype=[('class', 'U10'),('type', 'U10'),
                          ('score','f4')], delimiter=" ")
    
    data_new = np.zeros([data.shape[0], 2])
    for idx, data_entry in enumerate(data):
        
        data_new[idx, 0] = data_entry[-1]
        if data_entry[1] == 'target':
            data_new[idx, 1] = 2
        elif data_entry[1] == 'nontarget':
            data_new[idx, 1] = 1
        else:
            data_new[idx, 1] = 0
    return data_new


def read_cm_txt_file(file_path):
    data = np.genfromtxt(
        file_path, dtype=[('class', 'U10'),('type', 'U10'),
                          ('flag', 'U10'),
                          ('score','f4')], delimiter=" ")
    
    data_new = np.zeros([data.shape[0], 2])
    for idx, data_entry in enumerate(data):
        
        data_new[idx, 0] = data_entry[-1]
        if data_entry[-2] == 'bonafide':
            data_new[idx, 1] = 1
        else:
            data_new[idx, 1] = 0
    return data_new

if __name__ == "__main__":
    
    asv_scores = read_asv_txt_file('./data/asvspoof2019/asv_dev.txt')
    cm_scores = read_cm_txt_file('./data/asvspoof2019/cm_dev.txt')

    tar_asv = asv_scores[asv_scores[:, 1]==2, 0]
    non_asv = asv_scores[asv_scores[:, 1]==1, 0]
    spoof_asv = asv_scores[asv_scores[:, 1]==0, 0]

    bona_cm = cm_scores[cm_scores[:, 1]==1, 0]
    spoof_cm = cm_scores[cm_scores[:, 1]==0, 0]

    mintdcf, eer, eer_threshold = tDCF_wrapper(
        bona_cm, spoof_cm, tar_asv, non_asv, spoof_asv)

    print("min tDCF: {:f}".format(mintdcf))
    print("EER: {:f}%".format(eer*100))
