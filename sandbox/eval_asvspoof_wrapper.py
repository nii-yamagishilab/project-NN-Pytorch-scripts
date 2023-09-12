#!/usr/bin/env python
"""
Wrapper to evaluate EER over different databases.

This is used internally to compute EER on multiple testsets.
"""


from __future__ import print_function

import os
import sys
import copy
import tqdm

import numpy as np
import pandas as pd

from core_scripts.data_io import io_tools
import sandbox.eval_asvspoof as eval_asvspoof

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"


# Load protocols
def parse_protocol(protocol_path, tag):
    """protocol_pd = parse_protocol(protocol_path, tag)
    
    input
    -----
      protocol_path: str, path to the protocol file
      tag: str, type of the protocol format: 
                LA2021, DF2021, or other

    output
    ------
      protocol_pd: pandas data frame, protocol
    """
        
    if tag == 'LA2021':
        tmp_pd = pd.read_csv(
            protocol_path, 
            names = ['voice', 'trial', 'codec', 'trans', 'attack', 
                     'label', 'trim', 'track'], 
            sep = ' ', index_col = 'trial')

    elif tag == 'DF2021':
        tmp_pd = pd.read_csv(
            protocol_path, 
            names = ['voice', 'trial', 'codec', 'year', 'attack', 
                     'label', 'trim', 'track', 'type'], 
            sep = ' ', index_col = 'trial')
    else:
        tmp_pd = pd.read_csv(
            protocol_path, 
            names = ['voice', 'trial', '-', 'attack', 'label'], 
            sep = ' ', index_col = 'trial')
    tmp_pd['dataset'] = tag
    return tmp_pd



def parse_score(score_file, return_score_only=True):
    """ score_pd = parse_score(score_file, return_score_only=True)

    input
    -----
      score_file: str, path to the score file
      return_score_only: bool, whether to return the whole dataframe 
                         or just score column
    output
    ------
      score_pd: pandas frame, score frame, index is trial name, columns is score
    """
    # decide the number of additional columns (except trial name and score)
    if os.stat(score_file).st_size == 0:
        return []
    
    # assume format trial_name score evidence-1 evidence-2 ...
    with open(score_file, 'r') as file_ptr:
        line = next(file_ptr)
        n_evi = len(line.split()) - 2
        if line.rstrip('\n')[-1] == ' ':
            n_evi += 1
    
    # load score file
    names = ['trial', 'score'] + ['evi-{:d}'.format(x+1) for x in range(n_evi)]
    score_pd = pd.read_csv(score_file, names = names,  sep = ' ', 
                           skipinitialspace=True, index_col = 'trial')
    
    # return
    if return_score_only:
        return score_pd[['score']]
    else:
        return score_pd

    
def get_eer_ver2(data_pd):
    """eer = get_eer_ver2(data_pd)

    input
    -----
      data_pd: pandas dataFrame, that contains score and labels

    output
    ------
      eer: float, EER
    """
    
    # retrieve score array
    bona_query = "label == 'bonafide'"
    spoof_query = "label == 'spoof'"
    
    bona_score = data_pd.query(bona_query)['score'].to_numpy()
    spoof_score = data_pd.query(spoof_query)['score'].to_numpy()
    
    # compute EER
    eer, threshold = eval_asvspoof.compute_eer(bona_score, spoof_score)
    return eer


def get_cllr_ver2(data_pd):
    """cllr = get_cllr_ver2(data_pd)

    input
    -----
      data_pd: pandas dataFrame, that contains score and labels

    output
    ------
      cllr: float, cllr
    """
    
    # retrieve score array
    bona_query = "label == 'bonafide'"
    spoof_query = "label == 'spoof'"
    
    bona_score = data_pd.query(bona_query)['score'].to_numpy()
    spoof_score = data_pd.query(spoof_query)['score'].to_numpy()
    
    # compute EER
    cllr = eval_asvspoof.compute_cllr(bona_score, spoof_score)
    return cllr

def get_eer_ver3(data_pds):
    """eer = get_eer_ver3(data_pd_lists)

    Different from get_eer_ver2, this API assumes input is a list of
    score dataFrame. EER will be computed assuming a mixture of score
    distribution
    
    input
    -----
      data_pds: list of pandas dataFrame, 

    output
    ------
      eer: float, EER
    """
    
    # retrieve score array
    bona_query = "label == 'bonafide'"
    spoof_query = "label == 'spoof'"
    
    bona_scores = [x.query(bona_query)['score'].to_numpy() for x in data_pds]
    spoof_scores = [x.query(spoof_query)['score'].to_numpy() for x in data_pds]
    
    # compute EER
    eer, threshold = eval_asvspoof.compute_eer(bona_scores, spoof_scores)
    return eer


g_eer_pooled_mixture = True
g_basic_eval_config = [
    ["LA19 trn", 
     ["config_train_asvspoof2019"], ["dataset == 'LA2019'"], 
     [[5128, 45096]]], 
    
    ["LA15 eval", 
     ["config_test_asvspoof2015"], ["dataset == 'LA2015'"], 
     [[9404, 184000]]],
    
    ["LA19 eval",
     ["config_test_asvspoof2019"], ["dataset == 'LA2019'"], 
     [[7355, 63882]]],
    
    ["LA19 etrim",
     ["config_test_asvspoof2019_nosil"], ["dataset == 'LA2019'"], 
     [[7355, 63882]]],
    
    ["LA21 prog", 
     ["config_test_asvspoof2021"], 
     ["dataset == 'LA2021' and track == 'progress' and trim == 'notrim'"], 
     [[ 1676, 14788 ]]],
    
    ["LA21 eval", 
     ["config_test_asvspoof2021"], 
     ["dataset == 'LA2021' and track == 'eval' and trim == 'notrim'"], 
     [[14816, 133360]]],
    
    ["LA21 hid", 
     ["config_test_asvspoof2021"], 
     ["dataset == 'LA2021' and track == 'hidden_track'"], 
     [[1960, 14966] ]],
    
    ["DF21 prog", 
     ["config_test_asvspoof2021_DF_full"], 
     ["dataset == 'DF2021' and track == 'progress' and trim == 'notrim'"], 
     [[5768, 53557]]],
    
    ["DF21 eval", 
     ["config_test_asvspoof2021_DF_full"], 
     ["dataset == 'DF2021' and track == 'eval' and trim == 'notrim'"], 
     [[14869, 519059]]],
    
    ["DF21 evalsub", 
     ["config_test_asvspoof2021_DF_full"],
     ["dataset == 'DF2021' and track == 'eval' and year == 'asvspoof'"], 
     [[7939, 73856 ]]],
    
    ["DF21 hid", 
     ["config_test_asvspoof2021_DF_full"], 
     ["dataset == 'DF2021' and track == 'hidden_track'"], 
     [[1980, 16596]]],
    
    ["WF eval", 
     ["config_test_others_wavefake"], 
     ["dataset == 'WaveFake'"], 
     [[18071, 117983]]],
    
    ["DFWild", 
     ["config_test_deepfake_in_wild"], 
     ["dataset == 'DFWild'"], 
     [[19963, 11816]]],
    
    ["GroupA", 
     ["config_test_asvspoof2019",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full"], 
     ["dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'eval' and trim == 'notrim'",
      "dataset == 'DF2021' and track == 'eval' and trim == 'notrim'"],
     [[7355, 63882], [14816, 133360], [14869, 519059]]],
    
    ["GroupB", 
     ["config_test_asvspoof2019_nosil",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full",
      "config_test_others_wavefake",
      "config_test_deepfake_in_wild"], 
     ["dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'hidden_track'",
      "dataset == 'DF2021' and track == 'hidden_track'",
      "dataset == 'WaveFake'",
      "dataset == 'DFWild'"],
     [[7355, 63882], [1960, 14966], [1980, 16596], 
      [18071, 117983], [19963, 11816]]],

    ["All", 
     ["config_test_asvspoof2019",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full",
      "config_test_asvspoof2019_nosil",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full",
      "config_test_others_wavefake",
      "config_test_deepfake_in_wild"], 
     ["dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'eval' and trim == 'notrim'",
      "dataset == 'DF2021' and track == 'eval' and trim == 'notrim'",
      "dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'hidden_track'",
      "dataset == 'DF2021' and track == 'hidden_track'",
      "dataset == 'WaveFake'",
      "dataset == 'DFWild'"],
     [[7355, 63882], [14816, 133360], [14869, 519059], 
      [7355, 63882], [1960, 14966], [1980, 16596], 
      [18071, 117983], [19963, 11816]]],


    ["GroupA_mix", 
     ["config_test_asvspoof2019",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full"], 
     ["dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'eval' and trim == 'notrim'",
      "dataset == 'DF2021' and track == 'eval' and trim == 'notrim'"],
     [[7355, 63882], [14816, 133360], [14869, 519059]],
     g_eer_pooled_mixture],
    
    ["GroupB_mix", 
     ["config_test_asvspoof2019_nosil",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full",
      "config_test_others_wavefake",
      "config_test_deepfake_in_wild"], 
     ["dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'hidden_track'",
      "dataset == 'DF2021' and track == 'hidden_track'",
      "dataset == 'WaveFake'",
      "dataset == 'DFWild'"],
     [[7355, 63882], [1960, 14966], [1980, 16596], 
      [18071, 117983], [19963, 11816]],
     g_eer_pooled_mixture],

    ["All_mix", 
     ["config_test_asvspoof2019",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full",
      "config_test_asvspoof2019_nosil",
      "config_test_asvspoof2021",
      "config_test_asvspoof2021_DF_full",
      "config_test_others_wavefake",
      "config_test_deepfake_in_wild"], 
     ["dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'eval' and trim == 'notrim'",
      "dataset == 'DF2021' and track == 'eval' and trim == 'notrim'",
      "dataset == 'LA2019'", 
      "dataset == 'LA2021' and track == 'hidden_track'",
      "dataset == 'DF2021' and track == 'hidden_track'",
      "dataset == 'WaveFake'",
      "dataset == 'DFWild'"],
     [[7355, 63882], [14816, 133360], [14869, 519059], 
      [7355, 63882], [1960, 14966], [1980, 16596], 
      [18071, 117983], [19963, 11816]],
     g_eer_pooled_mixture],
]

g_basic_eval_config_v1 = g_basic_eval_config[0:16]
g_basic_eval_config_v2 = g_basic_eval_config[0:19]


class EERManager():
    """EERManager
    """
    def __init__(self, 
                 protocol_paths, 
                 dump_path, 
                 eval_config_add = [], 
                 version = 'v1',
                 dry_run = False):
        
        # protocol paths
        self.protocol_paths = protocol_paths

        # parse all the protocols
        self.protocol_pd = pd.concat(
            [parse_protocol(x[0], x[1]) for x in protocol_paths])
        
        # load configuration file
        if version == 'v2':
            self.eval_configs = g_basic_eval_config_v2
        else:
            self.eval_configs = g_basic_eval_config_v1

        if eval_config_add:
            self.eval_configs = self.eval_configs + eval_config_add

        # name of test sets
        self.test_set_name = [x[0] for x in self.eval_configs]

        # path to dump EER results and so on
        self.dump_path = dump_path
        if not os.path.isdir(self.dump_path):
            os.makedirs(self.dump_path, exist_ok=True)
        
        # dry run?
        self.dry_run = dry_run

        return


    def show_test_set_name(self):
        """
        print test set name
        """
        for idx, x in enumerate(self.test_set_name):
            print(idx, x)

    def _get_score_file_name(self, system_path, filename, log_type = None):
        if log_type == 'evi':
            tmp = 'log_output_' + filename + '_evi'
        else:
            tmp = 'log_output_' + filename + '_score'    
        return system_path + '/' + tmp + '.txt'

    #def _wrap_eer(self, eer_dict):
    #    return pd.DataFrame.from_dict(
    #        eer_dict, orient='index', columns = ['eer'])

    #def _wrap_cllr(self, cllr_dict):
    #    return pd.DataFrame.from_dict(
    #        cllr_dict, orient='index', columns = ['cllr'])

    def _wrap_output(self, output_dict, keyname):
        return pd.DataFrame.from_dict(
            output_dict, orient='index', columns = [keyname])


    def _get_dump_path(self, system_path):
        return self.dump_path + '/' + system_path.replace('/', '_') + '.raw.pkl'

    #def _get_eer_path(self, system_path):
    #    return self.dump_path + '/'+system_path.replace('/', '_') +'.eer.pkl'
    #def _get_cllr_path(self, system_path):
    #    return self.dump_path + '/'+system_path.replace('/', '_')+'.cllr.pkl'

    def _get_output_path(self, system_path, typename):
        tmp = self.dump_path + '/' + system_path.replace('/', '_')
        tmp = tmp + '.{:s}.pkl'.format(typename)
        return tmp

    def _compute_eer(self, system_path):
        """
        """
        
        empty_list = []
        dump_list = {}
        eer_list = {}
        cllr_list = {}

        # loop over different test sets
        for eval_config in tqdm.tqdm(self.eval_configs):
            
            eval_name = eval_config[0]
            file_lists = eval_config[1]
            eval_queries = eval_config[2]
            trial_nums = eval_config[3]
            if len(eval_config) == 5:
                flag_eer_mix = eval_config[4]
            else:
                flag_eer_mix = False
            
            # load PD
            score_pd_list = []
            
            # loop over different test subsets
            for file_name, eval_query, trial_num in \
                zip(file_lists, eval_queries, trial_nums):
                 
                # spoofed + bona fide data
                trial_total_num = np.sum(trial_num)
                
                # load score file
                score_file = self._get_score_file_name(system_path, file_name)

                if os.path.isfile(score_file):
                    score_pd = parse_score(score_file)

                if not os.path.isfile(score_file) or len(score_pd) == 0:
                    empty_list.append(score_file)
                    continue

                # load protocol corresponding to the score file
                if eval_query is not None:
                    tmp_protocol_pd = self.protocol_pd.query(eval_query) 
                else:
                    tmp_protocol_pd = self.protocol_pd

                # merge score and protocol PD
                tmp_pd = score_pd.join(tmp_protocol_pd)
                tmp_pd = tmp_pd[~tmp_pd['score'].isna()]
                tmp_pd = tmp_pd[~tmp_pd['label'].isna()]
                
                if len(tmp_pd) != trial_total_num:
                    print(score_file)
                    print(len(tmp_pd), trial_total_num)
                    raise ValueError
                
                score_pd_list.append(tmp_pd)
                
            # 
            if len(score_pd_list) and len(score_pd_list) == len(eval_queries):

                if flag_eer_mix:
                    # compute pooled EER assuming scores are from mixture
                    if not self.dry_run:
                        eer = get_eer_ver3(score_pd_list)
                    else:
                        eer = 100
                else:
                    # simply pooled scores together
                    score_pd = pd.concat(score_pd_list)
                    # compute EER and save
                    if not self.dry_run:
                        eer = get_eer_ver2(score_pd)
                    else:
                        eer = 100

                if not self.dry_run:
                    cllr = get_cllr_ver2(pd.concat(score_pd_list))
                else:
                    cllr = np.nan
            else:
                score_pd = None
                eer = np.nan
                cllr = np.nan
                
            dump_list[eval_name] = score_pd
            eer_list[eval_name] = eer
            cllr_list[eval_name] = cllr

        return dump_list, \
            self._wrap_output(eer_list, 'eer'), \
            self._wrap_output(cllr_list, 'cllr')
        
    def compute_eer(self, system_path):
        
        if os.path.isdir(system_path):
            print("Compute ", system_path)
            dump_pd_list = {}
            eer_pd_list = {}
            cllr_pd_list = {}
            tmp_prjs = [system_path + '/' + x for x in os.listdir(system_path)]

            for x in filter(os.path.isdir, tmp_prjs):
                print(x)
                dump_list, eer_pd, cllr_pd = self._compute_eer(x)
                dump_pd_list[x] = dump_list
                eer_pd_list[x] = eer_pd
                cllr_pd_list[x] = cllr_pd
        
            save_dump_path = self._get_dump_path(system_path)
            io_tools.pickle_dump(dump_pd_list, save_dump_path)
            save_eer_path = self._get_output_path(system_path, 'eer')
            io_tools.pickle_dump(eer_pd_list, save_eer_path)
            save_cllr_path = self._get_output_path(system_path, 'cllr')
            io_tools.pickle_dump(cllr_pd_list, save_cllr_path)
            
            print("Dump to", save_dump_path)
            print("\t", save_eer_path)
            print("\t", save_cllr_path)
        else:
            print("not available skip", system_path)

        return

    def load_raw_dump(self, system_path):

        save_path = self._get_dump_path(system_path)
        if os.path.isfile(save_path):
            dump_data = io_tools.pickle_load(save_path)
        else:
            print("not available {:s}".format(system_path))
            dump_data = None
        return dump_data
        
    
    def load_output_dump(self, system_path, tmp_dump_path=None, keyname='eer'):

        if tmp_dump_path:
            tmp_dump_path, self.dump_path = self.dump_path, tmp_dump_path

        save_path = self._get_output_path(system_path, keyname)
        if os.path.isfile(save_path):
            output_pds = io_tools.pickle_load(save_path)
            output_pd = pd.concat(output_pds.values(), axis=1).mean(axis=1)
            output_pd = self._wrap_output(output_pd.to_dict(), keyname)
        else:
            output_dic = {x:np.nan for x in self.test_set_name}
            output_pd = self._wrap_output(output_dic, keyname)

        assert len(output_pd) == len(self.test_set_name), "incompatible"

        if tmp_dump_path:
            tmp_dump_path, self.dump_path = self.dump_path, tmp_dump_path

        return output_pd
        
    def load_output_mat(self, 
                        system_paths, 
                        testset_selected_idx=None, 
                        tmp_dump_path=None,
                        keyname='eer'):
        if tmp_dump_path:
            tmp_dump_path, self.dump_path = self.dump_path, tmp_dump_path

        output_pds = [self.load_output_dump(x, tmp_dump_path, keyname) 
                      for x in system_paths]
        output_mat = np.stack(
            [x[keyname].to_numpy() for x in output_pds], axis=1)

        if testset_selected_idx:
            output_mat = output_mat[testset_selected_idx]

        if tmp_dump_path:
            tmp_dump_path, self.dump_path = self.dump_path, tmp_dump_path
        return output_mat

    def load_eers_mat(self, 
                     system_paths, 
                     testset_selected_idx=None, 
                     tmp_dump_path=None):

        return self.load_output_mat(system_paths, 
                                    testset_selected_idx, 
                                    tmp_dump_path,keyname='eer')

    def load_cllrs_mat(self, 
                      system_paths, 
                      testset_selected_idx=None, 
                      tmp_dump_path=None):

        return self.load_output_mat(system_paths, 
                                    testset_selected_idx, 
                                    tmp_dump_path, keyname='cllr')

if __name__ == "__main__":
    print("eval_asvspoof_wrapper")

