#!/usr/bin/env python
"""
Tools to load and parse protocol
"""
from __future__ import absolute_import

import os
import sys
import pandas as pd

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"

g_asvspoof19_protocol = ['speaker', 'utterance', '-', 'attack', 'label']


def protocol_to_spk_utt(protocol_pd):
    """dic_spk_utt = protocol_to_spk_utt(protocol_pd)
    Generate speaker-utterance dictionary from protocol file
    
    input
    -----
      protocol_pd: pd.frame, which has columns defined in g_asvspoof19_protocol
    
    output
    ------
      dic_spk_utt: dictionary
    
      {'spk_id1':{'bonafide': [utt1, utt2],
                  'spoof': [utt5]},                                    
       'spk_id2':{'bonafide': [utt3, utt4, utt8],
                  'spoof': [utt6, utt7]}
      }
    """
    dic_spk_utt = dict()
    for speaker in protocol_pd['speaker'].unique():
        utts = protocol_pd.query('speaker == "{:s}"'.format(speaker))
        dic_spk_utt[speaker] = dict()
        for label in utts['label'].unique():
            utts_sub = utts.query('label == "{:s}"'.format(label))
            dic_spk_utt[speaker][label] = utts_sub['utterance'].to_list()
    return dic_spk_utt


def load_cm_protocol(filepath, names=g_asvspoof19_protocol, sep=' '):
    """
    """
    protocol_pd = pd.read_csv(filepath, sep=sep, names=names)
    return protocol_pd

def load_csv(filepath):
    return pd.read_csv(filepath)


if __name__ == "__main__":
    print(__doc__)
