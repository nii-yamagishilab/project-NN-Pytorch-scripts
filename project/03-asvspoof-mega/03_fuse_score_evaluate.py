#!/usr/bin/python
""" 
Wrapper to fuse score and compute EER and min tDCF
Simple score averaging.

Usage:
python 03_fuse_score_evaluate.py log_output_testset_1 log_output_testset_2 ...

The log_output_testset is produced by the pytorch code, for
example, ./lfcc-lcnn-lstmsum-am/01/__pretrained/log_output_testset

It has information like:
...
Generating 71230,LA_E_9999427,0,43237,0, time: 0.005s
Output, LA_E_9999487, 0, 0.172325
...
(See README for the format of this log)

This script will extract the line starts with "Output, ..."

"""

import os
import sys
import numpy as np
from sandbox import eval_asvspoof

def parse_txt(file_path):
    bonafide = []
    bonafide_file_name = []
    spoofed = []
    spoofed_file_name = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if line.startswith('Output,'):
                #Output, LA_E_9999487, 0, 0.172325
                temp = line.split(',')
                flag = int(temp[2])
                name = temp[1]
                if flag:
                    bonafide_file_name.append(name)
                    bonafide.append(float(temp[-1]))
                else:
                    spoofed.append(float(temp[-1]))
                    spoofed_file_name.append(name)
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, spoofed, bonafide_file_name, spoofed_file_name


def fuse_score(file_path_lists):
    bonafide_score = {}
    spoofed_score = {}
    for data_path in file_path_lists:
        bonafide, spoofed, bona_name, spoof_name = parse_txt(data_path)
        for score, name in zip(bonafide, bona_name):
            if name in bonafide_score:
                bonafide_score[name].append(score)
            else:
                bonafide_score[name] = [score]
        for score, name in zip(spoofed, spoof_name):
            if name in spoofed_score:
                spoofed_score[name].append(score)
            else:
                spoofed_score[name] = [score]
    fused_bonafide = np.array([np.mean(y) for x, y in bonafide_score.items()])
    fused_spoofed = np.array([np.mean(y) for x, y in spoofed_score.items()])
    return fused_bonafide, fused_spoofed
        

if __name__ == "__main__":
    
    data_paths = sys.argv[1:]
    bonafide, spoofed = fuse_score(data_paths)
    mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed)
    print("Score file: {:s}".format(str(data_paths)))
    print("mintDCF: {:1.4f}".format(mintDCF))
    print("EER: {:2.3f}%".format(eer * 100))
    print("Threshold: {:f}".format(threshold))
