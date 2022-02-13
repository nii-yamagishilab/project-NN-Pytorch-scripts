#!/usr/bin/python
""" 
Wrapper to parse the score file and compute EER and min tDCF

Usage:
 $: python 02_evaluate.py score_file protocol
  
 score_file: score file produced by 01_eval.sh.
   It has lines like:
   ...
   LA_E_9999487, 0, 0.172325, ...
   (See README for on the format of the score file)

 protocol: a protocol file that shows the bonafide/spoof label
   It has line like
   LA_0069 LA_D_1047731 - - bonafide
   
   The second column is the trial name, the last column is label

Note:
  02_evaluate.py is made for this project only.
  score files from ../03_asvspoof_mega or other projects may 
  have a different format, not compatible with this 02_evaluate.py
"""
import os
import sys
import numpy as np
from sandbox import eval_asvspoof

def parse_protocol(protocol_file):
    protocol = {}
    with open(protocol_file, 'r') as file_ptr:
        for line in file_ptr:
            trial_name = line.split()[1]
            label = line.split()[-1]
            if label == 'bonafide':
                protocol[trial_name] = 1
            else:
                protocol[trial_name] = 0
    return protocol

def parse_txt(file_path, protocol):
    bonafide = []
    spoofed = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            temp = line.split()
            if len(temp) != 4:
                print("Not supported score format")
                sys.exit(1)
            name = temp[0]
            flag = int(protocol[name])
            if flag:
                bonafide.append(float(temp[2]))
            else:
                spoofed.append(float(temp[2]))
                    
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, spoofed

if __name__ == "__main__":
    
    data_path = sys.argv[1]
    protocol = sys.argv[2]
    protocol = parse_protocol(protocol)
    bonafide, spoofed = parse_txt(data_path, protocol)
    mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed)
    print("Score file: {:s}".format(data_path))
    print("mintDCF (using ASVspoof2019 LA config): {:1.4f}".format(mintDCF))
    print("EER: {:2.3f}%".format(eer * 100))
    print("Threshold: {:f}".format(threshold))
