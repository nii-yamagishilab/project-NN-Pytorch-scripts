#!/usr/bin/python
""" 
Wrapper to parse the score file and compute EER and min tDCF

Usage:
python 00_evaluate.py log_file

"""
import os
import sys
import numpy as np
from sandbox import eval_asvspoof

def parse_txt(file_path):
    bonafide = []
    spoofed = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if line.startswith('Output,'):
                temp = line.split(',')
                flag = int(temp[2])
                if flag:
                    bonafide.append(float(temp[-1]))
                else:
                    spoofed.append(float(temp[-1]))
    bonafide = np.array(bonafide)
    spoofed = np.array(spoofed)
    return bonafide, spoofed



if __name__ == "__main__":
    
    data_path = sys.argv[1]
    
    bonafide, spoofed = parse_txt(data_path)
    mintDCF, eer, threshold = eval_asvspoof.tDCF_wrapper(bonafide, spoofed)
    print("mintDCF: %f\tEER: %2.3f %%\tThreshold: %f" % (mintDCF, eer * 100, 
                                                         threshold))
