#!/usr/bin/python
"""
Convert txt score files to binary data matrix, in format [score, class]

class: 0: spoof, 1: non-target, 2: target
"""

import core_scripts.data_io.io_tools as nii_io
import numpy as np
import os
import sys

def read_txt_file(file_path):
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

def convert_format(file_path):
    data = read_txt_file(file_path)
    file_new_path = os.path.splitext(file_path)[0] + '.bin'
    nii_io.f_write_raw_mat(data, file_new_path)

if __name__ == "__main__":
    
    file_list = ['ASVspoof2019.LA.asv.dev.gi.trl.scores.txt',
                 'ASVspoof2019.PA.asv.dev.gi.trl.scores.txt',
                 'ASVspoof2019.LA.asv.eval.gi.trl.scores.txt',
                 'ASVspoof2019.PA.asv.eval.gi.trl.scores.txt']
    
    for filename in file_list:
        convert_format(filename)
