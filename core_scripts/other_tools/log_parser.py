#!/usr/bin/env python
"""
log_parser

tools to parse log_train and log_err
"""

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import re
import sys

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def f_read_log_err(file_path, train_num, val_num):
    """ 
    log_train, log_val = f_read_log_err(log_err, num_train_utt, num_val_utt)

    input:
    -----
     log_err: path to the log_err file
     num_train_utt: how many training utterances
     num_val_utt: how many validation utterances
    
    output:
    ------
     log_train: np.array, average error values per epoch on training set
     log_val: np.array, average error values per epoch on valiation set
    """
    
    data_str = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if not line.count('skip'):
                try:
                    tmp = int(line[0])
                    data_str.append(line)
                except ValueError:
                    pass

    row = len(data_str)
    col = len(np.fromstring(data_str[0], dtype=np.float32, sep=','))
    
    data = np.zeros([row,col])
    for idx, line in enumerate(data_str):
        data[idx, :] = np.fromstring(line, dtype=np.float32, sep=',')
    
    print(data.shape[0])
    total_num = train_num + val_num
    epoch_num = int(data.shape[0] / total_num)
    data_train = np.zeros([epoch_num, data.shape[1]])
    data_val = np.zeros([epoch_num, data.shape[1]])
    
    for x in range(epoch_num):
        temp_data = data[x * total_num:(x+1)*total_num, :]
        train_part = temp_data[0:train_num,:]
        val_part = temp_data[train_num:(train_num+val_num),:]
        data_train[x, :] = np.mean(train_part, axis=0)
        data_val[x, :] = np.mean(val_part, axis=0)
    
    return data_train, data_val


def f_read_log_train(file_path):
    """ 
    data_train, data_val, time_per_epoch = read_log_train(path_to_log_train)
    
    input:
    -----
     path_to_log_train: path to the log_train file
    
    output:
    ------
     data_train: error values per epoch on training set
     data_val: error values per epoch on valiation set
     time_per_epoch: training time per epoch
    """
    read_flag = False
    
    data_str = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if read_flag and line.count('|') > 2:
                data_str.append(line)
            if line.count('Duration'):
                read_flag = True
            
    row = len(data_str)

    data_train = np.zeros([row, 3])
    data_val   = np.zeros([row, 3])
    time_per_epoch = np.zeros(row)
    for idx, line in enumerate(data_str):
        try:
            time_per_epoch[idx] = float(line.split('|')[1])
        except ValueError:
            continue
        trn_data = line.split('|')[2].split('/')
        val_data = line.split('|')[3].split('/')
        for idx2 in np.arange(len(trn_data)):
            data_train[idx, idx2] = float(trn_data[idx2])
            data_val[idx,idx2] = float(val_data[idx2])

    return data_train, data_val, time_per_epoch
    

def read_log_err_pytorch(file_path, merge_epoch=False):
    def set_size(line):
        return int(line.split('/')[1].split(',')[0])
    def data_line(line):
        if line.count("Time:"):
            return True
        else:
            return False
    def get_data(line):
        return [float(x.split(":")[1]) for x in line.split(',') if x.count("Loss:")]
    
    trn_utt_num = None
    val_utt_num = None
    trn_total_num = 0
    val_total_num = 0
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if not data_line(line):
                continue
            temp_num = set_size(line)
            col_num = len(get_data(line))
            if trn_utt_num is None:
                trn_utt_num = temp_num
            if temp_num != val_utt_num and temp_num != trn_utt_num:
                val_utt_num = temp_num
            if trn_utt_num == temp_num:
                trn_total_num += 1
            if val_utt_num == temp_num:
                val_total_num += 1
                
    if trn_utt_num is None:
        print("Cannot parse file")
        return
    if val_utt_num is None:
        print("Trn %d, no val" % (trn_utt_num))
    else:
        print("Trn %d, val %d" % (trn_utt_num, val_utt_num))
    print("Trn data %d, val data %d" % (trn_total_num, val_total_num))
    trn_data = np.zeros([trn_total_num, col_num]) 
    val_data = np.zeros([val_total_num, col_num]) 
    trn_utt_cnt = 0
    val_utt_cnt = 0
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if not data_line(line):
                continue
            data = get_data(line)
            temp_num = set_size(line)
            if trn_utt_num == temp_num:
                trn_data[trn_utt_cnt, :] = np.array(data)
                trn_utt_cnt += 1
            if val_utt_num == temp_num:
                val_data[val_utt_cnt, :] = np.array(data)
                val_utt_cnt += 1
    if merge_epoch:
        trn_data_new = np.zeros([trn_total_num // trn_utt_num, col_num])
        val_data_new = np.zeros([val_total_num // val_utt_num, col_num])
        for idx in range(min([trn_total_num // trn_utt_num, val_total_num // val_utt_num])):
            trn_data_new[idx, :] = trn_data[idx*trn_utt_num:(idx+1)*trn_utt_num, :].mean(axis=0)
            val_data_new[idx, :] = val_data[idx*val_utt_num:(idx+1)*val_utt_num, :].mean(axis=0)
        return trn_data_new, val_data_new
    else:
        return trn_data, val_data

if __name__ == "__main__":
    print("logParser")
