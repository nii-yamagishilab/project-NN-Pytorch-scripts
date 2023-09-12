#!/usr/bin/env python
"""
log_parser

tools to parse log_train and log_err
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import re
import sys
import numpy as np
import scipy
try:
    import pandas as pd
except ImportError:
    print("Log_parser requires pandas")
    print("Many functions cannot be used without pandas")


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


# ====
# utilities
# ====

# filtering the error curve
def smooth_geo(data, factor = 0.1, tap = 10):
    taps = np.arange(tap)
    b_coef = np.power(factor, taps) * (1-factor)
    return scipy.signal.lfilter(
        b_coef, [1], np.concatenate([np.ones([tap]) * data[0], data]))[tap:]

def smooth_ave(data, tap = 100):
    # check freq-response
    # w, h = scipy.signal.freqz(b_coef)
    # plot_API.plot_API(20 * np.log10(abs(h)), plot_lib.plot_signal, 'single')
    b_coef = np.ones([tap]) * 1/tap
    return scipy.signal.lfilter(
        b_coef, [1], np.concatenate([np.ones([tap]) * data[0], data]))[tap:]

# === 
# APIs
# ===

def f_read_log_err_pd(file_path, cell_losses = None):
    """ pd = f_read_log_err_pd(file_path)

    Automatically parse the log_err file, where each line has format like:
    10753,LJ045-0082,0,9216,0, 22/12100, Time: 0.190877s, Loss: 85.994621, ... 

    input:  file_path,     str, path to the log file
    input:  num_of_losses, int or None, number of Loss cells in the log
                           if None, it will be estimated from 1st line of log
    output: pd, pandas.Frame
    """
    def _load_number_of_losses(file_path):
        with open(file_path, 'r') as file_ptr:
            for line in file_ptr:
                cells = line.rstrip().split(',')
                num_of_loss = [x for x in cells if x.count("Loss:")]
                break
            return len(num_of_loss)
    def _convert_loss(cell_value):
        if cell_value.count('Loss:'):
            return float(cell_value.split(':')[-1])
        else:
            return np.nan

    if cell_losses is None:
        num_of_losses = _load_number_of_losses(file_path)
        cell_losses = ['loss-{:d}'.format(x) for x in range(num_of_losses)]
    
    # names for csv parsing
    names = ['id', 'trial', 'seg_idx', 'length', 'start_pos', 'order', 'time']
    names = names + cell_losses

    # convert Loss: yyy to yyy in float number
    converters = {x: _convert_loss for x in cell_losses}

    # parse the log
    return pd.read_csv(file_path, sep=',', 
                       names = names, converters = converters)
            

# ========
# Old APIs for parsing training logs
# ========
def f_read_log_err(file_path):
    """
    each line looks like 
    10753,LJ045-0082,0,9216,0, 22/12100, Time: 0.190877s, Loss: 85.994621, ...
    """
    def parse_line(line_input):
        line_tmps = line_input.split(',')
        tmp_loss = []
        for tmp in line_tmps:
            if tmp.count('Time'):
                tmp_time = float(tmp.lstrip(' Time:').rstrip('s'))
            elif tmp.count('Loss'):
                tmp_loss.append(float(tmp.lstrip(' Loss:')))
        return tmp_time, tmp_loss
                
    time_mat = []
    error_mat = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            line = line.rstrip()
            if line.count("Loss"):
                tmp_time, tmp_loss = parse_line(line)
                time_mat.append(tmp_time)
                error_mat.append(tmp_loss)
    return np.array(error_mat), np.array(time_mat)

# This function is obsolete
def f_read_log_err_old(file_path, train_num, val_num):
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


def pass_number(input_str):
    return np.array([float(x) for x in input_str.split()]).sum()


def f_read_log_train(file_path, sep='/'):
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
    def parse_line(line_input, sep):
        if sep == ' ':
            return line_input.split()
        else:
            return line_input.split(sep)

    read_flag = False
    
    data_str = []
    with open(file_path, 'r') as file_ptr:
        for line in file_ptr:
            if read_flag and line.count('|') > 2:
                data_str.append(line)
            if line.count('Duration'):
                read_flag = True
            
    row = len(data_str)
    data_train = None 
    data_val   = None

    time_per_epoch = np.zeros(row)
    for idx, line in enumerate(data_str):
        try:
            time_per_epoch[idx] = float(line.split('|')[1])
        except ValueError:
            continue
        
        trn_data = parse_line(line.split('|')[2], sep)
        val_data = parse_line(line.split('|')[3], sep)
        
        if data_train is None or data_val is None:
            data_train = np.zeros([row, len(trn_data)])
            data_val = np.zeros([row, len(val_data)])
        
        for idx2 in np.arange(len(trn_data)):
            data_train[idx, idx2] = pass_number(trn_data[idx2])
            data_val[idx,idx2] = pass_number(val_data[idx2])

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
