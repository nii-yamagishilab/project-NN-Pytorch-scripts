#!/usr/bin/env python
"""
data_io

Interface to load data

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
import torch.utils.data

import core_scripts.other_tools.list_tools as nii_list_tools
import core_scripts.other_tools.display as nii_warn
import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.data_io.io_tools as nii_io_tk
import core_scripts.data_io.wav_tools as nii_wav_tk
import core_scripts.data_io.conf as nii_dconf
import core_scripts.data_io.seq_info as nii_seqinfo
import core_scripts.math_tools.stats as nii_stats

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

###
## functions wrappers to read/write data for this data_io
###
def _data_reader(file_path, dim):
    """ A wrapper to read raw data or waveform
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.wav':
        sr, data = nii_wav_tk.waveReadAsFloat(file_path)
    else:
        data = nii_io_tk.f_read_raw_mat(file_path, dim)
    return data

def _data_writer(data, file_path, sr = 16000):
    """ A wrapper to write raw data or waveform
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.wav':
        nii_wav_tk.waveFloatToPCMFile(data, file_path, sr = sr)
    else:
        nii_io_tk.f_write_raw_mat(data, file_path)
    return

def _data_len_reader(file_path):
    """ A wrapper to read length of data
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.wav':
        sr, data = nii_wav_tk.waveReadAsFloat(file_path)
        length = data.shape[0]
    else:
        length = nii_io_tk.f_read_raw_mat_length(file_path)
    return length

###
# Definition of DataSet
###
class NIIDataSet(torch.utils.data.Dataset):
    """ General class for NII speech dataset
    For definition of customized Dataset, please refer to 
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """
    def __init__(self,
                 dataset_name, \
                 file_list, \
                 input_dirs, input_exts, input_dims, input_reso, \
                 input_norm, \
                 output_dirs, output_exts, output_dims, output_reso, \
                 output_norm, \
                 stats_path, \
                 data_format = '<f4', \
                 truncate_seq = None, \
                 min_seq_len = None, \
                 save_mean_std = True, \
                 wav_samp_rate = None):
        """
        Args:
            dataset_name: name of this data set
            file_list: a list of file name strings (without extension)
            input_dirs: a list of dirs from each input feature is loaded
            input_exts: a list of input feature name extentions
            input_dims: a list of input feature dimensions
            input_reso: a list of input feature temporal resolutions
            output_dirs: a list of dirs from each output feature is loaded
            output_exts: a list of output feature name extentions
            output_dims: a list of output feature dimensions
            output_reso: a list of output feature temporal resolutions
            stat_path: path to the directory that saves mean/std, 
                       utterance length
            data_format: method to load the data
                    '<f4' (default): load data as float32m little-endian
                    'htk': load data as htk format
            truncate_seq: None or int, truncate sequence into truncks.
                          truncate_seq > 0 specifies the trunck length
        """
        # initialization
        self.m_set_name = dataset_name
        self.m_file_list = file_list
        self.m_input_dirs = input_dirs
        self.m_input_exts = input_exts
        self.m_input_dims = input_dims
        
        self.m_output_dirs = output_dirs
        self.m_output_exts = output_exts
        self.m_output_dims = output_dims

        if len(self.m_input_dirs) != len(self.m_input_exts) or \
           len(self.m_input_dirs) != len(self.m_input_dims):
            nii_warn.f_print("Input dirs, exts, dims, unequal length",
                             'error')
            nii_warn.f_print(str(self.m_input_dirs), 'error')
            nii_warn.f_print(str(self.m_input_exts), 'error')
            nii_warn.f_print(str(self.m_input_dims), 'error')
            nii_warn.f_die("Please check input dirs, exts, dims")

        if len(self.m_output_dims) != len(self.m_output_exts) or \
           (self.m_output_dirs and \
            len(self.m_output_dirs) != len(self.m_output_exts)):
            nii_warn.f_print("Output dirs, exts, dims, unequal length", \
                             'error')
            nii_warn.f_die("Please check output dirs, exts, dims")

        # fill in m_*_reso and m_*_norm
        def _tmp_f(list2, default_value, length):
            if list2 is None:
                return [default_value for x in range(length)]
            else:
                return list2
            
        self.m_input_reso = _tmp_f(input_reso, 1, len(input_dims))
        self.m_input_norm = _tmp_f(input_norm, True, len(input_dims))
        self.m_output_reso = _tmp_f(output_reso, 1, len(output_dims))
        self.m_output_norm = _tmp_f(output_norm, True, len(output_dims))
        if len(self.m_input_reso) != len(self.m_input_dims):
            nii_warn.f_die("Please check input_reso")
        if len(self.m_output_reso) != len(self.m_output_dims):
            nii_warn.f_die("Please check output_reso")
        if len(self.m_input_norm) != len(self.m_input_dims):
            nii_warn.f_die("Please check input_norm")
        if len(self.m_output_norm) != len(self.m_output_dims):
            nii_warn.f_die("Please check output_norm")
        
        # dimensions
        self.m_input_all_dim = sum(self.m_input_dims)
        self.m_output_all_dim = sum(self.m_output_dims)
        self.m_io_dim = self.m_input_all_dim + self.m_output_all_dim

        self.m_truncate_seq = truncate_seq
        self.m_min_seq_len = min_seq_len
        self.m_save_ms = save_mean_std

        # in case there is waveform data in input or output features 
        self.m_wav_sr = wav_samp_rate
            
        # sanity check on resolution configuration
        # currently, only input features can have different reso,
        # and the m_input_reso must be the same for all input features
        if any([x != self.m_input_reso[0] for x in self.m_input_reso]):
            nii_warn.f_print("input_reso: %s" % (str(self.m_input_reso)),\
                             'error')
            nii_warn.f_print("NIIDataSet not support", 'error', end='')
            nii_warn.f_die(" different input_reso")

        if any([x != self.m_output_reso[0] for x in self.m_output_reso]):
            nii_warn.f_print("output_reso: %s" % (str(self.m_output_reso)),\
                             'error')
            nii_warn.f_print("NIIDataSet not support", 'error', end='')
            nii_warn.f_die(" different input_reso")
        # no need to contrain output_reso = 1
        #if any([x != 1 for x in self.m_output_reso]):
        #    nii_warn.f_print("NIIDataSet only supports", 'error', end='')
        #    nii_warn.f_die(" output_reso = [1, 1, ... 1]")
        #self.m_single_reso = self.m_input_reso[0]
        self.m_single_reso = np.max(self.m_input_reso + self.m_output_reso)
        
        # To make sure that target waveform length is exactly equal
        #  to the up-sampled sequence length
        # self.m_truncate_seq must be changed to be N * up_sample
        if self.m_truncate_seq is not None:
            # assume input resolution is the same
            self.m_truncate_seq = self.f_adjust_len(self.m_truncate_seq)

        # method to load/write raw data
        if data_format == '<f4':
            self.f_load_data = _data_reader
            self.f_length_data = _data_len_reader
            self.f_write_data = lambda x, y: _data_writer(x, y, \
                                                          self.m_wav_sr)
        else:
            nii_warn.f_print("Unsupported dtype %s" % (data_format))
            nii_warn.f_die("Only supports np.float32 <f4")
            
        # check the validity of data
        self.f_check_file_list()
        
        # log down statiscs 
        #  1. length of each data utterance
        #  2. mean / std of feature feature file
        def get_name(stats_path, set_name, file_name):
            tmp = set_name + '_' + file_name
            return os.path.join(stats_path, tmp)
        
        self.m_ms_input_path = get_name(stats_path, self.m_set_name, \
                                        nii_dconf.mean_std_i_file)
        self.m_ms_output_path = get_name(stats_path, self.m_set_name, \
                                         nii_dconf.mean_std_o_file)
        self.m_data_len_path = get_name(stats_path, self.m_set_name, \
                                        nii_dconf.data_len_file)
        
        # initialize data length and mean /std
        flag_cal_len = self.f_init_data_len_stats(self.m_data_len_path)
        flag_cal_mean_std = self.f_init_mean_std(self.m_ms_input_path,
                                                 self.m_ms_output_path)
            
        # if data information is not available, read it again from data
        if flag_cal_len or flag_cal_mean_std:
            self.f_calculate_stats(flag_cal_len, flag_cal_mean_std) 
            
        # check
        if self.__len__() < 1:
            nii_warn.f_print("Fail to load any data", "error")
            nii_warn.f_die("Please check configuration")
        # done
        return                
        
    def __len__(self):
        """ __len__():
        Return the number of samples in the list
        """
        return len(self.m_seq_info)

    def __getitem__(self, idx):
        """ __getitem__(self, idx):
        Return input, output
        
        For test set data, output can be None
        """
        try:
            tmp_seq_info = self.m_seq_info[idx]
        except IndexError:
            nii_warn.f_die("Sample %d is not in seq_info" % (idx))

        # file_name
        file_name = tmp_seq_info.seq_tag()
        
        # For input data
        input_reso = self.m_input_reso[0]
        seq_len = int(tmp_seq_info.seq_length() // input_reso)
        s_idx = int(tmp_seq_info.seq_start_pos() // input_reso)
        e_idx = s_idx + seq_len
        
        input_dim = self.m_input_all_dim
        in_data = np.zeros([seq_len, input_dim], dtype=nii_dconf.h_dtype)
        s_dim = 0
        e_dim = 0

        # loop over each feature type
        for t_dir, t_ext, t_dim, t_res in \
            zip(self.m_input_dirs, self.m_input_exts, \
                self.m_input_dims, self.m_input_reso):
            e_dim = s_dim + t_dim
            
            # get file path and load data
            file_path = nii_str_tk.f_realpath(t_dir, file_name, t_ext)
            try:
                tmp_d = self.f_load_data(file_path, t_dim) 
            except IOError:
                nii_warn.f_die("Cannot find %s" % (file_path))

            # write data
            if tmp_d.shape[0] == 1:
                # input data has only one frame, duplicate
                if tmp_d.ndim > 1:
                    in_data[:,s_dim:e_dim] = tmp_d[0,:]
                elif t_dim == 1:
                    in_data[:,s_dim] = tmp_d
                else:
                    nii_warn.f_die("Dimension wrong %s" % (file_path))
            else:
                # normal case
                if tmp_d.ndim > 1:
                    # write multi-dimension data
                    in_data[:,s_dim:e_dim] = tmp_d[s_idx:e_idx,:]
                elif t_dim == 1:
                    # write one-dimension data
                    in_data[:,s_dim] = tmp_d[s_idx:e_idx]
                else:
                    nii_warn.f_die("Dimension wrong %s" % (file_path))
            s_dim = e_dim

        # load output data
        if self.m_output_dirs:
            output_reso = self.m_output_reso[0]
            seq_len = int(tmp_seq_info.seq_length() // output_reso)
            s_idx = int(tmp_seq_info.seq_start_pos() // output_reso)
            e_idx = s_idx + seq_len
        
            out_dim = self.m_output_all_dim
            out_data = np.zeros([seq_len, out_dim], \
                                dtype = nii_dconf.h_dtype)
            s_dim = 0
            e_dim = 0
            for t_dir, t_ext, t_dim in zip(self.m_output_dirs, \
                                           self.m_output_exts, \
                                           self.m_output_dims):
                e_dim = s_dim + t_dim
                # get file path and load data
                file_path = nii_str_tk.f_realpath(t_dir, file_name, t_ext)
                try:
                    tmp_d = self.f_load_data(file_path, t_dim) 
                except IOError:
                    nii_warn.f_die("Cannot find %s" % (file_path))

                if tmp_d.shape[0] == 1:
                    if tmp_d.ndim > 1:
                        out_data[:,s_dim:e_dim] = tmp_d[0,:]
                    elif t_dim == 1:
                        out_data[:,s_dim]=tmp_d
                    else:
                        nii_warn.f_die("Dimension wrong %s" % (file_path))
                else:
                    if tmp_d.ndim > 1:
                        out_data[:,s_dim:e_dim] = tmp_d[s_idx:e_idx,:]
                    elif t_dim == 1:
                        out_data[:,s_dim]=tmp_d[s_idx:e_idx]
                    else:
                        nii_warn.f_die("Dimension wrong %s" % (file_path))
                s_dim = s_dim + t_dim
        else:
            out_data = []

        return in_data, out_data, tmp_seq_info.print_to_str(), idx
    
    def f_get_num_seq(self):
        """ __len__():
        Return the number of samples in the list
        """
        return len(self.m_seq_info)

    
    def f_get_mean_std_tuple(self):
        return (self.m_input_mean, self.m_input_std,
                self.m_output_mean, self.m_output_std)

    
    def f_check_file_list(self):
        """ f_check_file_list():
            Check the file list after initialization
            Make sure that the file in file_list appears in every 
            input/output feature directory. 
            If not, get a file_list in which every file is avaiable
            in every input/output directory
        """
        if not isinstance(self.m_file_list, list):
            nii_warn.f_print("Read file list from directories")
            self.m_file_list = None
        
        #  get a initial file list
        if self.m_file_list is None:
            self.m_file_list = nii_list_tools.listdir_with_ext(
                self.m_input_dirs[0], self.m_input_exts[0])

        # check the list of files exist in all input/output directories
        for tmp_d, tmp_e in zip(self.m_input_dirs, \
                                self.m_input_exts):
            tmp_list = nii_list_tools.listdir_with_ext(tmp_d, tmp_e)
            self.m_file_list = nii_list_tools.common_members(
                tmp_list, self.m_file_list)

        if len(self.m_file_list) < 1:
            nii_warn.f_print("No input features after scannning", 'error')
            nii_warn.f_print("Please check input config", 'error')
            nii_warn.f_print("Please check feature directory", 'error')

        # check output files if necessary
        if self.m_output_dirs:
            for tmp_d, tmp_e in zip(self.m_output_dirs, \
                                    self.m_output_exts):
                tmp_list = nii_list_tools.listdir_with_ext(tmp_d, tmp_e)
                self.m_file_list = nii_list_tools.common_members(
                    tmp_list, self.m_file_list)

            if len(self.m_file_list) < 1:
                nii_warn.f_print("No output data found", 'error')
                nii_warn.f_die("Please check outpupt config")
        else:
            #nii_warn.f_print("Not loading output features")
            pass
        
        # done
        return
        

    def f_valid_len(self, t_1, t_2, min_length):
        """ f_valid_time_steps(time_step1, time_step2, min_length)
        When either t_1 > min_length or t_2 > min_length, check whether 
        time_step1 and time_step2 are too different       
        """
        if max(t_1, t_2) > min_length:
            if (np.abs(t_1 - t_2) * 1.0 / t_1) > 0.1:
                return False
        return True

    def f_check_specific_data(self, file_name):
        """ check the data length of a specific file
        """
        tmp_dirs = self.m_input_dirs.copy()
        tmp_exts = self.m_input_exts.copy()
        tmp_dims = self.m_input_dims.copy()
        tmp_reso = self.m_input_reso.copy()
        tmp_dirs.extend(self.m_output_dirs)
        tmp_exts.extend(self.m_output_exts)
        tmp_dims.extend(self.m_output_dims)
        tmp_reso.extend(self.m_output_reso)        
        
        # loop over each input/output feature type
        for t_dir, t_ext, t_dim, t_res in \
            zip(tmp_dirs, tmp_exts, tmp_dims, tmp_reso):

            file_path = nii_str_tk.f_realpath(t_dir, file_name, t_ext)
            if not nii_io_tk.file_exist(file_path):
                nii_warn.f_die("%s not found" % (file_path))
            else:        
                t_len  = self.f_length_data(file_path) // t_dim
                print("%s, length %d, dim %d, reso: %d" % \
                      (file_path, t_len, t_dim, t_res))
        return

        
    def f_log_data_len(self, file_name, t_len, t_reso):
        """ f_log_data_len(file_name, t_len, t_reso):
        Log down the length of the data file.

        When comparing the different input/output features for the same
        file_name, only keep the shortest length
        """
        # the length for the sequence with the fast tempoeral rate
        # For example, acoustic-feature -> waveform 16kHz,
        # if acoustic-feature is one frame per 5ms,
        #  tmp_len = acoustic feature frame length * (5 * 16)
        # where t_reso = 5*16 is the up-sampling rate of acoustic feature
        tmp_len = t_len * t_reso
        
        # save length when have not read the file
        if file_name not in self.m_data_length:
            self.m_data_length[file_name] = tmp_len

        # check length
        if t_len == 1:
            # if this is an utterance-level feature, it has only 1 frame
            pass
        elif self.f_valid_len(self.m_data_length[file_name], tmp_len, \
                            nii_dconf.data_seq_min_length):
            # if the difference in length is small
            if self.m_data_length[file_name] > tmp_len:
                self.m_data_length[file_name] = tmp_len
        else:
            nii_warn.f_print("Sequence length mismatch:", 'error')
            self.f_check_specific_data(file_name)
            nii_warn.f_print("Please the above features", 'error')
            nii_warn.f_die("Possible invalid data %s" % (file_name))

        # adjust the length so that, when reso is used,
        # the sequence length will be N * reso
        tmp = self.m_data_length[file_name]
        self.m_data_length[file_name] = self.f_adjust_len(tmp)
        return

    def f_adjust_len(self, length):
        """ When input data will be up-sampled by self.m_single_reso,
        Make sure that the sequence length at the up-sampled level is
         = N * self.m_single_reso
        For data without up-sampling m_single_reso = 1
        """
        return length // self.m_single_reso * self.m_single_reso
    
    def f_log_seq_info(self):
        """ After m_data_length has been created, create seq_info
        
        """
        for file_name in self.m_file_list:

            # if file_name is not logged, ignore this file
            if file_name not in self.m_data_length:
                continue
            
            # if not truncate, save the seq_info directly
            # otherwise, save truncate_seq info
            length_remain = self.m_data_length[file_name]
            start_pos = 0
            seg_idx = 0
            if self.m_truncate_seq is not None:
                while(length_remain > 0):
                    info_idx = len(self.m_seq_info)
                    seg_length = min(self.m_truncate_seq, length_remain)
                    seq_info = nii_seqinfo.SeqInfo(seg_length, 
                                                   file_name, seg_idx,
                                                   start_pos, info_idx)
                    if self.m_min_seq_len is None or \
                       seg_length >= self.m_min_seq_len:
                        self.m_seq_info.append(seq_info)
                        seg_idx += 1
                    start_pos += seg_length
                    length_remain -= seg_length
            else:
                info_idx = len(self.m_seq_info)
                seq_info = nii_seqinfo.SeqInfo(length_remain,
                                               file_name, seg_idx,
                                               start_pos, info_idx)
                if self.m_min_seq_len is None or \
                   length_remain >= self.m_min_seq_len:
                    self.m_seq_info.append(seq_info)
        
        # get the total length
        self.m_data_total_length = self.f_sum_data_length()
        return
        
    def f_init_mean_std(self, ms_input_path, ms_output_path):
        """ f_init_mean_std
        Initialzie mean and std vectors for input and output
        """
        self.m_input_mean = np.zeros([self.m_input_all_dim])
        self.m_input_std = np.ones([self.m_input_all_dim])
        self.m_output_mean = np.zeros([self.m_output_all_dim])
        self.m_output_std = np.ones([self.m_output_all_dim])
        
        flag = True
        if not self.m_save_ms:
            # assume mean/std will be in the network
            flag = False

        if os.path.isfile(ms_input_path) and \
           os.path.isfile(ms_output_path):
            # load mean and std if exists
            ms_input = self.f_load_data(ms_input_path, 1)
            ms_output = self.f_load_data(ms_output_path, 1)
            
            if ms_input.shape[0] != (self.m_input_all_dim * 2) or \
               ms_output.shape[0] != (self.m_output_all_dim * 2):
                if ms_input.shape[0] != (self.m_input_all_dim * 2):
                    nii_warn.f_print("%s incompatible" % (m_input_path),
                                     'warning')
                if ms_output.shape[0] != (self.m_output_all_dim * 2):
                    nii_warn.f_print("%s incompatible" % (m_output_path),
                                     'warning')
                nii_warn.f_print("mean/std will be recomputed", 'warning')
            else:
                self.m_input_mean = ms_input[0:self.m_input_all_dim]
                self.m_input_std = ms_input[self.m_input_all_dim:]
                
                self.m_output_mean = ms_output[0:self.m_output_all_dim]
                self.m_output_std = ms_output[self.m_output_all_dim:]
                nii_warn.f_print("Load mean/std from %s and %s" % \
                                 (ms_input_path, ms_output_path))
                flag = False
        return flag


    def f_sum_data_length(self):
        """
        """
        
        return sum([x.seq_length() for x in self.m_seq_info])
        
    def f_init_data_len_stats(self, data_path):
        """
        flag = f_init_data_len_stats(self, data_path)
        Check whether data length has been stored in data_pat.
        If yes, load data_path and return False
        Else, return True
        """
        self.m_seq_info = []
        self.m_data_length = {}
        self.m_data_total_length = 0
        
        flag = True
        if os.path.isfile(data_path):
            # load data length from pre-stored *.dic
            dic_seq_infos = nii_io_tk.read_dic(self.m_data_len_path)
            for dic_seq_info in dic_seq_infos:
                seq_info = nii_seqinfo.SeqInfo()
                seq_info.load_from_dic(dic_seq_info)
                self.m_seq_info.append(seq_info)
                seq_tag = seq_info.seq_tag()
                if seq_tag not in self.m_data_length:
                    self.m_data_length[seq_tag] = seq_info.seq_length()
                else:
                    self.m_data_length[seq_tag] += seq_info.seq_length()
            self.m_data_total_length = self.f_sum_data_length()
            
            # check whether *.dic contains files in filelist
            if nii_list_tools.list_identical(self.m_file_list,\
                                             self.m_data_length.keys()):
                nii_warn.f_print("Read sequence info: %s" % (data_path))
                flag = False
            elif nii_list_tools.list_b_in_list_a(self.m_file_list, 
                                                 self.m_data_length.keys()):
                nii_warn.f_print("Read sequence info: %s" % (data_path))
                nii_warn.f_print(
                    "However %d samples are ignoed %d" % \
                    (len(self.m_file_list)-len(self.m_data_length)))
                flag = False
            else:
                self.m_seq_info = []
                self.m_data_length = {}
                self.m_data_total_length = 0

        return flag

    def f_save_data_len(self, data_len_path):
        """
        """
        nii_io_tk.write_dic([x.print_to_dic() for x in self.m_seq_info], \
                            data_len_path)
        
    def f_save_mean_std(self, ms_input_path, ms_output_path):
        """
        """
        # save mean and std
        ms_input = np.zeros([self.m_input_all_dim * 2])
        ms_input[0:self.m_input_all_dim] = self.m_input_mean
        ms_input[self.m_input_all_dim :] = self.m_input_std
        self.f_write_data(ms_input, ms_input_path)

        ms_output = np.zeros([self.m_output_all_dim * 2])
        ms_output[0:self.m_output_all_dim] = self.m_output_mean
        ms_output[self.m_output_all_dim :] = self.m_output_std
        self.f_write_data(ms_output, ms_output_path)

        return

    def f_print_info(self):
        """
        """
        mes = "Dataset {}:".format(self.m_set_name)
        mes += "\n  Time steps: {:d} ".format(self.m_data_total_length)
        if self.m_truncate_seq is not None:
            mes += "\n  Truncate length: {:d}".format(self.m_truncate_seq)
        mes += "\n  Data sequence num: {:d}".format(len(self.m_seq_info))
        tmp_min_len = min([x.seq_length() for x in self.m_seq_info])
        tmp_max_len = max([x.seq_length() for x in self.m_seq_info])
        mes += "\n  Maximum sequence length: {:d}".format(tmp_max_len)
        mes += "\n  Minimum sequence length: {:d}".format(tmp_min_len)
        if self.m_min_seq_len is not None:
            mes += "\n  Shorter sequences are ignored"
        mes += "\n  Inputs\n    Dirs:"
        for subdir in self.m_input_dirs:
            mes += "\n        {:s}".format(subdir)
        mes += "\n    Exts:{:s}".format(str(self.m_input_exts))
        mes += "\n    Dims:{:s}".format(str(self.m_input_dims))
        mes += "\n    Reso:{:s}".format(str(self.m_input_reso))
        mes += "\n    Norm:{:s}".format(str(self.m_input_norm))
        mes += "\n  Outputs\n    Dirs:"
        for subdir in  self.m_output_dirs:
            mes += "\n        {:s}".format(subdir)
        mes += "\n    Exts:{:s}".format(str(self.m_output_exts))
        mes += "\n    Dims:{:s}".format(str(self.m_output_dims))
        mes += "\n    Reso:{:s}".format(str(self.m_output_reso))
        mes += "\n    Norm:{:s}".format(str(self.m_output_norm))
        nii_warn.f_print_message(mes)
        return
    
    def f_calculate_stats(self, flag_cal_data_len, flag_cal_mean_std):
        """ f_calculate_stats
        Log down the number of time steps for each file
        Calculate the mean/std
        """
        # check
        #if not self.m_output_dirs:
        #    nii_warn.f_print("Calculating mean/std", 'error')
        #    nii_warn.f_die("But output_dirs is not provided")

        # prepare the directory, extension, and dimensions
        tmp_dirs = self.m_input_dirs.copy()
        tmp_exts = self.m_input_exts.copy()
        tmp_dims = self.m_input_dims.copy()
        tmp_reso = self.m_input_reso.copy()
        tmp_norm = self.m_input_norm.copy()        
        tmp_dirs.extend(self.m_output_dirs)
        tmp_exts.extend(self.m_output_exts)
        tmp_dims.extend(self.m_output_dims)
        tmp_reso.extend(self.m_output_reso)
        tmp_norm.extend(self.m_output_norm)
        
        # starting dimension of one type of feature
        s_dim = 0
        # ending dimension of one type of feature        
        e_dim = 0
        
        # loop over each input/output feature type
        for t_dir, t_ext, t_dim, t_reso, t_norm in \
            zip(tmp_dirs, tmp_exts, tmp_dims, tmp_reso, tmp_norm):
            
            s_dim = e_dim
            e_dim = s_dim + t_dim
            t_cnt = 0
            mean_i, var_i = np.zeros([t_dim]), np.zeros([t_dim])
            
            # loop over all the data
            for file_name in self.m_file_list:
                # get file path
                file_path = nii_str_tk.f_realpath(t_dir, file_name, t_ext)
                if not nii_io_tk.file_exist(file_path):
                    nii_warn.f_die("%s not found" % (file_path))
                    
                # read the length of the data
                if flag_cal_data_len:
                    t_len  = self.f_length_data(file_path) // t_dim
                    self.f_log_data_len(file_name, t_len, t_reso)
                    
                    
                # accumulate the mean/std recursively
                if flag_cal_mean_std:
                    t_data  = self.f_load_data(file_path, t_dim)

                    # if the is F0 data, only consider voiced data
                    if t_ext in nii_dconf.f0_unvoiced_dic:
                        unvoiced_value = nii_dconf.f0_unvoiced_dic[t_ext]
                        t_data = t_data[t_data > unvoiced_value]
                    # mean_i, var_i, t_cnt will be updated using online
                    # accumulation method
                    mean_i, var_i, t_cnt = nii_stats.f_online_mean_std(
                        t_data, mean_i, var_i, t_cnt)

            # save mean and std for one feature type
            if flag_cal_mean_std:
                # if not normalize this dimension, set mean=0, std=1
                if not t_norm:
                    mean_i[:] = 0
                    var_i[:] = 1
                    
                if s_dim < self.m_input_all_dim:
                    self.m_input_mean[s_dim:e_dim] = mean_i

                    std_i = nii_stats.f_var2std(var_i)
                    self.m_input_std[s_dim:e_dim] = std_i
                else:
                    tmp_s = s_dim - self.m_input_all_dim
                    tmp_e = e_dim - self.m_input_all_dim
                    self.m_output_mean[tmp_s:tmp_e] = mean_i
                    std_i = nii_stats.f_var2std(var_i)
                    self.m_output_std[tmp_s:tmp_e] = std_i

        if flag_cal_data_len:
            # create seq_info
            self.f_log_seq_info()
            self.f_save_data_len(self.m_data_len_path)
            
        if flag_cal_mean_std:
            self.f_save_mean_std(self.m_ms_input_path,
                                 self.m_ms_output_path)
        # done
        return
        
    def f_putitem(self, output_data, save_dir, data_infor_str):
        """ 
        """
        # Change the dimension to (length, dim)
        if output_data.ndim == 3 and output_data.shape[0] == 1:
            # When input data is (batchsize=1, length, dim)
            output_data = output_data[0]
        elif output_data.ndim == 2 and output_data.shape[0] == 1:
            # When input data is (batchsize=1, length)
            output_data = np.expand_dims(output_data[0], -1)
        else:
            nii_warn.f_print("Output data format not supported.", "error")
            nii_warn.f_print("Format is not (batch, len, dim)", "error")
            nii_warn.f_die("Please use batch_size = 1 in generation")

        # Save output
        if output_data.shape[1] != self.m_output_all_dim:
            nii_warn.f_print("Output data dim != expected dim", "error")
            nii_warn.f_print("Output:%d" % (output_data.shape[1]), \
                             "error")
            nii_warn.f_print("Expected:%d" % (self.m_output_all_dim), \
                             "error")
            nii_warn.f_die("Please check configuration")
        
        if not os.path.isdir(save_dir):
            try:
                os.mkdir(save_dir)
            except OSError:
                nii_warn.f_die("Cannot carete {}".format(save_dir))

        # read the sentence information
        tmp_seq_info = nii_seqinfo.SeqInfo()
        tmp_seq_info.parse_from_str(data_infor_str)

        # write the data
        file_name = tmp_seq_info.seq_tag()
        s_dim = 0
        e_dim = 0
        for t_ext, t_dim in zip(self.m_output_exts, self.m_output_dims):
            e_dim = s_dim + t_dim
            file_path = nii_str_tk.f_realpath(save_dir, file_name, t_ext)
            self.f_write_data(output_data[:, s_dim:e_dim], file_path)
        
        return

    def f_input_dim(self):
        """
        f_input_dim()
        return the total dimension of input features
        """ 
        return self.m_input_all_dim
    
    def f_output_dim(self):
        """
        f_output_dim
        return the total dimension of output features
        """
        return self.m_output_all_dim
    
class NIIDataSetLoader:
    """ NIIDataSetLoader:
    A wrapper over torch.utils.data.DataLoader 
    
    self.m_dataset will be the dataset
    self.m_loader  will be the dataloader
    """
    def __init__(self,
                 dataset_name, \
                 file_list, \
                 input_dirs, input_exts, input_dims, input_reso, \
                 input_norm, \
                 output_dirs, output_exts, output_dims, output_reso, \
                 output_norm, \
                 stats_path, \
                 data_format = '<f4', \
                 params = None, \
                 truncate_seq = None, \
                 min_seq_len = None,
                 save_mean_std = True, \
                 wav_samp_rate = None):
        """
        NIIDataSetLoader(
                 data_set_name,
                 file_list,
                 input_dirs,
                 input_exts,
                 input_dims,
                 input_reso,
                 input_norm,
                 output_dirs,
                 output_exts,
                 output_dims,
                 output_reso,
                 output_norm,
                 stats_path,
                 data_format = '<f4',
                 params = None,
                 truncate_seq = None):
        Args:
            data_set_name: a string to name this dataset
                           this will be used to name the statistics files
                           such as the mean/std for this dataset
            file_list: a list of file name strings (without extension)
            input_dirs: a list of dirs from each input feature is loaded
            input_exts: a list of input feature name extentions
            input_dims: a list of input feature dimensions
            input_reso: a list of input feature temporal resolution,
                        or None
            output_dirs: a list of dirs from each output feature is loaded
            output_exts: a list of output feature name extentions
            output_dims: a list of output feature dimensions
            output_reso: a list of output feature temporal resolution, 
                         or None
            stats_path: path to the directory of statistics(mean/std)
            data_format: method to load the data
                    '<f4' (default): load data as float32m little-endian
                    'htk': load data as htk format
            params: parameter for torch.utils.data.DataLoader
            truncate_seq: None or int, 
                          truncate data sequence into smaller truncks
                          truncate_seq > 0 specifies the trunck length
        Methods:
        get_loader(): return a torch.util.data.DataLoader
        get_dataset(): return a torch.util.data.DataSet
        """
        nii_warn.f_print_w_date("Loading dataset %s" % (dataset_name),
                                level="h")
        
        # create torch.util.data.DataSet
        self.m_dataset = NIIDataSet(dataset_name, \
                                    file_list, \
                                    input_dirs, input_exts, \
                                    input_dims, input_reso, \
                                    input_norm, \
                                    output_dirs, output_exts, \
                                    output_dims, output_reso, \
                                    output_norm, \
                                    stats_path, data_format, \
                                    truncate_seq, min_seq_len,\
                                    save_mean_std, \
                                    wav_samp_rate)
        
        # create torch.util.data.DataLoader
        if params is None:
            tmp_params = nii_dconf.default_loader_conf
        else:
            tmp_params = params
        self.m_loader = torch.utils.data.DataLoader(self.m_dataset,
                                                    **tmp_params)
        # done
        return
        
    
    def get_loader(self):
        """ get_loader():
        Return the dataLoader (torch.util.data.DataLoader)
        """
        return self.m_loader
    
    def get_dataset(self):
        """ get_dataset():
        Return the dataset (torch.util.data.Dataset)
        """
        return self.m_dataset

    def get_data_mean_std(self):
        """
        """
        return self.m_dataset.f_get_mean_std_tuple()

    def print_info(self):
        """
        """
        self.m_dataset.f_print_info()

    def putitem(self, output_data, save_dir, data_infor_str):
        """ Decompose the output_data from network into
        separate files
        """
        self.m_dataset.f_putitem(output_data, save_dir, data_infor_str)

    def get_in_dim(self):
        """ Return the dimension of input features
        """ 
        return self.m_dataset.f_input_dim()

    def get_out_dim(self):
        """ Return the dimension of output features
        """
        return self.m_dataset.f_output_dim()

    def get_seq_num(self):
        """ Return the number of sequences (after truncation)
        """ 
        return self.m_dataset.f_get_num_seq()
    
if __name__ == "__main__":
    pass
