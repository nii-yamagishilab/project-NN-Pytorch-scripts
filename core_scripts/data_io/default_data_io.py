#!/usr/bin/env python
"""
data_io

Interface to load data

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
from inspect import signature
import torch
import torch.utils.data

import core_scripts.other_tools.list_tools as nii_list_tools
import core_scripts.other_tools.display as nii_warn
import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.data_io.io_tools as nii_io_tk
import core_scripts.data_io.wav_tools as nii_wav_tk
import core_scripts.data_io.text_process.text_io as nii_text_tk
import core_scripts.data_io.conf as nii_dconf

import core_scripts.data_io.seq_info as nii_seqinfo
import core_scripts.math_tools.stats as nii_stats
import core_scripts.data_io.customize_collate_fn as nii_collate_fn
import core_scripts.data_io.customize_sampler as nii_sampler_fn

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

###
## functions wrappers to read/write data for this data_io
###
def _data_reader(file_path, dim, flag_lang, g2p_tool):
    """ A wrapper to read raw binary data, waveform, or text
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.wav':
        sr, data = nii_wav_tk.waveReadAsFloat(file_path)
        if data.ndim > 1 and data.shape[-1] != dim:
            nii_warn.f_print("Expect {:d} channel(s)".format(dim), 'error')
            nii_warn.f_die("But {:s} has {:d} channel(s)".format(
                file_path, data.shape[-1]))
    elif file_ext == '.flac':
        sr, data = nii_wav_tk.flacReadAsFloat(file_path)
        if data.ndim > 1 and data.shape[-1] != dim:
            nii_warn.f_print("Expect {:d} channel(s)".format(dim), 'error')
            nii_warn.f_die("But {:s} has {:d} channel(s)".format(
                file_path, data.shape[-1]))
    elif file_ext == '.txt':
        data = nii_text_tk.textloader(file_path, flag_lang, g2p_tool)
    else:
        data = nii_io_tk.f_read_raw_mat(file_path, dim)
    return data

def _data_writer(data, file_path, sr = 16000):
    """ A wrapper to write raw binary data or waveform
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.wav':
        nii_wav_tk.waveFloatToPCMFile(data, file_path, sr = sr)
    elif file_ext == '.txt':
        nii_warn.f_die("Cannot write to {:s}".format(file_path))
    else:
        nii_io_tk.f_write_raw_mat(data, file_path)
    return

def _data_len_reader(file_path):
    """ A wrapper to read length of data
    """
    file_name, file_ext = os.path.splitext(file_path)
    if file_ext == '.wav':
        #sr, data = nii_wav_tk.waveReadAsFloat(file_path)
        #length = data.shape[0]
        length = nii_wav_tk.readWaveLength(file_path)
    elif file_ext == '.flac':
        sr, data = nii_wav_tk.flacReadAsFloat(file_path)
        length = data.shape[0]
    elif file_ext == '.txt':
        # txt, no need to account length
        # note that this is for tts task
        length = 0
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
                 data_format = nii_dconf.h_dtype_str, \
                 truncate_seq = None, \
                 min_seq_len = None, \
                 save_mean_std = True, \
                 wav_samp_rate = None, \
                 flag_lang = 'EN', \
                 global_arg = None, \
                 dset_config = None, \
                 input_augment_funcs = None, \
                 output_augment_funcs = None,
                 inoutput_augment_func = None):
        """
        args
        ----
          dataset_name: name of this data set
          file_list: a list of file name strings (without extension)
                     or, path to the file that contains the file names
          input_dirs: a list of dirs from which input feature is loaded
          input_exts: a list of input feature name extentions
          input_dims: a list of input feature dimensions
          input_reso: a list of input feature temporal resolutions
          input_norm: a list of bool, whether normalize input feature or not
          output_dirs: a list of dirs from which output feature is loaded
          output_exts: a list of output feature name extentions
          output_dims: a list of output feature dimensions
          output_reso: a list of output feature temporal resolutions
          output_norm: a list of bool, whether normalize target feature or not
          stat_path: path to the directory that saves mean/std, 
                     utterance length
          data_format: method to load the data
                    '<f4' (default): load data as float32m little-endian
                    'htk': load data as htk format
          truncate_seq: None (default) or int, truncate sequence into truncks.
                        truncate_seq > 0 specifies the trunck length 
          min_seq_len: None (default) or int, minimum length of an utterance
                        utterance shorter than min_seq_len will be ignored
          save_mean_std: bool, True (default): save mean and std 
          wav_samp_rate: None (default) or int, if input data has  waveform, 
                         please set sampling rate. It is used by _data_writer
          flag_lang: str, 'EN' (default), if input data has text, the text will
                     be converted into code indices. flag_lang indicates the 
                     language for the text processer. It is used by _data_reader
          global_arg: argument parser returned by arg_parse.f_args_parsed()
                      default None
          dset_config: object, dataset configuration, default None
          input_augment_funcs: list of functions for input data transformation
                               default None
          output_augment_funcs: list of output data transformation functions
                                default None
          inoutput_augment_func: a single data augmentation function, 
                                default None

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
            nii_warn.f_die("len(input_reso) != len(input_dims) in config")
        if len(self.m_output_reso) != len(self.m_output_dims):
            nii_warn.f_die("len(output_reso) != len(input_dims) in config")
        if len(self.m_input_norm) != len(self.m_input_dims):
            nii_warn.f_die("len(input_norm) != len(input_dims) in config")
        if len(self.m_output_norm) != len(self.m_output_dims):
            nii_warn.f_die("len(output_norm) != len(output_dims) in config")

        
        if global_arg is not None:
            self.m_ignore_length_invalid = global_arg.ignore_length_invalid_data
            self.m_ignore_cached_finfo = global_arg.ignore_cached_file_infor
            self.m_force_skip_scanning = global_arg.force_skip_datadir_scanning
        else:
            self.m_ignore_length_invalid = False
            self.m_ignore_cached_finfo = False
            self.m_force_skip_scanning = False

        # check augmentation funcctions
        if input_augment_funcs:
            if len(input_augment_funcs) != len(self.m_input_dims):
                nii_warn.f_die("len(input_augment_funcs) != len(input_dims)")
            self.m_inaug_funcs = input_augment_funcs
        else:
            self.m_inaug_funcs = []
            
        if output_augment_funcs:
            if len(output_augment_funcs) != len(self.m_output_dims):
                nii_warn.f_die("len(output_augment_funcs) != len(output_dims)")
            self.m_ouaug_funcs = output_augment_funcs
        else:
            self.m_ouaug_funcs = []

        if inoutput_augment_func:
            self.m_inouaug_func = inoutput_augment_func
        else:
            self.m_inouaug_func = None

        # dimensions
        self.m_input_all_dim = sum(self.m_input_dims)
        self.m_output_all_dim = sum(self.m_output_dims)
        self.m_io_dim = self.m_input_all_dim + self.m_output_all_dim

        self.m_truncate_seq = truncate_seq
        self.m_min_seq_len = min_seq_len
        self.m_save_ms = save_mean_std

        # in case there is waveform data in input or output features 
        self.m_wav_sr = wav_samp_rate
        # option to process waveform with simple VAD
        if global_arg is not None:
            self.m_opt_wav_handler = global_arg.opt_wav_silence_handler
        else:
            self.m_opt_wav_handler = 0

        # in case there is text data in input or output features
        self.m_flag_lang = flag_lang
        self.m_g2p_tool = None
        if hasattr(dset_config, 'text_process_options') and \
           type(dset_config.text_process_options) is dict:
            self.m_flag_lang = dset_config.text_process_options['flag_lang']
            if 'g2p_tool' in dset_config.text_process_options:
                self.m_g2p_tool = dset_config.text_process_options['g2p_tool']
        
        # sanity check on resolution configuration
        # currently, only input features can have different reso,
        # and the m_input_reso must be the same for all input features
        if any([x != self.m_input_reso[0] for x in self.m_input_reso]):
            nii_warn.f_print("input_reso {:s}".format(str(self.m_input_reso)),
                             'error')
            nii_warn.f_print("NIIDataSet not support", 'error', end='')
            nii_warn.f_die(" different input_reso")

        if any([x != self.m_output_reso[0] for x in self.m_output_reso]):
            nii_warn.f_print("output_reso {:s}".format(str(self.m_output_reso)),
                             'error')
            nii_warn.f_print("NIIDataSet not support", 'error', end='')
            nii_warn.f_die(" different output_reso")
        if np.any(np.array(self.m_output_reso) < 0):
            nii_warn.f_print("NIIDataSet not support negative reso", 
                             'error', end='')
            nii_warn.f_die(" Output reso: {:s}".format(str(self.m_output_reso)))
        if np.any(np.array(self.m_input_reso) < 0):
            nii_warn.f_print("input_reso: {:s}".format(str(self.m_input_reso)))
            nii_warn.f_print("Data IO for unaligned input and output pairs")
            if truncate_seq is not None:
                nii_warn.f_print("truncate is set to None", 'warning')
                self.m_truncate_seq = None
                self.m_min_seq_len = None


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

        # similarly on self.m_min_seq_len
        if self.m_min_seq_len is not None:
            # assume input resolution is the same
            self.m_min_seq_len = self.f_adjust_len(self.m_min_seq_len)

        # method to load/write raw data
        if data_format == nii_dconf.h_dtype_str:
            self.f_load_data = lambda x, y: _data_reader(
                x, y, self.m_flag_lang, self.m_g2p_tool)
            self.f_length_data = _data_len_reader
            self.f_write_data = lambda x, y: _data_writer(x, y, self.m_wav_sr)
        else:
            nii_warn.f_print("Unsupported dtype {:s}".format(data_format))
            nii_warn.f_die("Only supports {:s} ".format(nii_dconf.h_dtype_str))
            
        # whether input file name in list contains part of the path
        # this will be confirmed after reading the file list in the next step
        self.flag_filename_with_path = False

        # log down statiscs 
        #  1. length of each data utterance
        #  2. mean / std of feature feature file
        def get_name(stats_path, set_name, file_name):
            tmp = set_name + '_' + file_name
            return os.path.join(stats_path, tmp)
        if global_arg is not None and global_arg.path_cache_file:
            nii_warn.f_print("Cached files are re-directed to {:s}".format(
                global_arg.path_cache_file))
            tmp_stats_path = global_arg.path_cache_file
        else:
            tmp_stats_path = stats_path
        self.m_ms_input_path = get_name(tmp_stats_path, self.m_set_name, \
                                        nii_dconf.mean_std_i_file)
        self.m_ms_output_path = get_name(tmp_stats_path, self.m_set_name, \
                                         nii_dconf.mean_std_o_file)
        self.m_data_len_path = get_name(tmp_stats_path, self.m_set_name, \
                                        nii_dconf.data_len_file)

        # load and check the validity of data list
        self.f_check_file_list(self.m_data_len_path)
                
        # initialize data length and mean /std, read prepared data stats
        flag_cal_len = self.f_init_data_len_stats(self.m_data_len_path)
        flag_cal_mean_std = self.f_init_mean_std(self.m_ms_input_path,
                                                 self.m_ms_output_path)
            
        # if data information is not available, read it again from data
        if flag_cal_len or flag_cal_mean_std:
            self.f_calculate_stats(flag_cal_len, flag_cal_mean_std) 
            

        # if some additional flags are turned on
        if hasattr(global_arg, "flag_reverse_data_loading_order") and \
           global_arg.flag_reverse_data_loading_order:
            self.m_flag_reverse_load_order = True
        else:
            self.m_flag_reverse_load_order = False
        
        #
        if hasattr(global_arg, "force_update_seq_length") and \
           global_arg.force_update_seq_length:
            self.m_force_update_seq_length = True
        else:
            self.m_force_update_seq_length = False
        

        # check
        if self.__len__() < 1:
            nii_warn.f_print("Fail to load any data", "error")
            nii_warn.f_print("Possible reasons: ", "error")
            mes = "1. Old cache {:s}. Do rm it.".format(self.m_data_len_path)
            mes += "\n2. input_dirs, input_exts, "
            mes += "output_dirs, or output_exts incorrect."
            mes += "\n3. all data are less than minimum_len in length. "
            mes += "\nThe last case may happen if truncate_seq == mininum_len "
            mes += "and truncate_seq % input_reso != 0. Then, the actual "
            mes += "truncate_seq becomes truncate_seq//input_reso*input_reso "
            mes += "and it will be shorter than minimum_len. Please change "
            mes += "truncate_seq and minimum_len so that "
            mes += "truncate_seq % input_reso == 0."
            nii_warn.f_print(mes, "error")
            nii_warn.f_die("Please check configuration file")
        # done
        return                
        
    def __len__(self):
        """ __len__():
        Return the number of samples in the list
        """
        return len(self.m_seq_info)

    def __getitem__(self, idx_input):
        """ __getitem__(self, idx):
        Return input, output
        
        For test set data, output can be None
        """
        # option to select the (N - i + 1)-th sample
        if self.m_flag_reverse_load_order:
            idx = len(self.m_seq_info) - idx_input - 1
        else:
            idx = idx_input
        
        # get the sample information 
        try:
            tmp_seq_info = self.m_seq_info[idx]
        except IndexError:
            nii_warn.f_die("Sample {:d} is not in seq_info".format(idx))

        # file_name
        file_name = tmp_seq_info.seq_tag()
        
        # For input data
        input_reso = self.m_input_reso[0]
        seq_len = int(tmp_seq_info.seq_length() // input_reso)
        s_idx = int(tmp_seq_info.seq_start_pos() // input_reso)
        e_idx = s_idx + seq_len
        
        # in case the input length not account using tmp_seq_info.seq_length
        if seq_len < 0:
            seq_len = 0
            s_idx = 0
            e_idx = 0

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
                nii_warn.f_die("Cannot find {:s}".format(file_path))

            # write data
            if t_res < 0:
                # if this is for input data not aligned with output
                # make sure that the input is in shape (seq_len, dim)
                #  f_load_data should return data in shape (seq_len, dim)
                if tmp_d.ndim == 1:
                    in_data = np.expand_dims(tmp_d, axis=1)
                elif tmp_d.ndim == 2:
                    in_data = tmp_d
                else:
                    nii_warn.f_die("IO not support {:s}".format(file_path))
            elif tmp_d.shape[0] == 1:
                # input data has only one frame, duplicate
                if tmp_d.ndim > 1:
                    in_data[:,s_dim:e_dim] = tmp_d[0,:]
                elif t_dim == 1:
                    in_data[:,s_dim] = tmp_d
                else:
                    nii_warn.f_die("Dimension wrong {:s}".format(file_path))
            else:
                # check
                try:
                    # normal case
                    if tmp_d.ndim > 1:
                        # write multi-dimension data
                        in_data[:,s_dim:e_dim] = tmp_d[s_idx:e_idx,:]
                    elif t_dim == 1:
                        # write one-dimension data
                        in_data[:,s_dim] = tmp_d[s_idx:e_idx]
                    else:
                        nii_warn.f_die("Dimension wrong {:s}".format(file_path))
                except ValueError:
                    if in_data.shape[0] != tmp_d[s_idx:e_idx].shape[0]:
                        mes = 'Expected length is {:d}.\n'.format(e_idx-s_idx)
                        mes += "Loaded length "+str(tmp_d[s_idx:e_idx].shape[0])
                        mes += '\nThis may be due to an incompatible cache *.dic.'
                        mes += '\nPlease check the length in *.dic'
                        mes += '\nPlease delete it if the cached length is wrong.'
                        nii_warn.f_print(mes)
                        nii_warn.f_die("fail to load {:s}".format(file_name))
                    else:
                        nii_warn.f_print("unknown data io error")
                        nii_warn.f_die("fail to load {:s}".format(file_name))
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
                    nii_warn.f_die("Cannot find {:s}".format(file_path))

                if tmp_d.shape[0] == 1:
                    if tmp_d.ndim > 1:
                        out_data[:,s_dim:e_dim] = tmp_d[0,:]
                    elif t_dim == 1:
                        out_data[:,s_dim]=tmp_d
                    else:
                        nii_warn.f_die("Dimension wrong {:s}".format(file_path))
                else:
                    try:

                        if tmp_d.ndim > 1:
                            out_data[:,s_dim:e_dim] = tmp_d[s_idx:e_idx,:]
                        elif t_dim == 1:
                            out_data[:,s_dim]=tmp_d[s_idx:e_idx]
                        else:
                            nii_warn.f_die("Dim wrong {:s}".format(file_path))
                    except ValueError:
                        if out_data.shape[0] != tmp_d[s_idx:e_idx].shape[0]:
                            mes = 'Expected length is ' + str(e_idx-s_idx)
                            mes += ". Loaded "+str(tmp_d[s_idx:e_idx].shape[0])
                            mes += 'This may be due to an old cache *.dic.'
                            mes += '\nPlease check the length in *.dic\n'
                            mes += 'Please delete it if cached length is wrong.'
                            nii_warn.f_print(mes)
                            nii_warn.f_die("fail to load " +file_name)
                        else:
                            nii_warn.f_print("unknown data io error")
                            nii_warn.f_die("fail to load " +file_name)

                s_dim = s_dim + t_dim
        else:
            out_data = []
        
        # post processing if necessary
        in_data, out_data, tmp_seq_info, idx = self.f_post_data_process(
            in_data, out_data, tmp_seq_info, idx)

        # return data
        return in_data, out_data, tmp_seq_info.print_to_str(), idx


    def f_post_data_process(self, in_data, out_data, seq_info, idx):
        """A wrapper to process the data after loading from files
        """
        
        if self.m_opt_wav_handler == 0 \
           and not self.m_inaug_funcs and not self.m_ouaug_funcs \
           and not self.m_inouaug_func:
            # no any post-processing process
            return in_data, out_data, seq_info, idx

        else:
            # Do post processing one by one
            # The order is:
            # waveform silence handler -> input augementation functions ->
            # output augmentation functions -> input&output augmentation
            # 
            # Everthing can be handled in input&output augmentation. 
            # But to be compatible with old codes, we keep them all.
            # It is recommended to use the unified input&output augmentation
            
            
            ###
            # buffer infor
            ###
            # create a new sequence information buffer for the input and output
            tmp_seq_info = nii_seqinfo.SeqInfo(
                seq_info.length, seq_info.seq_name, seq_info.seg_idx,
                seq_info.start_pos, seq_info.info_id)
            
            ###
            # waveform silence handler
            ###
            # waveform handler, this is kept for compatibility
            if self.m_opt_wav_handler > 0:
                if len(self.m_input_exts) == 1 \
                   and self.m_input_exts[0][-3:] == 'wav':
                    
                    if self.m_opt_wav_handler == 1:
                        tmp_flag_output = self.m_opt_wav_handler
                        tmp_only_twoends = False
                    elif self.m_opt_wav_handler == 2:
                        tmp_flag_output = self.m_opt_wav_handler
                        tmp_only_twoends = False
                    elif self.m_opt_wav_handler == 3:
                        tmp_flag_output = 1
                        tmp_only_twoends = True
                    else:
                        print("Unknown option for wav handler {:d}".format(
                            self.m_opt_wav_handler))
                        sys.exit(1)

                    in_data_n = nii_wav_tk.silence_handler_wrapper(
                        in_data, self.m_wav_sr, 
                        flag_output = tmp_flag_output,
                        flag_only_startend_sil = tmp_only_twoends)
            
                    # this is temporary setting, if in_data.shape[0] 
                    #  corresponds to waveform length, update it
                    if tmp_seq_info.length == in_data.shape[0]:
                        tmp_seq_info.length = in_data_n.shape[0]
                        if self.m_force_update_seq_length:
                            seq_info.update_len_for_sampler(in_data_n.shape[0])
                else:
                    in_data_n = in_data

                if len(self.m_output_exts) == 1 \
                   and self.m_output_exts[0][-3:] == 'wav':
                    out_data_n = nii_wav_tk.silence_handler_wrapper(
                        out_data, self.m_wav_sr, 
                        flag_output = self.m_opt_wav_handler,
                        flag_only_startend_sil = (self.m_opt_wav_handler==3))
            
                    # this is temporary setting, use length if it is compatible
                    if tmp_seq_info.length == out_data.shape[0]:
                        tmp_seq_info.length = out_data_n.shape[0]
                        if self.m_force_update_seq_length:
                            seq_info.update_len_for_sampler(out_data_n.shape[0])

                else:
                    out_data_n = out_data
            else:
                in_data_n = in_data
                out_data_n = out_data
                
            ###
            # augmentation functions for input data
            ###
            if self.m_inaug_funcs:
                if len(self.m_input_exts) == 1:
                    # only a single input feature, 
                    sig = signature(self.m_inaug_funcs[0])
                    if len(sig.parameters) == 1:
                        in_data_n = self.m_inaug_funcs[0](in_data_n)
                    elif len(sig.parameters) == 2:
                        in_data_n = self.m_inaug_funcs[0](in_data_n, seq_info)
                    else:
                        in_data_n = self.m_inaug_funcs[0](in_data_n)

                    # more rules should be applied to handle the data length
                    # here, simply set length
                    if type(in_data_n) == np.ndarray:
                        if tmp_seq_info.length > in_data_n.shape[0]:
                            tmp_seq_info.length = in_data_n.shape[0]
                    elif type(in_data_n) == dict:
                        if 'length' in in_data_n:
                            tmp_seq_info.length = in_data_n['length']

                        if 'data' in in_data_n:
                            in_data_n = in_data_n['data']
                        else:
                            print("Input data aug method does not return data")
                            sys.exit(1)
                    
                    if self.m_force_update_seq_length:
                        # Update the data length so that correct data length
                        # can be used for --sampler block_shuffle_by_length
                        #
                        #tmp_len = seq_info.length
                        seq_info.update_len_for_sampler(tmp_seq_info.length)
                        #print("{:s} {:s} {:d} -> {:d}".format(
                        #    seq_info.seq_name, seq_info.print_to_str(),
                        #    tmp_len, seq_info.valid_len), 
                        #      flush=True)
                else:
                    # multiple input features, 
                    # must check whether func changes the feature length
                    # only fun that keeps the length will be applied
                    s_dim = 0
                    for func, dim in zip(self.m_inaug_funcs, self.m_input_dims):
                        e_dim = s_dim + dim
                        tmp_data = func(in_data_n[:, s_dim:e_dim])
                        if tmp_data.shape[0] == in_data_n.shape[0]:
                            in_data_n[:, s_dim:e_dim]  = tmp_data
                        s_dim = s_dim + dim
                        
            ###
            # augmentation functions for output data
            ###
            if self.m_ouaug_funcs:
                if len(self.m_output_exts) == 1:
                    # only a single output feature type
                    sig = signature(self.m_ouaug_funcs[0])
                    if len(sig.parameters) == 1:
                        out_data_n = self.m_ouaug_funcs[0](out_data_n)
                    elif len(sig.parameters) == 2:
                        out_data_n = self.m_ouaug_funcs[0](out_data_n, seq_info)
                    else:
                        out_data_n = self.m_ouaug_funcs[0](out_data_n)

                    # more rules should be applied to handle the data length
                    # here, simply set length
                    #if tmp_seq_info.length > out_data_n.shape[0]:
                    #    tmp_seq_info.length = out_data_n.shape[0]
                else:
                    # multiple output features, 
                    # must check whether func changes the feature length
                    # only fun that keeps the length will be applied
                    s_dim = 0
                    for func, dim in zip(self.m_ouaug_funcs,self.m_output_dims):
                        e_dim = s_dim + dim
                        tmp_data = func(out_data_n[:,s_dim:e_dim])
                        if tmp_data.shape[0] == out_data_n.shape[0]:
                            out_data_n[:, s_dim:e_dim] = tmp_data
                        s_dim = s_dim + dim
            
            ###
            # a unified augmentation function for input and output
            ###
            if self.m_inouaug_func:
                # update input output features
                in_data_n, out_data_n, tmp_len = self.m_inouaug_func(
                    in_data_n, out_data_n)
                # update sequence length
                tmp_seq_info.length = tmp_len

                if self.m_force_update_seq_length:
                    seq_info.update_len_for_sampler(tmp_seq_info.length)


            return in_data_n, out_data_n, tmp_seq_info, idx
        
    
    def f_get_num_seq(self):
        """ __len__():
        Return the number of samples in the list
        """
        return len(self.m_seq_info)

    def f_get_seq_len_list(self):
        """ Return length of each sequence as list
        """
        return [x.seq_length() for x in self.m_seq_info]
    
    def f_get_updated_seq_len_for_sampler_list(self):
        """ Similar to f_get_seq_len_list
        but it returns the updated data sequence length only for 
        length-based shuffling in sampler
        """
        return [x.seq_len_for_sampler() for x in self.m_seq_info]

    def f_update_seq_len_for_sampler_list(self, data_idx, data_len):
        try:
            self.m_seq_info[data_idx].update_len_for_sampler(data_len)
        except IndexError:
            nii_warn.f_die("Fail to index data {:d}".format(data_idx))
        return

    def f_get_mean_std_tuple(self):
        return (self.m_input_mean, self.m_input_std,
                self.m_output_mean, self.m_output_std)
    
    def f_filename_has_folderpath(self):
        """ Return True if file name in self.m_file_list contains '/',
        Which indicates that the file name is path/filename
        """
        return any([x.count(os.path.sep)>0 for x in self.m_file_list])
    
    def f_check_file_list(self, data_len_buf_path):
        """ f_check_file_list(data_len_buf_path):
            Check the file list after initialization
            Make sure that the file in file_list appears in every 
            input/output feature directory. 
            If not, get a file_list in which every file is avaiable
            in every input/output directory

        input
        -----
          data_len_buf_path:    str, path to the data length buffer
        """
        
        if self.m_file_list is None:
            # get a initial file list if self.m_file_list is None
            #
            # if file list is not provided, we only search the directory
            #  without recursing sub directories
            self.m_file_list = nii_list_tools.listdir_with_ext(
                self.m_input_dirs[0], self.m_input_exts[0])

        elif not isinstance(self.m_file_list, list):
            # if m_file_list is a string 
            # load file list
            if isinstance(self.m_file_list, str) and \
               os.path.isfile(self.m_file_list):
                # read the list if m_file_list is a str
                self.m_file_list = nii_list_tools.read_list_from_text(
                    self.m_file_list)
            else:
                nii_warn.f_print("Cannot read {:s}".format(self.m_file_list))
                nii_warn.f_print("Read file list from directories")
                self.m_file_list = nii_list_tools.listdir_with_ext(
                self.m_input_dirs[0], self.m_input_exts[0])
        else:
            # self.m_file_list is a list
            pass

        if type(self.m_file_list) is list and len(self.m_file_list) < 1:
            mes = "either input data list is wrong"
            mes += ", or {:s} is empty".format(self.m_input_dirs[0])
            mes += "\nPlease check the folder and data list"
            nii_warn.f_die(mes)
            
        # decide whether the file name in self.m_file_list contains
        #  sub folders
        flag_recur = self.f_filename_has_folderpath()
        self.flag_filename_with_path = flag_recur
        
        # if the stats cache will be loaded, let's skip the checking process
        if os.path.isfile(data_len_buf_path) and not self.m_ignore_cached_finfo:
            nii_warn.f_print("Skip scanning directories")
            return

        # check the list of files exist in all input/output directories
        if not self.m_force_skip_scanning:
            for tmp_d, tmp_e in zip(self.m_input_dirs, self.m_input_exts):
                # read a file list from the input directory
                tmp_list = nii_list_tools.listdir_with_ext(
                    tmp_d, tmp_e, flag_recur)
                # get the common set of the existing files and those in list
                tmp_new_list = nii_list_tools.common_members(
                    tmp_list, self.m_file_list)
            
                if len(tmp_new_list) < 1:
                    nii_warn.f_print("Possible error when scanning:", 'error')
                    nii_warn.f_print(" {:s} for {:s}".format(tmp_d, tmp_e), 'error')
                    nii_warn.f_print('Some file names to be scanned:', 'error')
                    nii_warn.f_print(' ' + ' '.join(self.m_file_list[0:10]),'error')
                    if self.m_file_list[0].endswith(tmp_e):
                        nii_warn.f_print('Names should not have {:s}'.format(tmp_e))
                    if os.path.isfile(self.m_file_list[0]):
                        mes = "The above name seems not to be the data name. "
                        mes += "It seems to be a file path. "
                        mes += "\nPlease check test_list, trn_list, val_list."
                        nii_warn.f_print(mes, 'error')
                    self.m_file_list = tmp_new_list
                    break
                else:
                    self.m_file_list = tmp_new_list

        if len(self.m_file_list) < 1:
            nii_warn.f_print("\nNo input features found after scanning",'error')
            nii_warn.f_print("Please check %s" \
                             % (str(self.m_input_dirs)), 'error')
            nii_warn.f_print("They should contain all files in file list", 
                             'error')
            nii_warn.f_print("Please also check filename extentions %s" \
                             % (str(self.m_input_exts)), 'error')
            nii_warn.f_print("They should be correctly specified", 'error')
            nii_warn.f_die("Failed to read input features")
            
        # check output files if necessary
        if self.m_output_dirs and not self.m_force_skip_scanning:
            for tmp_d, tmp_e in zip(self.m_output_dirs, \
                                    self.m_output_exts):
                tmp_list = nii_list_tools.listdir_with_ext(tmp_d, tmp_e, 
                                                           flag_recur)
                self.m_file_list = nii_list_tools.common_members(
                    tmp_list, self.m_file_list)

            if len(self.m_file_list) < 1:
                nii_warn.f_print("\nNo output data found", 'error')
                nii_warn.f_print("Please check %s" \
                                 % (str(self.m_output_dirs)), 'error')
                nii_warn.f_print("They should contain all files in file list", 
                                 'error')
                nii_warn.f_print("Please also check filename extentions %s" \
                                 % (str(self.m_output_exts)), 'error')
                nii_warn.f_print("They should be correctly specified", 'error')
                nii_warn.f_die("Failed to read output features")
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
        
        # We need to exclude features that should not be considered when
        #  calculating the sequence length
        #  1. sentence-level vector (t_len = 1)
        #  2. unaligned feature (text in text-to-speech) (t_reso < 0)
        valid_flag = t_len > 1 and t_reso > 0
        
        if valid_flag:
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
                # cannot come here, keep this line as history
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
                nii_warn.f_print("Please check the above features", 'error')

                if self.m_ignore_length_invalid:
                    nii_warn.f_print("ignore-length-invalid-data is on")
                    nii_warn.f_print("ignore {:s}".format(file_name))
                    return False
                else:
                    nii_warn.f_print("Or remove them from data list", 'error')
                    nii_warn.f_print("Or --ignore-length-invalid-data",'error')
                    nii_warn.f_die("Possible invalid data %s" % (file_name))

            # adjust the length so that, when reso is used,
            # the sequence length will be N * reso
            tmp = self.m_data_length[file_name]
            self.m_data_length[file_name] = self.f_adjust_len(tmp)
        else:
            # do nothing for unaligned input or sentence-level input
            pass
        
        return True

    def f_adjust_len(self, length):
        """ When input data will be up-sampled by self.m_single_reso,
        Make sure that the sequence length at the up-sampled level is
         = N * self.m_single_reso
        For data without up-sampling m_single_reso = 1
        """
        return length // self.m_single_reso * self.m_single_reso

    def f_precheck_data_length(self):
        """ For unaligned input and output, there is no way to know the 
        target sequence length before hand during inference stage
        
        self.m_data_length will be empty
        """
        
        if not self.m_data_length and not self.m_output_dirs and \
           all([x < 0 for x in self.m_input_reso]):
            # inference stage, when only input is given
            # manually create a fake data length for each utterance
            for file_name in self.m_file_list:
                self.m_data_length[file_name] = 0
        return

        
    
    def f_log_seq_info(self):
        """ After m_data_length has been created, create seq_info
        
        """
        for file_name in self.m_file_list:

            # if file_name is not logged, ignore this file
            if file_name not in self.m_data_length:
                nii_warn.f_eprint("Exclude %s from dataset" % (file_name))
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
            # assume mean/std will be loaded from the network
            # for example, for validation and test sets
            flag = False

        if not any(self.m_input_norm + self.m_output_norm):
            # none of the input / output features needs norm
            flag = False

        if os.path.isfile(ms_input_path) and \
           os.path.isfile(ms_output_path):
            # load mean and std if exists
            ms_input = self.f_load_data(ms_input_path, 1)
            ms_output = self.f_load_data(ms_output_path, 1)
            
            if ms_input.shape[0] != (self.m_input_all_dim * 2) or \
               ms_output.shape[0] != (self.m_output_all_dim * 2):
                if ms_input.shape[0] != (self.m_input_all_dim * 2):
                    nii_warn.f_print("%s incompatible" % (ms_input_path),
                                     'warning')
                if ms_output.shape[0] != (self.m_output_all_dim * 2):
                    nii_warn.f_print("%s incompatible" % (ms_output_path),
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
        if os.path.isfile(data_path) and not self.m_ignore_cached_finfo:
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
            # note: one file is not found in self.m_data_length if it
            #  is shorter than the truncate_seq
            if nii_list_tools.list_identical(self.m_file_list,\
                                             self.m_data_length.keys()):
                nii_warn.f_print("Read sequence info: %s" % (data_path))
                flag = False
            elif nii_list_tools.list_b_in_list_a(self.m_file_list, 
                                                 self.m_data_length.keys()):
                nii_warn.f_print("Read sequence info: %s" % (data_path))
                nii_warn.f_print(
                    "However %d samples are ignoed" % \
                    (len(self.m_file_list)-len(self.m_data_length)))
                tmp = nii_list_tools.members_in_a_not_in_b(
                    self.m_file_list, self.m_data_length.keys())
                for tmp_name in tmp:
                    nii_warn.f_eprint("Exclude %s from dataset" % (tmp_name))
                                    
                flag = False
            else:
                nii_warn.f_print("Incompatible cache: %s" % (data_path))

                tmp = nii_list_tools.members_in_a_not_in_b(
                    self.m_data_length.keys(), self.m_file_list)
                nii_warn.f_print("Possibly invalid data (a few examples):")
                for tmp_name in tmp[:10]:
                    nii_warn.f_print(tmp_name)
                nii_warn.f_print("...\nYou may carefully check these data.")
                nii_warn.f_print("\nThey may not be in the provided data list.")

                nii_warn.f_print("Re-read data statistics")
                self.m_seq_info = []
                self.m_data_length = {}
                self.m_data_total_length = 0
                
            # check wheteher truncating length has been changed
            if self.m_truncate_seq is not None and flag is False:
                tmp_max_len = max([x.seq_length() for x in self.m_seq_info])
                if tmp_max_len != self.m_truncate_seq:
                    mes = "WARNING: truncate_seq conflicts with cached infor. "
                    mes += "Please delete cache files *.dic if you want to"
                    mes += "  use the new truncate_seq"
                    nii_warn.f_print(mes, "warning")

        return flag

    def f_save_data_len(self, data_len_path):
        """
        """
        if not self.m_ignore_cached_finfo:
            nii_io_tk.write_dic([x.print_to_dic() for x in self.m_seq_info], \
                                data_len_path)
        return
        
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

        if self.m_opt_wav_handler > 0:
            # wav handler
            if len(self.m_input_exts) == 1 \
               and self.m_input_exts[0][-3:] == 'wav':
                mes += "\n    Waveform silence handler will be used on input"
            else:
                mes += "\n    Waveform silence handler NOT used on input"
                if len(self.m_input_exts) > 1:
                    mes += "\t    because multiple input features are used"
                
            if len(self.m_output_exts) == 1 \
               and self.m_output_exts[0][-3:] == 'wav':
                mes += "\n    Waveform silence handler will be used on output"
            else:
                mes += "\n    Waveform silence handler NOT used on output"
                if len(self.m_output_exts) > 1:
                    mes += "\t    because multiple output features are used"

        if self.m_inaug_funcs:
            mes += "\n    Use input feature transformation functions"
            if len(self.m_input_exts) > 1:
                mes += "\n     Functions that change data length are ignored"
                mes += "\n     If it is intend to change data length, "
                mes += "\n     please use inoutput_augment_func"
        if self.m_ouaug_funcs:
            mes += "\n    Use output feature transformation functions"
            if len(self.m_output_exts) > 1:
                mes += "\n     Functions that change data length are ignored"
                mes += "\n     If it is intend to change data length, "
                mes += "\n     please use inoutput_augment_func"        
        if self.m_inouaug_func:
            mes += "\n    Use a unified function to alter input and output data"


        if self.m_flag_reverse_load_order:
            mes += "\n    Reverse the data loading order from dataset "
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
        
        # print information
        load_cnt = 0
        total_cnt = len(tmp_dirs) * len(self.m_file_list)

        # print progress
        nii_warn.f_print("Get data statistis (may be slow due to data I/O)")
        bar_len = 50
        loading_marker = total_cnt // bar_len + 1
        nii_warn.f_print("".join(['-' for x in range(bar_len-2)])+">|", 'plain')
        

        # list of invalid data
        invalid_data_lst = []
        
        # loop over each input/output feature type
        for t_dir, t_ext, t_dim, t_reso, t_norm in \
            zip(tmp_dirs, tmp_exts, tmp_dims, tmp_reso, tmp_norm):
            
            s_dim = e_dim
            e_dim = s_dim + t_dim
            t_cnt = 0
            mean_i, var_i = np.zeros([t_dim]), np.zeros([t_dim])
            
            # loop over all the data
            for file_name in self.m_file_list:
                                
                load_cnt += 1
                if load_cnt % loading_marker == 0:
                    nii_warn.f_print('>', end='', flush=True, opt='')

                # get file path
                file_path = nii_str_tk.f_realpath(t_dir, file_name, t_ext)
                if not nii_io_tk.file_exist(file_path):
                    nii_warn.f_die("%s not found" % (file_path))
                    
                # read the length of the data
                if flag_cal_data_len:
                    t_len  = self.f_length_data(file_path) // t_dim
                    
                    if not self.f_log_data_len(file_name, t_len, t_reso):
                        # this data is not valid, ignore it
                        # but it is OK to use it to compute mean/std
                        invalid_data_lst.append(file_name)
                    
                    
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
            # remove invalid data (remove duplicated entries first)
            invalid_data_lst = list(set(invalid_data_lst))
            for tmp_file_name in invalid_data_lst:
                self.m_data_length.pop(tmp_file_name)
            # 
            self.f_precheck_data_length()
            # create seq_info
            self.f_log_seq_info()
            # save len information
            self.f_save_data_len(self.m_data_len_path)
            
        if flag_cal_mean_std:
            self.f_save_mean_std(self.m_ms_input_path,
                                 self.m_ms_output_path)
            
        nii_warn.f_print('')
        # done
        return
        
    def f_putitem(self, output_data, save_dir, filename_prefix, data_infor_str):
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
                os.makedirs(save_dir, exist_ok=True)
            except OSError:
                nii_warn.f_die("Cannot carete {}".format(save_dir))

        # read the sentence information
        tmp_seq_info = nii_seqinfo.SeqInfo()
        tmp_seq_info.parse_from_str(data_infor_str)

        # write the data
        file_name = tmp_seq_info.seq_tag()
        if len(filename_prefix):
            file_name = filename_prefix + file_name 

        seq_length = tmp_seq_info.seq_length()
        s_dim = 0
        e_dim = 0
        for t_ext, t_dim, t_reso in \
            zip(self.m_output_exts, self.m_output_dims, self.m_output_reso):
            e_dim = s_dim + t_dim
            file_path = nii_str_tk.f_realpath(save_dir, file_name, t_ext)
            
            # if this file_name contains part of the path, make sure that the 
            # parent folder has been created
            if self.flag_filename_with_path:
                tmp_save_dir = os.path.dirname(file_path)
                if not os.path.isdir(tmp_save_dir):
                    try:
                        os.makedirs(tmp_save_dir, exist_ok=True)
                    except OSError:
                        nii_warn.f_die("Cannot carete {}".format(tmp_save_dir))
            
            # check the length and write the data
            if seq_length > 0:
                expect_len = seq_length // t_reso
                # confirm that the generated file length is as expected
                if output_data.shape[0] < expect_len:
                    nii_warn.f_print("Warning {:s}".format(file_path), "error")
                    nii_warn.f_print("Generated data is shorter than expected")
                    nii_warn.f_print("Please check the generated file")
                if s_dim == 0 and e_dim == output_data.shape[1]:
                    # if there is only one output feature, directly output it
                    self.f_write_data(output_data[:expect_len], file_path)
                else:
                    # else, output the corresponding dimentions
                    self.f_write_data(output_data[:expect_len, s_dim:e_dim], 
                                      file_path)
            elif seq_length == 0:
                # if seq_length == 0, this is for unaligned input
                if s_dim == 0 and e_dim == output_data.shape[1]:
                    self.f_write_data(output_data, file_path)
                else:
                    self.f_write_data(output_data[s_dim:e_dim], file_path)
            else:
                nii_warn.f_die("Error: seq_length < 0 in generation")
        
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

    def f_adjust_idx(self, data_tuple, idx_shift):
        """
        f_adjust_idx

        This is to be used by customize_dataset for idx adjustment.
        When multiple data sets are merged, the idx from __getitem__
        should be adjusted.

        Only data_io itselts knows how to identify idx from the output of
        __getitem__, we need to define the function here
        """
        if isinstance(data_tuple[-1], list) \
           or isinstance(data_tuple[-1], torch.Tensor):
            # if data_tuple has been collated
            for idx in np.arange(len(data_tuple[-1])):
                data_tuple[-1][idx] += idx_shift
        else:
            # if data_tuple is from __getitem()__
            data_tuple = (data_tuple[0], data_tuple[1],
                          data_tuple[2], data_tuple[-1] + idx_shift)
        return data_tuple


    def f_manage_data(self, idx, opt):
        """
        f_mange_seq(self, idx)
        
        Args:
          idx: list of int, list of data indices
          opt: 'keep', keep only data in idx
               'delete', delete data in idx
        """
        if type(idx) is not list:
            nii_warn.f_die("f_delete_seq(idx) expects idx to be list")

        # get a new list of data for this database
        if opt == 'delete':
            # convert to set of int
            idx_set = set([int(x) for x in idx])
            tmp_idx = [x for x in range(self.__len__()) if x not in idx_set]
        else:
            tmp_idx = [int(x) for x in idx]

        # keep the specified data indices
        self.m_seq_info = [self.m_seq_info[x] for x in tmp_idx \
                           if x < self.__len__() and x >= 0]

        # re-compute the total length of data
        self.m_data_total_length = self.f_sum_data_length()
        return

    def f_get_seq_name_list(self):
        """ return list of data of names in the dataset
        """
        return [x.seq_tag() for x in self.m_seq_info]
    
    def f_get_seq_info(self):
        return [x.print_to_str() for x in self.m_seq_info]

    def f_get_seq_idx_from_name(self, data_names):
        """ return the data index given the data names

        This function is not used so often.
        """
        data_list = self.f_get_seq_name_list()
        try:
            return [data_list.index(x) for x in data_names]
        except ValueError:
            nii_warn.f_print("Not all data names are in this dataset")
            nii_warn.f_print("Return []")
            return []


class NIIDataSetLoader:
    """ NIIDataSetLoader:
    A wrapper over torch.utils.data.DataLoader and DataSet 

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
                 data_format = nii_dconf.h_dtype_str, \
                 params = None, \
                 truncate_seq = None, \
                 min_seq_len = None,
                 save_mean_std = True, \
                 wav_samp_rate = None, \
                 flag_lang = 'EN',
                 global_arg = None,
                 dset_config = None,
                 input_augment_funcs = None,
                 output_augment_funcs = None,
                 inoutput_augment_func = None):
        """
        NIIDataSetLoader(
               data_set_name,
               file_list,
               input_dirs, input_exts, input_dims, input_reso, input_norm,
               output_dirs, output_exts, output_dims, output_reso, output_norm,
               stats_path,
               data_format = '<f4',
               params = None,
               truncate_seq = None,
               min_seq_len = None,
               save_mean_std = True, \
               wav_samp_rate = None, \
               flag_lang = 'EN',
               global_arg = None,
               dset_config = None,
               input_augment_funcs = None,
               output_augment_funcs = None,
               inoutput_augment_func = None):
        Args
        ----
            data_set_name: a string to name this dataset
                           this will be used to name the statistics files
                           such as the mean/std for this dataset
            file_list: a list of file name strings (without extension)
                     or, path to the file that contains the file names
            input_dirs: a list of dirs from which input feature is loaded
            input_exts: a list of input feature name extentions
            input_dims: a list of input feature dimensions
            input_reso: a list of input feature temporal resolution,
                        or None
            input_norm: a list of bool, whether normalize input feature or not

            output_dirs: a list of dirs from which output feature is loaded
            output_exts: a list of output feature name extentions
            output_dims: a list of output feature dimensions
            output_reso: a list of output feature temporal resolution, 
                         or None
            output_norm: a list of bool, whether normalize target feature or not

            stats_path: path to the directory of statistics(mean/std)
            data_format: method to load the data
                    '<f4' (default): load data as float32m little-endian
                    'htk': load data as htk format
            params: parameter for torch.utils.data.DataLoader

            truncate_seq: None or int, 
                          truncate data sequence into smaller truncks
                          truncate_seq > 0 specifies the trunck length
            min_seq_len: None (default) or int, minimum length of an utterance
                         utterance shorter than min_seq_len will be ignored
            save_mean_std: bool, True (default): save mean and std 
            wav_samp_rate: None (default) or int, if input data has  waveform, 
                         please set sampling rate. It is used by _data_writer
            flag_lang: str, 'EN' (default), if input data has text, text will
                       be converted into code indices. flag_lang indicates the 
                       language for the text processer, used by _data_reader
            global_arg: argument parser returned by arg_parse.f_args_parsed()
                      default None
            input_augment_funcs: list of functions for input data augmentation,
                      default None
            output_augment_funcs: list of functions for output data augmentation
                      default None
            inoutput_augment_func: a single data augmentation function
                      default None
        Methods
        -------
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
                                    wav_samp_rate, \
                                    flag_lang, \
                                    global_arg,\
                                    dset_config, \
                                    input_augment_funcs,
                                    output_augment_funcs,
                                    inoutput_augment_func)
        
        # create torch.util.data.DataLoader
        if params is None:
            tmp_params = nii_dconf.default_loader_conf
        else:
            tmp_params = params.copy()
            
        # save parameters
        self.m_params = tmp_params

        # create data loader
        self.m_loader = self.build_loader()

        # done
        return

    def build_loader(self):
        """
        """
        # initialize sampler if necessary
        tmp_params = self.m_params.copy()
        if 'sampler' in tmp_params:
            tmp_sampler = None
            if tmp_params['sampler'] == nii_sampler_fn.g_str_sampler_bsbl:
                if 'batch_size' in tmp_params and tmp_params['batch_size']>1:
                    # initialize the sampler
                    tmp_sampler = nii_sampler_fn.SamplerBlockShuffleByLen(
                        self.m_dataset.f_get_seq_len_list(), 
                        tmp_params['batch_size'])
                    # turn off automatic shuffle
                    tmp_params['shuffle'] = False
                else:
                    nii_warn.f_print("{:s} off as batch-size is 1".format(
                        nii_sampler_fn.g_str_sampler_bsbl))
                    #nii_warn.f_die("Sampler requires batch size > 1")
            tmp_params['sampler'] = tmp_sampler

        # collate function
        if 'batch_size' in tmp_params and tmp_params['batch_size'] > 1:
            # for batch-size > 1, use customize_collate to handle
            # data with different length
            collate_fn = nii_collate_fn.customize_collate
        else:
            collate_fn = None
        
        # return the loader
        return torch.utils.data.DataLoader(
            self.m_dataset, collate_fn=collate_fn, **tmp_params)
        
    def get_loader_params(self):
        return self.m_params
    
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
        print(str(self.m_params))
        return

    def get_seq_name_list(self):
        return self.m_dataset.f_get_seq_name_list()

    def get_seq_info(self):
        return self.m_dataset.f_get_seq_info()
    

    def get_seq_idx_from_name(self, data_names):
        return self.m_dataset.f_get_seq_idx_from_name(data_names)
        

    def putitem(self, output_data, save_dir, filename_prefix, data_infor_str):
        """ Decompose the output_data from network into
        separate files
        """
        self.m_dataset.f_putitem(output_data, save_dir, filename_prefix, 
                                 data_infor_str)

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

    def adjust_utt_idx(self, data_tuple, utt_idx_shift):
        """ Return data tuple with adjusted utterance index in merged dataset
        
        This is used by customize_dataset.
        """
        return self.m_dataset.f_adjust_idx(data_tuple, utt_idx_shift)

    def manage_data(self, data_idx, opt):
        """
        manage_data(self, data_idx)
        
        Args:
          data_idx: list of indices, samples with these indices will be deleted
          opt: 'keep', keep only data in idx
               'delete', delete data in idx
        """
        # delete the data from dataset
        self.m_dataset.f_delete_seq(data_idx, opt)
        # rebuild dataloader
        self.m_loader = self.build_loader()        
        return
    
    def update_seq_len_in_sampler_sub(self, data_info):
        """
        """
        data_idx = seq_info.parse_idx(one_info)
        data_len = seq_info.parse_length(one_info)
        self.m_dataset.f_update_seq_len_for_sampler_list(data_idx, data_len)
        return 

    def update_seq_len_in_sampler(self):
        """update_seq_len()

        Update sequence length if sequence length has been changed
        (for example, during silence trim process)
        
        This is necessary when using shuffle_by_seq_length sampler
        and the sequences were trimmed in data augmentation function.
        """
        # only useful for shuffle_by_seq_length sampler
        if self.m_params['sampler'] == nii_sampler_fn.g_str_sampler_bsbl:
            if hasattr(self.m_loader.sampler, 'update_seq_length'):
                self.m_loader.sampler.update_seq_length(
                    self.m_dataset.f_get_updated_seq_len_for_sampler_list())
            else:
                print("Unknown error in update_seq_len_in_sampler")
                sys.exit(1)
        return
    
if __name__ == "__main__":
    pass
