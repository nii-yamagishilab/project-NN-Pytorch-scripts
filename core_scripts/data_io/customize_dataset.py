#!/usr/bin/env python
"""
customized dataset

NII_MergeDataSetLoader (to one minibatch): 
 We want to load dataset 1, 2, and 3, 
 We also want to draw sample from each dataset for one minibatch.
 One epoch over the merged datasets will be decided by the smallest dataset

"""

from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
import torch.utils.data

import core_scripts.other_tools.display as nii_warn
import core_scripts.data_io.default_data_io as nii_default_dset
import core_scripts.data_io.customize_collate_fn as nii_collate_fn
import core_scripts.data_io.conf as nii_dconf

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


###############################################
# Dataset definition to merge multiple datasets
###############################################

class merge_loader():
    """ customized data loader over multiple datasets
    """
    def __init__(self, datasets):
        # list of datasets
        self.m_datasets = datasets
        # initialized iterators 
        self.m_loaders = [x.get_loader() for x in self.m_datasets]
        # utterance index shift
        self.m_idx_shift = np.cumsum([0] + 
                                     [x.get_seq_num() for x in self.m_datasets])
        return

    def adjust_utt_idx(self, data_tuple, dataset_idx):
        """ when merging dataset 1, 2, 3 ...
        index for dataset 2: index += dataset_1.get_seq_num()
        index for dataset 3: index += dataset_1 + dataset_2.get_seq_num()
        
        We have to call dataset.f_adjust_idx because it is the dataset itself
        that knows how to pause the data_tuple
        """
        return self.m_datasets[dataset_idx].get_dataset().f_adjust_idx(
            data_tuple, self.m_idx_shift[dataset_idx])

    def __iter__(self):
        """
        create the list of iterators
        """
        self.m_loader_iter = [iter(x) for x in self.m_loaders]
        return self

    def __next__(self):
        """ try to load data from m_datasets, and merge them into a 
        single minibatch
        """
        try:
            data_list = []
            for dataset_idx, dataloader in enumerate(self.m_loader_iter):
                data_list.append(
                    self.adjust_utt_idx(next(dataloader), dataset_idx))
            # data shape should be the same
            return nii_collate_fn.customize_collate_from_batch(data_list)
        except StopIteration:
            raise StopIteration

class ConcatDataset(torch.utils.data.Dataset):
    """ Adopted from 
    https://discuss.pytorch.org/t/train-simultaneously-on-two-datasets/649/2

    But here we concatenate data not in minibatch.
    """
    def __init__(self, datasets):
        """ datasets must be torch.utils.data.Dataset
        """
        # all the sub sets
        self.datasets = datasets
        self.num_subset = len(datasets)
        # len of each sub set
        self.len_buffer = [x.__len__() for x in self.datasets]
        # for later use, to decide from which subset we draw the sample
        self.len_top = np.cumsum(self.len_buffer)
        self.len_bot = np.cumsum([0] + self.len_buffer[:-1])
        # done
        return

    def __getitem__(self, i):
        # for example, data1 = [a], data2 = [b, c]
        # self.len_buffer = [1, 2]
        # self.len_top = [1, 3] 
        # self.len_bot = [0, 1]
        #  __getitem__(0) -> data1[0-0] = a
        #  __getitem__(1) -> data2[1-1] = b
        #  __getitem__(2) -> data2[2-1] = c
        for idx_u, idx_d, subset in \
            zip(self.len_top, self.len_bot, self.datasets):
            if i < idx_u:
                return subset.__getitem__(i - idx_d)
            else:
                # keep going to the next subset
                pass
        nii_warn.f_die("Merge dataset: fatal error in __getitem__")
        return None

    def __len__(self):
        return sum(self.len_buffer)

class NII_MergeDataSetLoader():
    """ 
    """
    def __init__(self,
                 dataset_name, \
                 list_file_list, \
                 list_input_dirs, input_exts, input_dims, input_reso, \
                 input_norm, \
                 list_output_dirs, output_exts, output_dims, output_reso, \
                 output_norm, \
                 stats_path, \
                 data_format = '<f4', \
                 params = None, \
                 truncate_seq = None, \
                 min_seq_len = None,
                 save_mean_std = True, \
                 wav_samp_rate = None,
                 way_to_merge = 'concatenate'):
        
        # check whether input_dirs and output_dirs are lists
        if type(list_input_dirs[0]) is list and \
           type(list_output_dirs[0]) is list and \
           type(list_file_list) is list and \
           len(list_input_dirs) == len(list_output_dirs) and \
           len(list_input_dirs) == len(list_file_list):
            pass
        else:
            mes = "NII_MergeDataSetLoader: input_dirs, output_dirs, "
            mes += "and file_list should be list of lists. "
            mes += "They should have equal length. But we have:"
            mes += "{:s}\n{:s}\n{:s}".format(
                str(list_input_dirs), str(list_output_dirs), 
                str(list_file_list))
            nii_warn.f_die(mes)
        
        if type(dataset_name) is list:
            if len(dataset_name) != len(list_input_dirs):
                mes = "dataset_name should have {:d} elements. ".format(
                    len(list_file_list))
                mes += "But we have: {:s}".format(str(dataset_name))
                nii_warn.f_die(mes)
            elif len(list(set(dataset_name))) != len(list_input_dirs):
                mes = "dataset_name has duplicated elements: {:s}".format(
                    str(dataset_name))
                nii_warn.f_die(mes)
            else:
                tmp_dnames = dataset_name
        else:
            tmp_dnames = [dataset_name + '_sub_{:d}'.format(idx) \
                          for idx in np.arange(len(list_input_dirs))]
            
                

        # create individual datasets
        lst_dset = []
        for sub_input_dirs, sub_output_dirs, sub_file_list, tmp_name in \
            zip(list_input_dirs, list_output_dirs, list_file_list, tmp_dnames):
            
            lst_dset.append(
                nii_default_dset.NIIDataSetLoader(
                    tmp_name,
                    sub_file_list,
                    sub_input_dirs, input_exts, input_dims, input_reso, \
                    input_norm, \
                    sub_output_dirs, output_exts, output_dims, output_reso, \
                    output_norm, \
                    stats_path, data_format, params, truncate_seq, min_seq_len,
                    save_mean_std, wav_samp_rate))
        
        # list of the datasets
        self.m_datasets = lst_dset
        
        self.way_to_merge = way_to_merge
        # create data loader
        if way_to_merge == 'concatenate':
            # to create DataLoader, we need the pytorch.dataset
            py_datasets = ConcatDataset([x.get_dataset() for x in lst_dset])
            
            # create new loader after merge the datasets
            #  copied from default IO
            if params is None:
                tmp_params = nii_dconf.default_loader_conf
            else:
                tmp_params = params
            # collate function
            if 'batch_size' in params and params['batch_size'] > 1:
                collate_fn = nii_collate_fn.customize_collate
            else:
                collate_fn = None
            
            self.m_loader = torch.utils.data.DataLoader(
                py_datasets, collate_fn=collate_fn, **tmp_params)

        else:
            self.m_loader = merge_loader(lst_dset)
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
        return self.m_datasets

    def get_data_mean_std(self):
        """
        """
        # temporary solution: just use the first one
        return self.m_datasets[0].get_data_mean_std()

    def print_info(self):
        """
        """
        nii_warn.f_print_message("Merge datasets by: " + self.way_to_merge)
        for dset in self.m_datasets:
            dset.print_info()
        return

    def putitem(self, output_data, save_dir, data_infor_str):
        """ Decompose the output_data from network into
        separate files
        """
        # Since all datasets have similar configuration on feat dim,
        # use anyone is OK
        self.m_datasets[0].putitem(output_data, save_dir, data_infor_str)

    def get_in_dim(self):
        """ Return the dimension of input features
        """ 
        # Since all datasets have similar configuration on feat dim,
        # use anyone is OK
        return self.m_datasets[0].get_in_dim()

    def get_out_dim(self):
        """ Return the dimension of output features
        """
        # Since all datasets have similar configuration on feat dim,
        # use anyone is OK
        return self.m_datasets[0].get_out_dim()

    def get_seq_num(self):
        """ Return the number of sequences (after truncation)
        """ 
        return sum([x.get_seq_num() for x in self.m_datasets])



if __name__ == "__main__":
    print("Definition of customized Pytorch dataset")
