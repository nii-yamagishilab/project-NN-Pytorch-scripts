#!/usr/bin/env python
"""
op_process_monitor

A simple monitor on training / inference process

"""

from __future__ import print_function
import os
import sys
import numpy as np

import core_scripts.other_tools.display as nii_display

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


class Monitor():
    """  A monitor to log down all the training / 
         inference informations
    """
    def __init__(self, epoch_num, seq_num):
        self.loss_mat = np.zeros([epoch_num, seq_num])
        self.time_mat = np.zeros([epoch_num, seq_num])
        self.seq_names = {}
        self.epoch_num = epoch_num
        self.seq_num = seq_num
        self.cur_epoch = 0
        self.best_error = None
        self.best_epoch = None
        
    def clear(self):
        self.loss_mat[:, :] = 0
        self.time_mat[:, :] = 0
        self.cur_epoch = 0
        self.seq_names = {}
        self.best_error = None
        self.best_epoch = None
        
    def get_state_dic(self):
        """ create a dictionary to save process
        """
        state_dic = {}
        state_dic['loss_mat'] = self.loss_mat
        state_dic['time_mat'] = self.time_mat
        state_dic['epoch_num'] = self.epoch_num
        state_dic['seq_num'] = self.seq_num
        state_dic['cur_epoch'] = self.cur_epoch
        state_dic['best_error'] = self.best_error
        state_dic['best_epoch'] = self.best_epoch
        # no need to save self.seq_names
        return state_dic

    def load_state_dic(self, state_dic):
        """ resume training, load the information
        """
        try:
            if self.seq_num != state_dic['seq_num']:
                nii_display.f_print("Number of samples are different \
                from previous training", 'error')
                nii_display.f_die("Please make sure resumed training are \
                using the same training/development sets as before")

            self.loss_mat = state_dic['loss_mat']
            self.time_mat = state_dic['time_mat']
            self.epoch_num = state_dic['epoch_num']
            self.seq_num = state_dic['seq_num']
            # since the saved cur_epoch has been finished
            self.cur_epoch = state_dic['cur_epoch'] + 1
            self.best_error = state_dic['best_error']
            self.best_epoch = state_dic['best_epoch']
            self.seq_names = {}
        except KeyError:
            nii_display.f_die("Invalid op_process_monitor state_dic")

    def print_error_for_batch(self, cnt_idx, seq_idx, epoch_idx):
        try:
            t_1 = self.loss_mat[epoch_idx, seq_idx]
            t_2 = self.time_mat[epoch_idx, seq_idx]
            
            mes = "{}, ".format(self.seq_names[seq_idx])
            mes += "{:d}/{:d}, ".format(cnt_idx+1, \
                                             self.seq_num)
            mes += "Time: {:.6f}s, Loss: {:.6f}".format(t_2, t_1)
            nii_display.f_eprint(mes, flush=True)
        except IndexError:
            nii_display.f_die("Unknown sample index in Monitor")
        except KeyError:
            nii_display.f_die("Unknown sample index in Monitor")
        return
    
    def get_time(self, epoch):
        return np.sum(self.time_mat[epoch, :])
    
    def get_loss(self, epoch):
        return np.mean(self.loss_mat[epoch, :])

    def get_epoch(self):
        return self.cur_epoch

    def get_max_epoch(self):
        return self.epoch_num

    def print_error_for_epoch(self, epoch):
        loss = np.mean(self.loss_mat[epoch, :])
        time_sum = np.sum(self.time_mat[epoch, :])
        mes = "Epoch {:d}: ".format(epoch)
        mes += 'Time: {:.6f}, Loss: {:.6f}'.format(time_sum, loss)
        nii_display.f_print_message(mes)
        return "{}\n".format(mes)

    def log_loss(self, loss, time_cost, seq_info, seq_idx, \
                 epoch_idx):
        """ Log down the loss
        """
        self.seq_names[seq_idx] = seq_info
        self.loss_mat[epoch_idx, seq_idx] = loss
        self.time_mat[epoch_idx, seq_idx] = time_cost
        self.cur_epoch = epoch_idx
        return

    def is_new_best(self):
        """
        check whether epoch is the new_best
        """
        loss_this = np.sum(self.loss_mat[self.cur_epoch, :])
        if self.best_error is None or loss_this < self.best_error:
            self.best_error = loss_this
            self.best_epoch = self.cur_epoch
            return True
        else:
            return False
            
    def should_early_stop(self, no_best_epoch_num):
        """ 
        check whether to stop training early
        """
        if (self.cur_epoch - self.best_epoch) >= no_best_epoch_num:
            return True
        else:
            return False
            
if __name__ == "__main__":
    print("Definition of nn_process_monitor")
