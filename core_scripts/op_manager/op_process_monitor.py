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
        self.loss_mat = np.zeros([epoch_num, seq_num, 1])
        self.time_mat = np.zeros([epoch_num, seq_num])
        self.seq_names = {}
        self.epoch_num = epoch_num
        self.seq_num = seq_num
        self.cur_epoch = 0
        self.best_error = None
        self.best_epoch = None
        self.loss_flag = None

    def clear(self):
        self.loss_mat.fill(0)
        self.time_mat.fill(0)
        self.cur_epoch = 0
        self.seq_names = {}
        self.best_error = None
        self.best_epoch = None
        self.loss_flag = None
        
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
        state_dic['loss_flag'] = self.loss_flag
        # no need to save self.seq_names
        return state_dic

    def load_state_dic(self, state_dic):
        """ resume training, load the information
        """
        try:
            if self.seq_num != state_dic['seq_num']:
                nii_display.f_print("Number of samples are different \
                from previous training", 'error')
                nii_display.f_print("Please make sure that you are \
                using the same training/development sets as before.", "error")
                nii_display.f_print("Or\nPlease add --")
                nii_display.f_print("ignore_training_history_in_trained_model")
                nii_display.f_die(" to avoid loading training history")

            if self.epoch_num == state_dic['epoch_num']:
                self.loss_mat = state_dic['loss_mat']
                self.time_mat = state_dic['time_mat']
            else:
                # if training epoch is increased, resize the shape
                tmp_loss_mat = state_dic['loss_mat']
                self.loss_mat = np.resize(
                    self.loss_mat, 
                    [self.epoch_num, self.seq_num, tmp_loss_mat.shape[2]])
                self.loss_mat[0:tmp_loss_mat.shape[0]] = tmp_loss_mat
                self.time_mat[0:tmp_loss_mat.shape[0]] = state_dic['time_mat']

            self.seq_num = state_dic['seq_num']
            # since the saved cur_epoch has been finished
            self.cur_epoch = state_dic['cur_epoch'] + 1
            self.best_error = state_dic['best_error']
            self.best_epoch = state_dic['best_epoch']
            self.loss_flag = state_dic['loss_flag']
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
            mes += "Time: {:.6f}s".format(t_2)
            for loss_indi in t_1:
                mes += ", Loss: {:.6f}".format(loss_indi)
            nii_display.f_eprint(mes, flush=True)
        except IndexError:
            nii_display.f_die("Unknown sample index in Monitor")
        except KeyError:
            nii_display.f_die("Unknown sample index in Monitor")
        return
    
    def get_time(self, epoch):
        return np.sum(self.time_mat[epoch, :])
    
    def get_loss(self, epoch):
        # return a array
        return np.mean(self.loss_mat[epoch, :], axis=0)

    def get_epoch(self):
        return self.cur_epoch

    def get_max_epoch(self):
        return self.epoch_num

    def _get_loss_for_learning_stopping(self, epoch_idx):
        # compute the average loss values
        if epoch_idx > self.cur_epoch:
            nii_display.f_print("To find loss for future epochs", 'error')
            nii_display.f_die("Op_process_monitor: error")
        if epoch_idx < 0:
            nii_display.f_print("To find loss for NULL epoch", 'error')
            nii_display.f_die("Op_process_monitor: error")
        loss_this = np.sum(self.loss_mat[epoch_idx, :, :], axis=0)
        # compute only part of the loss for early stopping when necessary
        loss_this = np.sum(loss_this * self.loss_flag)
        return loss_this

    def print_error_for_epoch(self, epoch):
        loss = np.mean(self.loss_mat[epoch, :])
        time_sum = np.sum(self.time_mat[epoch, :])
        mes = "Epoch {:d}: ".format(epoch)
        mes += 'Time: {:.6f}, Loss: {:.6f}'.format(time_sum, loss)
        nii_display.f_print_message(mes)
        return "{}\n".format(mes)

    def log_loss(self, loss, loss_flag, time_cost, seq_info, seq_idx, \
                 epoch_idx):
        """ Log down the loss
        """
        self.seq_names[seq_idx] = seq_info
        if self.loss_mat.shape[-1] != len(loss):
            self.loss_mat = np.resize(self.loss_mat, 
                                      [self.loss_mat.shape[0], 
                                       self.loss_mat.shape[1], 
                                       len(loss)])
        self.loss_flag = loss_flag
        self.loss_mat[epoch_idx, seq_idx, :] = loss
        self.time_mat[epoch_idx, seq_idx] = time_cost
        self.cur_epoch = epoch_idx
        return

    def is_new_best(self):
        """
        check whether epoch is the new_best
        """
        loss_this = self._get_loss_for_learning_stopping(self.cur_epoch)
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
            #
            #tmp = []
            #for idx in np.arange(no_best_epoch_num+1):
            #    tmp.append(self._get_loss_for_learning_stopping(
            #        self.cur_epoch - idx))
            #if np.sum(np.diff(tmp) < 0) >= no_best_epoch_num:
            #    return True
            #else:
            #    return False
            return True
        else:
            return False
            
if __name__ == "__main__":
    print("Definition of nn_process_monitor")
