#!/usr/bin/env python
"""
nn_process_monitor

A simple monitor to show the training / inference process

"""
from __future__ import print_function
import os
import sys
import numpy as np

import core_scripts.other_tools.display as nii_display

def print_gen_info(seq_name, time):
    mes = "Generatin {}, time: {:.3f}s".format(seq_name, time)
    nii_display.f_print_message(mes)
    return mes + '\n'

def print_train_info(epoch, time_tr, loss_tr, \
                     time_val, loss_val, \
                     isbest):
    mes = "Epoch {:>4d} | ".format(epoch)
    mes = mes + "{:>12.1f} | ".format(time_tr + time_val)
    mes = mes + "{:>12.4f} | ".format(loss_tr)
    mes = mes + "{:>12.4f} | ".format(loss_val)    
    if isbest:
        mes = mes + "{:>5s}".format("yes")
    else:
        mes = mes + "{:>5s}".format("no")
    nii_display.f_print_message(mes, flush=True)
    return mes + '\n'

def print_log_head():
    nii_display.f_print_message("{:->65s}".format(""))
    mes = "{:>10s} | ".format("")
    mes = mes + "{:>12s} | ".format("Duration(s)")
    mes = mes + "{:>12s} | ".format("Train loss")
    mes = mes + "{:>12s} | ".format("Dev loss")
    mes = mes + "{:>5s}".format("Best")
    nii_display.f_print_message(mes)
    nii_display.f_print_message("{:->65s}".format(""))    
    return mes + '\n'

def print_log_tail():
    nii_display.f_print_message("{:->65s}".format(""))
    return
    

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
        
    def clear(self):
        self.loss_mat[:, :] = 0
        self.time_mat[:, :] = 0
        self.cur_epoch = 0
        self.seq_names = {}
        
    def print_error_for_batch(self, cnt_idx, seq_idx, epoch_idx):
        try:
            t_1 = self.loss_mat[epoch_idx, seq_idx]
            t_2 = self.time_mat[epoch_idx, seq_idx]
            
            mes = "{}, ".format(self.seq_names[seq_idx])
            mes += "{:d}/{:d}, ".format(cnt_idx+1, \
                                             self.seq_num)
            mes += "Time: {:.6f}s, Loss: {:.6f}".format(t_2, t_1)
            nii_display.f_eprint(mes)
        except IndexError:
            nii_display.f_die("Unknown sample index in Monitor")
        except KeyError:
            nii_display.f_die("Unknown sample index in Monitor")
        return
    
    def get_time(self, epoch):
        return np.sum(self.time_mat[epoch, :])
    
    def get_loss(self, epoch):
        return np.mean(self.loss_mat[epoch, :])
    
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
        if self.cur_epoch == 0:
            return True
        else:
            loss_this = np.sum(self.loss_mat[self.cur_epoch, :])
            loss_prev = np.sum(self.loss_mat[self.cur_epoch-1, :])
            if loss_this < loss_prev:
                return True
            else:
                return False
            
    def should_early_stop(self, no_best_epoch_num):
        """ 
        check whether to stop training early
        """
        starting_epoch  = self.cur_epoch - no_best_epoch_num
        if starting_epoch < 0:
            starting_epoch = 0
            
        tm = np.mean(self.loss_mat[starting_epoch: \
                                   self.cur_epoch+1, :], \
                     axis=1)
            
        tm = np.sum(np.diff(tm) > 0)
        if tm < no_best_epoch_num:
            return False
        else:
            return True
            
if __name__ == "__main__":
    print("Definition of nn_process_monitor")
