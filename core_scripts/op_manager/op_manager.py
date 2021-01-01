#!/usr/bin/env python
"""
op_manager

A simple wrapper to create optimizer

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
import torch.optim as torch_optim
import torch.optim.lr_scheduler as torch_optim_steplr


import core_scripts.other_tools.list_tools as nii_list_tools
import core_scripts.other_tools.display as nii_warn
import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.op_manager.conf as nii_op_config
import core_scripts.op_manager.op_process_monitor as nii_op_monitor
import core_scripts.op_manager.lr_scheduler as nii_lr_scheduler

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


class OptimizerWrapper():
    """ Wrapper over optimizer
    """
    def __init__(self, model, args):
        """ Initialize an optimizer over model.parameters()
        """
        # check valildity of model
        if not hasattr(model, "parameters"):
            nii_warn.f_print("model is not torch.nn", "error")
            nii_warn.f_die("Error in creating OptimizerWrapper")

        # set optimizer type
        self.op_flag = args.optimizer
        self.lr = args.lr
        self.l2_penalty = args.l2_penalty

        # grad clip norm is directly added in nn_manager
        self.grad_clip_norm = args.grad_clip_norm

        # create optimizer
        if self.op_flag == "Adam":
            if self.l2_penalty > 0:
                self.optimizer = torch_optim.Adam(model.parameters(), 
                                                  lr=self.lr, 
                                                  weight_decay=self.l2_penalty)
            else:
                self.optimizer = torch_optim.Adam(model.parameters(),
                                                  lr=self.lr)

        else:
            nii_warn.f_print("%s not availabel" % (self.op_flag),
                             "error")
            nii_warn.f_die("Please change optimizer")

        # number of epochs
        self.epochs = args.epochs
        self.no_best_epochs = args.no_best_epochs
        
        # lr scheduler
        self.lr_scheduler = nii_lr_scheduler.LRScheduler(self.optimizer, args)
        return

    def print_info(self):
        """ print message of optimizer
        """
        mes = "Optimizer:\n  Type: {} ".format(self.op_flag)
        mes += "\n  Learing rate: {:2.6f}".format(self.lr)
        mes += "\n  Epochs: {:d}".format(self.epochs)
        mes += "\n  No-best-epochs: {:d}".format(self.no_best_epochs)
        if self.lr_scheduler.f_valid():
            mes += self.lr_scheduler.f_print_info()
        if self.l2_penalty > 0:
            mes += "\n  With weight penalty {:f}".format(self.l2_penalty)
        if self.grad_clip_norm > 0:
            mes += "\n  With grad clip norm {:f}".format(self.grad_clip_norm)
        nii_warn.f_print_message(mes)

    def get_epoch_num(self):
        return self.epochs
    
    def get_no_best_epoch_num(self):
        return self.no_best_epochs

    def get_lr_info(self):
        
        if self.lr_scheduler.f_valid():
            # no way to look into the updated lr rather than using _last_lr
            tmp = ''
            for updated_lr in self.lr_scheduler.f_last_lr():
                if np.abs(self.lr - updated_lr) > 0.0000001:
                    tmp += "{:.2e} ".format(updated_lr)
            if tmp:
                tmp = " LR -> " + tmp
            return tmp
        else:
            return None
    
if __name__ == "__main__":
    print("Optimizer Wrapper")
