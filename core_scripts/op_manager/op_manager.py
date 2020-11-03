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

        # create optimizer
        if self.op_flag == "Adam":
            self.optimizer = torch_optim.Adam(model.parameters(), \
                                              lr = self.lr)
        else:
            nii_warn.f_print("%s not availabel" % (self.op_flag),
                             "error")
            nii_warn.f_die("Please change optimizer")

        # number of epochs
        self.epochs = args.epochs
        self.no_best_epochs = args.no_best_epochs
        
        # lr scheduler
        self.lr_decay = args.lr_decay_factor
        if self.lr_decay > 0:
            
            if self.no_best_epochs < 0:
                self.no_best_epochs = 5
                nii_warn.f_print("--no-best-epochs is set to 5 ")
                nii_warn.f_print("for learning rate decaying")

            self.lr_scheduler = torch_optim_steplr.ReduceLROnPlateau(
                optimizer=self.optimizer, factor=self.lr_decay, 
                patience=self.no_best_epochs)
        else:
            self.lr_scheduler = None

        return

    def print_info(self):
        """ print message of optimizer
        """
        mes = "Optimizer:\n  Type: {} ".format(self.op_flag)
        mes += "\n  Learing rate: {:2.6f}".format(self.lr)
        mes += "\n  Epochs: {:d}".format(self.epochs)
        mes += "\n  No-best-epochs: {:d}".format(self.no_best_epochs)
        if self.lr_scheduler:
            mes += "\n  LR decay with factor={:2.3f}".format(self.lr_decay)
        nii_warn.f_print_message(mes)

    def get_epoch_num(self):
        return self.epochs
    
    def get_no_best_epoch_num(self):
        return self.no_best_epochs

    def get_lr_info(self):
        
        if self.lr_scheduler:
            # no way to look into the updated lr rather than using _last_lr
            tmp = ''
            for updated_lr in self.lr_scheduler._last_lr:
                if np.abs(self.lr - updated_lr) > self.lr_scheduler.eps:
                    tmp += "{:.2e} ".format(updated_lr)
            if tmp:
                tmp = " LR -> " + tmp
            return tmp
        else:
            return None
    
if __name__ == "__main__":
    print("Optimizer Wrapper")
