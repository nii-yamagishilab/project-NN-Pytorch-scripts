#!/usr/bin/env python
"""
op_manager

A simple wrapper to create lr scheduler

"""
from __future__ import absolute_import

import os
import sys
import numpy as np
import torch
import torch.optim as torch_optim
import torch.optim.lr_scheduler as torch_optim_steplr

import core_scripts.other_tools.display as nii_warn

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"



class LRScheduler():
    """ Wrapper over different types of learning rate Scheduler
    
    """
    def __init__(self, optimizer, args):        
        
        # learning rate decay
        self.lr_decay = args.lr_decay_factor

        # lr scheduler type 
        # please check arg_parse.py for the number ID
        self.lr_scheduler_type = args.lr_scheduler_type
        
        # patentience for ReduceLROnPlateau
        self.lr_patience = 5

        # step size for stepLR
        self.lr_stepLR_size = 10

        if self.lr_decay > 0:
            if self.lr_scheduler_type == 1:
                # StepLR
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer=optimizer, step_size=self.lr_stepLR_size, 
                    gamma=self.lr_decay)
            else:
                # by default, ReduceLROnPlateau
                if args.no_best_epochs < 0:
                    self.lr_patience = 5
                    nii_warn.f_print("--no-best-epochs is set to 5 ")
                    nii_warn.f_print("for learning rate decaying")
                        
                self.lr_scheduler = torch_optim_steplr.ReduceLROnPlateau(
                    optimizer=optimizer, factor=self.lr_decay, 
                    patience=self.lr_patience)

            self.flag = True
        else:
            self.lr_scheduler = None
            self.flag =False
        return

    def f_valid(self):
        """ Whether this LR scheduler is valid
        """
        return self.flag
    
    def f_print_info(self):
        """ Print information about the LR scheduler
        """
        if not self.flag:
            mes = ""
        else:
            if self.lr_scheduler_type == 1:
                mes = "\n  LR scheduler, StepLR [gamma %f, step %d]" % (
                    self.lr_decay, self.lr_stepLR_size)
            else:
                mes = "\n  LR scheduler, ReduceLROnPlateau "
                mes += "[decay %f, patience %d]" % (
                    self.lr_decay, self.lr_patience)
        return mes
    
    def f_last_lr(self):
        """ Return the last lr
        """
        if self.f_valid():
            if hasattr(self.lr_scheduler, "get_last_lr"):
                return self.lr_scheduler.get_last_lr()
            else:
                return self.lr_scheduler._last_lr
        else:
            return []

    def f_load_state_dict(self, state):
        if self.f_valid():
            self.lr_scheduler.load_state_dict(state)
        return

    def f_state_dict(self):
        if self.f_valid():
            return self.lr_scheduler.state_dict()
        else:
            return None
    
    def f_step(self, loss_val):
        if self.f_valid():
            if self.lr_scheduler_type == 1:
                self.lr_scheduler.step()
            else:
                self.lr_scheduler.step(loss_val)
        return

    def f_allow_early_stopping(self):
        if self.f_valid():
            if self.lr_scheduler_type == 1:
                return True
            else:
                # ReduceLROnPlateau no need to use early stopping
                return False
        else:
            return True


if __name__ == "__main__":
    print("Definition of lr_scheduler")
