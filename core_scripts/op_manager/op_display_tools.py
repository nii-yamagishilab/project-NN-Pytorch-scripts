#!/usr/bin/env python
"""
op_display_tools

Functions to print/display the training/inference information

"""
from __future__ import print_function
import os
import sys
import numpy as np

import core_scripts.other_tools.display as nii_display

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def print_gen_info(seq_name, time):
    """ Print the information during inference
    """
    mes = "Generating {}, time: {:.3f}s".format(seq_name, time)
    nii_display.f_print_message(mes)
    return mes + '\n'

def _print_loss(loss_array):
    mes = ""
    if loss_array.shape[0] == 1:
        mes += "%12.4f " % (loss_array[0])
    else:
        mes = []
        for data in loss_array:
            mes.append('%6.2f' % (data))
        mes = ' '.join(mes)
    mes += "| "
    return mes

def print_train_info(epoch, time_tr, loss_tr, time_val, 
                     loss_val, isbest, lr_info):
    """ Print the information during training
    """
    mes = "{:>7d} | ".format(epoch)
    mes = mes + "{:>12.1f} | ".format(time_tr + time_val)
    mes += _print_loss(loss_tr)
    mes += _print_loss(loss_val)
    #mes = mes + "{:>12.4f} | ".format(loss_tr)
    #mes = mes + "{:>12.4f} | ".format(loss_val)    
    if isbest:
        mes = mes + "{:>5s}".format("yes")
    else:
        mes = mes + "{:>5s}".format("no")
    if lr_info:
        mes = mes + lr_info
    nii_display.f_print_message(mes, flush=True)
    return mes + '\n'

def print_log_head():
    """ Print the head information
    """ 
    nii_display.f_print_message("{:->62s}".format(""))
    mes = "{:>7s} | ".format("Epoch")
    mes = mes + "{:>12s} | ".format("Duration(s)")
    mes = mes + "{:>12s} | ".format("Train loss")
    mes = mes + "{:>12s} | ".format("Dev loss")
    mes = mes + "{:>5s}".format("Best")
    nii_display.f_print_message(mes)
    nii_display.f_print_message("{:->62s}".format(""), flush=True)
    return mes + '\n'

def print_log_tail():
    """ Print the tail line
    """
    nii_display.f_print_message("{:->62s}".format(""))
    return


if __name__ == "__main__":
    print("Op_display_tools")
