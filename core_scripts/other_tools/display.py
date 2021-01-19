#!/usr/bin/env python
"""
dispaly.py

Tools to display the commands or warnings

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import datetime

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


class DisplayColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[91m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def f_print(message, opt='ok', end='\n', flush=False):
    """ f_print(message, opt)
    Print message with specific style
    
    Args:
      message: str
      opt: str, "warning", "highlight", "ok", "error"
    """
    if opt == 'warning':
        print(DisplayColors.WARNING + str(message) + DisplayColors.ENDC, 
              flush = flush, end = end)
    elif opt == 'highlight':
        print(DisplayColors.OKGREEN + str(message) + DisplayColors.ENDC, 
              flush = flush, end = end)
    elif opt == 'ok':
        print(DisplayColors.OKBLUE + str(message) + DisplayColors.ENDC, 
              flush = flush, end = end)
    elif opt == 'error':
        print(DisplayColors.FAIL + str(message) + DisplayColors.ENDC, 
              flush = flush, end = end)
    else:
        print(message, flush=flush, end=end)
    return

def f_print_w_date(message, level='h'):
    """ f_print_w_date(message, level)
    
    Print message with date shown
    
    Args: 
      message: a string
      level: which can be 'h' (high-level), 'm' (middle-level), 'l' (low-level)
    """
    if level == 'h':
        message = '---  ' + str(message) + ' ' \
                  + str(datetime.datetime.now())  + ' ---'
        tmp = ''.join(['-' for x in range(len(message))])
        f_print(tmp)
        f_print(message)
        f_print(tmp)
    elif level == 'm':
        f_print('---' + str(message) + ' ' \
                + str(datetime.datetime.now().time()) + '---')
    else:
        f_print(str(message) + ' ' + str(datetime.datetime.now().time()))
    sys.stdout.flush()
    return

def f_die(message):
    """ f_die(message)
    Print message in "error" mode and exit program with sys.exit(1)
    """
    f_print("Error: " + message, 'error')
    sys.exit(1)


def f_eprint(*args, **kwargs):
    """ f_eprint(*args, **kwargs)
    Print
    """
    print(*args, file=sys.stderr, **kwargs)

def f_print_message(message, flush=False, end='\n'):
    f_print(message, 'normal', flush=flush, end=end)
    
if __name__ == "__main__":
    pass
