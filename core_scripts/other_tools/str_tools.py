#!/usr/bin/env python
"""
str_tools

tools to process string
"""
from __future__ import absolute_import

import os
import sys

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def f_realpath(f_dir, f_name, f_ext):
    """ file_path = f_realpath(f_dir, f_name, f_ext)
    Args:
      f_dir: string, directory
      f_name: string, file name
      f_ext: string, file name extension

    Return:
      file_path: realpath     
    """
    file_path = os.path.join(f_dir, f_name)
    if f_ext.startswith(os.extsep):
        file_path = file_path + f_ext
    else:
        file_path = file_path + os.extsep + f_ext
    return file_path
                
def string_chop(InStr):
    """string_chop(InStr):                                             
        Chop the input string                                       
        InStr: the input string                                     
        FChopMore: True: both '0x0d' and '0x0a' at the              
                         end will be chopped                        
                   False: only chop 0x0a                            
                   default: True                                    
    """
    if ord(InStr[-1]) == 10 and ord(InStr[-2]) == 13:
        return InStr[:-2]
    elif ord(InStr[-1]) == 10:
        return InStr[:-1]
    else:
        return InStr
                    
