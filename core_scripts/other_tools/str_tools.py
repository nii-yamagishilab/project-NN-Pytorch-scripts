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
    """ output = string_chop(InStr)
    Chop the ending '\r' and '\n' from input string
    
    Args:
        InStr: str, the input string

    Return:
        output: str
    
    '\r' corresponds to '0x0d' or 13,
    '\n' corresponds to '0x0a' or 10                               
    """
    if len(InStr) >= 2 and ord(InStr[-1]) == 10 and ord(InStr[-2]) == 13:
        return InStr[:-2]
    elif len(InStr) >= 1 and ord(InStr[-1]) == 10:
        return InStr[:-1]
    else:
        return InStr

if __name__ == "__main__":
    print("string tools")
