#!/usr/bin/env python
"""
list_tools.py

Tools to process list(s)
"""
from __future__ import absolute_import

import os
import sys
import collections
import core_scripts.other_tools.display as nii_warn
import core_scripts.other_tools.str_tools as nii_str_tool

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def listdir_with_ext(file_dir, file_ext=None):
    """ 
    file_list = lstdir_with_ext(file_dir, file_ext=None)
    Return a list of file names with specified extention

    Args:
        file_dir: a file directory
        file_ext: string, specify the extention, e.g., txt, bin
    Return:
        file_list: a list of file_names
    """
    try:
    
        if file_ext is None:
            file_list = [os.path.splitext(x)[0] for x in os.listdir(file_dir) \
                        if not x.startswith('.')]
        else:
            file_list = [os.path.splitext(x)[0] for x in os.listdir(file_dir) \
                         if not x.startswith('.') and x.endswith(file_ext)]
        return file_list
    except OSError:
        nii_warn.f_print("Cannot access %s" % (file_dir), "error")
        return []

def common_members(list_a, list_b):
    """ list_c = common_members(list_a, list_b)
    Return a list (sorted) of common members in list_a, list_b
    
    Parameters:
        list_a: list
        list_b: list
    Returns:
        list_c: a list of common members in list_a and list_b
    
    """
    list_c = list(set(list_a).intersection(list_b))
    list_c.sort()
    return list_c


def list_identical(list_a, list_b):
    """ flag = list_identical(list_a, list_b)
    Return true/false, check whether list_a is identical to list_b
    stackoverflow.com/a/19244156/403423
    """
    return collections.Counter(list_a) == collections.Counter(list_b)

def list_b_in_list_a(list_a, list_b):
    """ list_b_in_list_a(list_a, list_b)
    Whether list_b is subset of list_a

    Parameters:
        list_a: list
        list_b: list
    Return: False or True
    """
    return set(list_b) <= set(list_a)

def members_in_a_not_in_b(list_a, list_b):
    """ members_in_a_not_b(list_a, list_b):
    Return a list of members that are in list_a but not in list_b
    
    Parameters:
        list_a: list
        list_b: list
    Return: list
    """
    return list(set(list_a) - set(list_b))

def read_list_from_text(filename, f_chop=True):
    """                                       
    """
    f = open(filename,'r')
    data = []
    for line in f:
        if f_chop:
            line = nii_str_tool.string_chop(line)
        data.append(line)
    f.close()
    return data

if __name__ == "__main__":
    print("Definition of tools for list operation")
