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

from pathlib import Path as pathtool

def listdir_with_ext_flat(file_dir, file_ext=None):
    """
    file_list = lstdir_with_ext_flat(file_dir, file_ext=None)
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
    
    

def listdir_with_ext_recur(file_dir, file_ext=None, recursive=True):
    """
    file_list = lstdir_with_ext(file_dir, file_ext=None)
    Return a list of file names with specified extention

    Args:
        file_dir: a file directory
        file_ext: string, specify the extention, e.g., txt, bin
    Return:
        file_list: a list of file_names
    """
    file_dir_tmp = file_dir if file_dir.endswith('/') else file_dir + '/'
    
    file_list = []
    for rootdir, dirs, files in os.walk(file_dir_tmp, followlinks=True):
        tmp_path = rootdir.replace(file_dir_tmp, '')
        # concatenate lists may be slow for large data set
        # change it in the future
        if file_ext:
            file_list += [os.path.splitext(os.path.join(tmp_path, x))[0] \
                          for x in files if x.endswith(file_ext)]    
        else:
            file_list += [os.path.splitext(os.path.join(tmp_path, x))[0] \
                          for x in files]    
    return file_list

def listdir_with_ext(file_dir, file_ext=None, recursive=False):
    """
    file_list = lstdir_with_ext(file_dir, file_ext=None, recursive=False)
    Return a list of file names with specified extention

    Args:
        file_dir: a file directory
        file_ext: string, specify the extention, e.g., txt, bin
        recursive: bool, whether search recursively (default False)
    Return: 
        file_list: a list of file_names
    """
    if not recursive:
        return listdir_with_ext_flat(file_dir, file_ext)
    else:
        return listdir_with_ext_recur(file_dir, file_ext)

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
    Return: 
        flag: bool
    """
    return set(list_b) <= set(list_a)

def members_in_a_not_in_b(list_a, list_b):
    """ members_in_a_not_b(list_a, list_b):
    Return a list of members that are in list_a but not in list_b
    
    Args:
        list_a: list
        list_b: list
    Return: 
        list
    """
    return list(set(list_a) - set(list_b))

def read_list_from_text(filename, f_chop=True):
    """out_list = read_list_from_text(filename, f_chop=True)
    Read a text file and return a list, where each text line is one element
    
    Args:
      filename: str, path to the file
      f_chop: bool, whether trim the newline symbol at the end of each line
              (default True)
    Return:
      output_list: list, each element is one line in the input text file
    """
    data = []
    with open(filename,'r') as file_ptr: 
        for line in file_ptr:
            line = nii_str_tool.string_chop(line) if f_chop else line
            data.append(line)
    return data

def write_list_to_text_file(data_list, filepath, endl='\n'):
    """write_list_to_text(data_list, filepath, endl='\n')              
    Save a list of data to a text file                                 
                                                                       
    Args:                                                              
      data_list: list, data list to be saved                           
      filepath: str, path to the output text file                      
      endl: str, ending of each new line, default \n                   
                                                                       
    If each element in data_list is not str, it will be converted to   
    str by str().                                                      
    """
    with open(filepath, 'w') as file_ptr:
        for data_entry in data_list:
            if type(data_entry) is str:
                file_ptr.write(data_entry + endl)
            else:
                file_ptr.write(str(data_entry) + endl)
    return

if __name__ == "__main__":
    #print("Definition of tools for list operation")
    
    input_list1 = read_list_from_text(sys.argv[1])
    input_list2 = read_list_from_text(sys.argv[2])
    residual_list = members_in_a_not_in_b(input_list1, input_list2)
    for filename in residual_list:
        print(filename)
    
    
