#!/usr/bin/env python
"""
io_tools

Functions to load data

"""
from __future__ import absolute_import

import os
import sys
import json
import pickle
import numpy as np

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"

def f_read_raw_mat(filename, col, data_format='f4', end='l'):
    """data = f_read_raw_mat(filename, col, data_format='float', end='l')
    Read the binary data from filename
    Return data, which is a (N, col) array

    input
    -----    
       filename: str, path to the binary data on the file system
       col:      int, number of column assumed by the data matrix
       format:   str, please use the Python protocal to write format
                 default: 'f4', float32
       end:      str, little endian 'l' or big endian 'b'?
                 default: 'l'
    output
    ------
       data: np.array, shape (N, col), where N is the number of rows
           decided by total_number_elements // col
    """
    f = open(filename,'rb')
    if end=='l':
        data_format = '<'+data_format
    elif end=='b':
        data_format = '>'+data_format
    else:
        data_format = '='+data_format
    datatype = np.dtype((data_format,(col,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:,0]
    else:
        return data

def f_read_raw_mat_length(filename, data_format='f4'):
    """len = f_read_raw_mat_length(filename, data_format='f4')
    Read length of data, i.e., number of elements in the data file.
    If data is in shape (N, M), then len = N * M
    
    input
    -----
      filename: str, path to the binary data on the file system
      format:   str, please use the Python protocal to write format
                 default: 'f4', float32
    output
    ------
      len: int, number of data elements in the data file
    """
    f = open(filename,'rb')
    tmp = f.seek(0, 2)
    bytes_num = f.tell()
    f.close()
    if data_format == 'f4':
        return int(bytes_num / 4)
    else:
        return bytes_num

def f_read_htk(filename, data_format='f4', end='l'):
    """data = read_htk(filename, data_format='f4', end='l')
    Read HTK File and return the data as numpy.array
    
    input
    -----
       filename: str, path to the binary HTK data on file system
       data_format: str, format of the returned data
                    default: 'f4' float32
       end:        little endian 'l' or big endian 'b'?
                   default: 'l'
    output
    ------
       data: numpy.array
    """
    if end=='l':
        data_format = '<'+data_format
        data_formatInt4 = '<i4'
        data_formatInt2 = '<i2'
    elif end=='b':
        data_format = '>'+data_format
        data_formatInt4 = '>i4'
        data_formatInt2 = '>i2'
    else:
        data_format = '='+data_format
        data_formatInt4 = '=i4'
        data_formatInt2 = '=i2'

    head_type = np.dtype([('nSample',data_formatInt4), 
                          ('Period',data_formatInt4),
                          ('SampleSize',data_formatInt2), 
                          ('kind',data_formatInt2)])
    f = open(filename,'rb')
    head_info = np.fromfile(f,dtype=head_type,count=1)
    
    """if end=='l':
        data_format = '<'+data_format
    elif end=='b':
        data_format = '>'+data_format
    else:
        data_format = '='+data_format
    """    
    if 'f' in data_format:
        sample_size = int(head_info['SampleSize'][0]/4)
    else:
        print("Error in read_htk: input should be float32")
        return False
        
    datatype = np.dtype((data_format,(sample_size,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    return data


def f_read_htk_length(filename, data_format='f4', end='l'):
    """length = read_htk(filename, data_format='f4', end='l')
    Read HTK File and return the number of data elements in the file

    Read HTK File and return the data as numpy.array
    
    input
    -----
       filename: str, path to the binary HTK data on file system
       data_format: str, format of the returned data
                    default: 'f4' float32
       end:        little endian 'l' or big endian 'b'?
                   default: 'l'
    output
    ------
       length: int, number of data elements in the file
    """
    if end=='l':
        data_format = '<'+data_format
        data_formatInt4 = '<i4'
        data_formatInt2 = '<i2'
    elif end=='b':
        data_format = '>'+data_format
        data_formatInt4 = '>i4'
        data_formatInt2 = '>i2'
    else:
        data_format = '='+data_format
        data_formatInt4 = '=i4'
        data_formatInt2 = '=i2'

    head_type = np.dtype([('nSample',data_formatInt4), 
                          ('Period',data_formatInt4),
                          ('SampleSize',data_formatInt2), 
                          ('kind',data_formatInt2)])
    f = open(filename,'rb')
    head_info = np.fromfile(f,dtype=head_type,count=1)
    f.close()
    
    sample_size = int(head_info['SampleSize'][0]/4)
    return sample_size

def f_write_raw_mat(data, filename, data_format='f4', end='l'):
    """flag = write_raw_mat(data, filename, data_format='f4', end='l')
    Write data to file on the file system as binary data

    input
    -----
      data:     np.array, data to be saved
      filename: str, path of the file to save the data
      data_format:   str, data_format for numpy
                 default: 'f4', float32
      end: str   little endian 'l' or big endian 'b'?
                 default: 'l'

    output   
    ------
      flag: bool, whether the writing is done or not
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input should be np.array")
        return False
    f = open(filename,'wb')
    if len(data_format)>0:
        if end=='l':
            data_format = '<'+data_format
        elif end=='b':
            data_format = '>'+data_format
        else:
            data_format = '='+data_format
        datatype = np.dtype(data_format)
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f,'')
    f.close()
    return True

def f_append_raw_mat(data, filename, data_format='f4', end='l'):
    """flag = write_raw_mat(data, filename, data_format='f4', end='l')
    Append data to an existing file on the file system as binary data

    input
    -----
      data:     np.array, data to be saved
      filename: str, path of the file to save the data
      data_format:   str, data_format for numpy
                 default: 'f4', float32
      end: str   little endian 'l' or big endian 'b'?
                 default: 'l'

    output   
    ------
      flag: bool, whether the writing is done or not
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input shoul be np.array")
        return False
    f = open(filename,'ab')
    if len(data_format)>0:
        if end=='l':
            data_format = '<'+data_format
        elif end=='b':
            data_format = '>'+data_format
        else:
            data_format = '='+data_format
        datatype = np.dtype(data_format)
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f,'')
    f.close()
    return True

def f_write_htk(data, targetfile, 
                sampPeriod=50000, sampKind=9, data_format='f4', end='l'):
    """
    write_htk(data,targetfile,
      sampPeriod=50000,sampKind=9,data_format='f4',end='l')
    
    Write data as HTK-compatible format
    
    input
    -----
      data: np.array, data to be saved
      targetfile: str, path of the file to save the data
      ...
    
    output
    ------
    """
    if data.ndim==1:
        nSamples, vDim = data.shape[0], 1
    else:
        nSamples, vDim = data.shape
    if data_format=='f4':
        sampSize = vDim * 4;
    else:
        sampSize = vDim * 8;
    
    f = open(targetfile,'wb')

    if len(data_format)>0:
        if end=='l':
            data_format1 = '<i4'
            data_format2 = '<i2'
        elif end=='b':
            data_format1 = '>i4'
            data_format2 = '>i2'
        else:
            data_format1 = '=i4'
            data_format2 = '=i2'
    
    temp_data = np.array([nSamples, sampPeriod], 
                         dtype=np.dtype(data_format))
    temp_data.tofile(f, '')
    
    temp_data = np.array([sampSize, sampKind], dtype=np.dtype(data_format2))
    temp_data.tofile(f, '')
    
    
    if len(data_format)>0:
        if end=='l':
            data_format = '<'+data_format
        elif end=='b':
            data_format = '>'+data_format
        else:
            data_format = '='+data_format
        datatype = np.dtype(data_format)
        temp_data = data.astype(datatype)
    else:
        temp_data = data
    temp_data.tofile(f, '')
    f.close()
    return True

def read_dic(file_path):
    """ dic = read_dic(file_path)
    Read a json file from file_path and return a dictionary
    
    input
    -----
      file_path: string, path to the file

    output
    ------
      dic: a dictionary
    """
    try:
        data = json.load( open(file_path) )
    except IOError:
        print("Cannot find %s" % (file_path))
        sys.exit(1)
    except json.decoder.JSONDecodeError:
        print("Cannot parse %s" % (file_path))
        sys.exit(1)
    return data


        
def write_dic(dic, file_path):
    """ write_dic(dic, file_path)
    Write a dictionary to file
    
    input
    -----
      dic: dictionary to be dumped
      file_path: file to store the dictionary
    """

    try:
        json.dump(dic, open(file_path, 'w'))
    except IOError:
        print("Cannot write to %s " % (file_path))
        sys.exit(1)

def file_exist(file_path):
    """ file_exit(file_path)
    Whether file exists
    """
    return os.path.isfile(file_path) or os.path.islink(file_path)


def pickle_dump(data, file_path):
    """ pickle_dump(data, file_path)
    Dump data into a pickle file

    inputs:
      data: python object, data to be dumped
      file_path: str, path to save the pickle file
    """
    try:
        os.mkdir(os.path.dirname(file_path))
    except OSError:
        pass

    with open(file_path, 'wb') as file_ptr:
        pickle.dump(data, file_ptr)
    return

def pickle_load(file_path):
    """ data = pickle_load(file_path)
    Load data from a pickle dump file
    
    inputs:
      file_path: str, path of the pickle file
    
    output:
      data: python object
    """
    with open(file_path, 'rb') as file_ptr:
        data = pickle.load(file_ptr)
    return data


def wrapper_data_load_with_cache(file_path, method_data_load,
                                 cache_dir='__cache', 
                                 use_cached_data=True, verbose=False):
    """wrapper_data_load_with_cache(file_path, method_data_load,
         cache_dir='__cache', 
         use_cached_data=True, verbose=False):

    Load data from file and save data as pickle file in cache.
    
    input
    -----
      file_path: str, path of input file
      method_data_load: python function, funtion to load the data
      cache_dir: str, default __cache, the directory to save cached pickle file
      use_cached_data: bool, default True, use cached data when available
      verbose: bool, default False, print information on reading/writing
    
    output
    ------
      data: python object decided by method_data_load
    
    This method is useful to load large text file. No need to parse text 
    everytime because the data will be saved as pickle file in cache after
    the first time of execution

    Example:
    from core_scripts.data_io import io_tools
    from core_scripts.other_tools import list_tools
    data = io_tools.wrapper_data_load_with_cache('test_file', 
              list_tools.read_list_from_text)
    """
    try:
        os.mkdir(cache_dir)
    except OSError:
        pass

    cache_file_path = '_'.join(file_path.split(os.path.sep))
    cache_file_path = os.path.join(cache_dir, cache_file_path)
    cache_file_path += '.pkl'

    if use_cached_data and os.path.isfile(cache_file_path):
        if verbose:
            print("Load cached data {:s}".format(cache_file_path))
        return pickle_load(cache_file_path)
    else:
        data = method_data_load(file_path)
        pickle_dump(data, cache_file_path)
        if verbose:
            print("Load data {:s}".format(file_path))
            print("Save cahced data {:s}".format(cache_file_path))
        return data



if __name__ == "__main__":
    print("Definition of tools for I/O operation")
