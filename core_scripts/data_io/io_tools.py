#!/usr/bin/env python
"""
io_tools

Functions to load data

"""
from __future__ import absolute_import

import os
import sys
import json
import numpy as np

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

def f_read_raw_mat(filename, col, data_format='f4', end='l'):
    """read_raw_mat(filename,col,data_format='float',end='l')
       Read the binary data from filename
       Return data, which is a (N, col) array
    
       filename: the name of the file, take care about '\\'
       col:      the number of column of the data
       format:   please use the Python protocal to write format
                 default: 'f4', float32
                 see for more format:
       end:      little endian 'l' or big endian 'b'?
                 default: 'l'
       
       dependency: numpy
       Note: to read the raw binary data in python, the question
             is how to interprete the binary data. We can use
             struct.unpack('f',read_data) to interprete the data
             as float, however, it is slow.
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
    """f_read_raw_mat_length(filename,data_format='float',end='l')
       Read length of data
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
    """read_htk(filename, data_format='f4', end='l')
        Read HTK File and return the data as numpy.array 
        filename:   input file name
        data_format:     the data_format of the data
                    default: 'f4' float32
        end:        little endian 'l' or big endian 'b'?
                    default: 'l'
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

    head_type = np.dtype([('nSample',data_formatInt4), ('Period',data_formatInt4),
                          ('SampleSize',data_formatInt2), ('kind',data_formatInt2)])
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
    """read_htk(filename, data_format='f4', end='l')
        Read HTK File and return the data as numpy.array 
        filename:   input file name
        data_format:     the data_format of the data
                    default: 'f4' float32
        end:        little endian 'l' or big endian 'b'?
                    default: 'l'
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

    head_type = np.dtype([('nSample',data_formatInt4), ('Period',data_formatInt4),
                          ('SampleSize',data_formatInt2), ('kind',data_formatInt2)])
    f = open(filename,'rb')
    head_info = np.fromfile(f,dtype=head_type,count=1)
    f.close()
    
    sample_size = int(head_info['SampleSize'][0]/4)
    return sample_size

def f_write_raw_mat(data,filename,data_format='f4',end='l'):
    """write_raw_mat(data,filename,data_format='',end='l')
       Write the binary data from filename. 
       Return True
       
       data:     np.array
       filename: the name of the file, take care about '\\'
       data_format:   please use the Python protocal to write data_format
                 default: 'f4', float32
       end:      little endian 'l' or big endian 'b'?
                 default: '', only when data_format is specified, end
                 is effective
       
       dependency: numpy
       Note: we can also write two for loop to write the data using
             f.write(data[a][b]), but it is too slow
    """
    if not isinstance(data, np.ndarray):
        print("Error write_raw_mat: input shoul be np.array")
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


def f_write_htk(data,targetfile,sampPeriod=50000,sampKind=9,data_format='f4',end='l'):
    """
    write_htk(data,targetfile,
    sampPeriod=50000,sampKind=9,data_format='f4',end='l')
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
    
    Args:
      file_path: string, path to the file

    Returns:
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
    
    Args:
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


