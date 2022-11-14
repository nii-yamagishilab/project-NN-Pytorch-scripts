#!/usr/bin/env python
"""
Library of utilities for printing latex table
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#####################
## Latex table
#####################
def return_one_row_latex(content_buffer):
    return " & ".join(content_buffer) + r"\\ " + "\n"
        
def return_one_row_text(content_buffer):
    return " ".join(content_buffer) + "\n"

def fill_cell(text, length, sep=''):
    return "{str:^{wid}}".format(str=text, wid=length) + sep
    
def wrap_value(data, wrap_factor=0):
    if wrap_factor == 0:
        return data
    else:
        ratio = (1+wrap_factor) / (1-wrap_factor)
        return np.power((1 - np.power(1 - data, ratio)), 1/ratio)

def return_latex_color_cell(value, val_min, val_max, scale, wrap, color_func):
    
    # clip the value for color rendering
    value = np.clip(value, val_min, val_max)
    
    # normalized value
    if scale < 0:
        value = wrap_value((value - val_min) / (val_max - val_min), wrap)*-scale
        value = -scale - value
    else:
        value = wrap_value((value - val_min) / (val_max - val_min), wrap)*scale

    # only use RGB, not RGBA
    color_code = color_func(value)[:-1]
    
    color_code = ', '.join(["{:0.2f}".format(x) for x in color_code])
    return r"\cellcolor[rgb]{" + color_code + "}"

def is_valid_float(val):
    try:
        float(val)
    except ValueError:
        return False
    else:
        if val != np.inf and val == val:
            return True
        else:
            return False

def return_valid_number_idx(data_array):
    """return the index of data ceil that has valid nummerical value
    """
    is_numeric_3 = np.vectorize(is_valid_float, otypes = [bool])
    return is_numeric_3(data_array)

    
def print_table(data_array, column_tag, row_tag, 
                print_format = "1.2f", 
                with_color_cell = True,
                colormap='Greys', 
                colorscale = 0.5, 
                colorwrap = 0, 
                col_sep = '', 
                print_latex_table=True, 
                print_text_table=True,
                print_format_along_row=True,
                color_minmax_in = 'global',
                pad_data_column = 0,
                pad_dummy_col = 0,
                func_after_row = None,
                data_display_array = None):
    """
    print a latex table given the data (np.array) and tags    
    step1. table will be normalized so that values will be (0, 1.0)
    step2. each normalzied_table[i,j] will be assigned a RGB color tuple 
           based on color_func( normalzied_table[i,j] * color_scale)

    input
    -----
      data_array: np.array [M, N]
      column_tag: list of str, length N, tag in the first row
      row_tag: list of str, length M, tags in first col of each row
      
      print_format: str or list of str, specify the format to print number
                    default "1.2f"
      print_format_along_row: bool, when print_format is a list, is this
                    list specified for rows? Default True
                    If True, row[n] will use print_format[n]
                    If False, col[n] will use print_format[n]

      with_color_cell: bool, default True,
                      whether to use color in each latex cell
      colormap: str, color map name (matplotlib)
      colorscale: float, default 0.5, 
                    normalized table value will be scaled 
                    color = color_func(nomrlized_table[i,j] * colorscale)
                  list of float
                    depends on configuration of color_minmax_in
                    if color_minmax_in = 'row', colorscale[i] for the i-th row
                    if color_minmax_in = 'col', colorscale[j] for the j-th row
                  np.array
                    color_minmax_in cannot be 'row' or 'col'. 
                    colorscale[i, j] is used for normalized_table[i, j]
      colorwrap: float, default 0, wrap the color-value mapping curve
                 colorwrap > 0 works like mels-scale curve
      col_sep: str, additional string to separate columns. 
               You may use '\t' or ',' for CSV
      print_latex_table: bool, print the table as latex command (default True)
      print_text_table: bool, print the table as text format (default True)
      color_minmax_in: how to decide the max and min to compute cell color?
                 'global': get the max and min values from the input matrix 
                 'row': get the max and min values from the current row
                 'col': get the max and min values from the current column
                  (min, max): given the min and max values
                 default is global
      pad_data_column: int, pad columns on the left or right of data matrix
                  (the tag column will still be on the left)
                  0: no padding (default)
                  -N: pad N dummy data columns to the left
                   N: pad N dummy data columns to the right

      pad_dummy_col: int, pad columns to the left or right of the table
                  (the column will be padded to the left of head column)
                  0: no padding (default)
                  N: pad N columns to the left

    output
    ------
      latext_table, text_table
      
    Tables will be printed to the screen.
    The latex table will be surrounded by begin{tabular}...end{tabular}
    It can be directly pasted to latex file.
    However, it requires usepackage{colortbl} to show color in table cell.    
    """
    
    # default column and row are empty string
    if column_tag is None:
        column_tag = ["" for data in data_array[0, :]]
    if row_tag is None:
        row_tag = ["" for data in data_array]
    
    # 
    if data_display_array is None:
        data_display_array = data_array + np.nan
        flag_data_display = False
    else:
        flag_data_display = True

    # if padding of the data array is necessary
    if pad_data_column < 0:
        column_tag = ["" for x in range(-pad_data_column)] + column_tag
        dummy_col = np.zeros([data_array.shape[0], -pad_data_column]) + np.nan
        data_array = np.concatenate([dummy_col, data_array], axis=1)
        data_display_array = np.concatenate([dummy_col, data_display_array], axis=1)
    elif pad_data_column > 0:
        column_tag = ["" for x in range(pad_data_column)] + column_tag
        dummy_col = np.zeros([data_array.shape[0], pad_data_column]) + np.nan
        data_array = np.concatenate([data_array, dummy_col], axis=1)
        data_display_array = np.concatenate([data_display_array, dummy_col], axis=1)
    else:
        pass

    # check print_format
    if type(print_format) is not list:
        if print_format_along_row:
            # repeat the tag
            print_format = [print_format for x in row_tag]
        else:
            print_format = [print_format for x in column_tag]
    else:
        if print_format_along_row:
            assert len(print_format) == len(row_tag)
        else:
            assert len(print_format) == len(column_tag)


    # color configuration
    color_func = cm.get_cmap(colormap)
    #data_idx = return_valid_number_idx(data_array)    
    #value_min = np.min(data_array[data_idx])
    #value_max = np.max(data_array[data_idx])
    
    def get_latex_color(data_array, row_idx, col_idx, color_minmax_in):
        x = data_array[row_idx, col_idx]
        if color_minmax_in == 'row':
            data_idx = return_valid_number_idx(data_array[row_idx])
            value_min = np.min(data_array[row_idx][data_idx])
            value_max = np.max(data_array[row_idx][data_idx])
            if type(colorscale) is list:
                colorscale_tmp = colorscale[row_idx]
        elif color_minmax_in == 'col':
            data_idx = return_valid_number_idx(data_array[:, col_idx])
            value_min = np.min(data_array[:, col_idx][data_idx])
            value_max = np.max(data_array[:, col_idx][data_idx])    
            if type(colorscale) is list:
                colorscale_tmp = colorscale[col_idx]
        elif type(color_minmax_in) is tuple or type(color_minmax_in) is list:
            value_min = color_minmax_in[0]
            value_max = color_minmax_in[1]
            if type(colorscale) is np.ndarray:
                colorscale_tmp = colorscale[row_idx, col_idx]
        else:
            data_idx = return_valid_number_idx(data_array)
            value_min = np.min(data_array[data_idx])
            value_max = np.max(data_array[data_idx])
            if type(colorscale) is np.ndarray:
                colorscale_tmp = colorscale[row_idx, col_idx]
            
        if type(colorscale) is not list:
            colorscale_tmp = colorscale
            

        # return a color command for latex cell
        return return_latex_color_cell(x, value_min, value_max, 
                                       colorscale_tmp, colorwrap, color_func)
    
    # maximum width for tags in 1st column
    row_tag_max_len = max([len(x) for x in row_tag])

    # maximum width for data and tags for other columns
    if print_format_along_row:
        tmp_len = []
        for idx, data_row in enumerate(data_array):
            if len(print_format[0]):
                if flag_data_display:
                    max_len = max([len(x) for x in data_display_array[idx]])
                else:
                    max_len = max([len("{num:{form}}".format(num=x, 
                                                             form=print_format[idx])) \
                                   for x in data_row])
                
                tmp_len.append(max_len)
            else:
                tmp_len.append(0)
    else:
        tmp_len = []
        for idx, data_col in enumerate(data_array.T):
            if len(print_format[0]):
                if flag_data_display:
                    max_len = max([len(x) for x in data_display_array[:, idx]])
                else:
                    max_len = max([len("{num:{form}}".format(num=x, 
                                                   form=print_format[idx])) \
                                   for x in data_col])
                tmp_len.append(max_len)
            else:
                tmp_len.append(0)

    col_tag_max_len = max([len(x) for x in column_tag] + tmp_len)
    
    # prepare buffer
    text_buffer = ""
    latex_buffer = ""
    text_cell_buffer = []
    latex_cell_buffer = []

    # latex head
    if pad_dummy_col > 0:
        latex_buffer += r"\begin{tabular}{" \
                        + ''.join(['c' for x in column_tag + ['']])
        latex_buffer += ''.join(['c' for x in range(pad_dummy_col)]) + r"}"+"\n"
    else:
        latex_buffer += r"\begin{tabular}{" \
                        + ''.join(['c' for x in column_tag + ['']]) + r"}"+"\n"

    latex_buffer += r"\toprule" + "\n"
    
    # head row
    #  for latex
    hrow = [fill_cell("", row_tag_max_len)] \
           + [fill_cell(x, col_tag_max_len) for x in column_tag]
    if pad_dummy_col > 0:
        hrow = [fill_cell("", 1) for x in range(pad_dummy_col)] + hrow

    latex_buffer += return_one_row_latex(hrow)
    latex_buffer += r"\midrule" + "\n"

    latex_cell_buffer.append(hrow)

    #  for plain text (add additional separator for each column)
    hrow = [fill_cell("", row_tag_max_len, col_sep)] \
           + [fill_cell(x, col_tag_max_len, col_sep) for x in column_tag]
    text_buffer += return_one_row_text(hrow)
    text_cell_buffer.append(hrow)

    # contents
    row = data_array.shape[0]
    col = data_array.shape[1]
    for row_idx in np.arange(row):
        # row head
        row_content_latex = [fill_cell(row_tag[row_idx], row_tag_max_len)]
        row_content_text = [fill_cell(row_tag[row_idx],row_tag_max_len,col_sep)]
        
        if pad_dummy_col > 0:
            row_content_latex = [fill_cell("", 1) for x in range(pad_dummy_col)] \
                                + row_content_latex

        # each column in the raw
        for col_idx in np.arange(col):

            if print_format_along_row:
                tmp_print_format = print_format[row_idx]
            else:
                tmp_print_format = print_format[col_idx]

            if is_valid_float(data_array[row_idx,col_idx]):
                if len(tmp_print_format):
                    num_str = "{num:{form}}".format(
                        num=data_array[row_idx,col_idx],
                        form=tmp_print_format)
                else:
                    num_str = ""
                latex_color_cell = get_latex_color(
                    data_array, row_idx, col_idx,
                    color_minmax_in)

            elif type(data_array[row_idx,col_idx]) is str:
                if len(tmp_print_format):
                    num_str = "{num:{form}}".format(
                        num=data_array[row_idx,col_idx],
                        form=tmp_print_format)
                else:
                    num_str = ""
                latex_color_cell = ''
            else:
                num_str = ''
                latex_color_cell = ''
                
            if not with_color_cell:
                latex_color_cell = ''
                
            if flag_data_display:
                num_str = data_display_array[row_idx, col_idx]

            row_content_text.append(
                fill_cell(num_str, col_tag_max_len, col_sep))

            row_content_latex.append(
                fill_cell(latex_color_cell + ' ' + num_str, col_tag_max_len))
            
        # latex table content
        latex_buffer += return_one_row_latex(row_content_latex)
        latex_cell_buffer.append(row_content_latex)
        # text content
        text_buffer += return_one_row_text(row_content_text)
        text_cell_buffer.append(row_content_text)

        if func_after_row is not None: 
            latex_buffer += func_after_row(row_idx)
        
            
        

    latex_buffer += r"\bottomrule" + "\n"
    latex_buffer += r"\end{tabular}" + "\n"

    if print_latex_table:
        print(latex_buffer)
    if print_text_table:
        print(text_buffer)
    return latex_buffer, text_buffer, latex_cell_buffer, text_cell_buffer



def concatenate_table(table_list, ignore_initial=True, 
                      add_separator=1, latex=True):
    """
    """
    rows = [len(x) for x in table_list]
    if len(list(set(rows))) > 1:
        print("Input tables have different row numbers")
        return None
    
    output_text = ""
    output_table = []
    for row in range(len(table_list[0])):
        temp = []
        for idx, subtable in enumerate(table_list):
            if ignore_initial:
                temp += subtable[row][1:]
            else:
                temp += subtable[row]
            if add_separator and idx < len(table_list)-1:
                temp += ['' for x in range(add_separator)]
        output_table.append(temp)
        output_text += return_one_row_latex(temp)
        
        
    # latex head 
    latex_buffer = r"\begin{tabular}{" \
                 + ''.join(['c' for x in temp + ['']]) + r"}" + "\n"
    latex_buffer += output_text
    latex_buffer += r"\end{tabular}" + "\n"
    
    return latex_buffer, output_table

if __name__ == "__main__":
    print("Tools for printing table for latex")

    # example
    data = np.random.randn(5, 3)
    col_tags = ['A', 'B', 'C']
    row_tags = ['1', '2', '3', '4', '5']
    _ = print_table(data, col_tags, row_tags)
    
    # Latex code of the colored table will be printed
