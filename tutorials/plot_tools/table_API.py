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
    value = wrap_value((value - val_min) / (val_max - val_min), wrap) * scale
    # only use RGB, not RGBA
    color_code = color_func(value)[:-1]
    color_code = ', '.join(["{:0.2f}".format(x) for x in color_code])
    return r"\cellcolor[rgb]{" + color_code + "}"
    
def print_table(data_array, column_tag, row_tag, 
                print_format="1.2f", 
                with_color_cell = True,
                colormap='Greys', colorscale=0.5, colorwrap=0, col_sep='', 
                print_latex_table=True, print_text_table=True):
    """
    print a latex table given the data and tags
    
    input
    -----
      data_array: np.array [M, N]
      column_tag: list of str, length N, tag in the first row
      row_tag: list of str, length M, tags in first col of each row
      
      print_format: str, default "1.2f", used to specify number format
      
      with_color_cell: bool, default True,
                      whether to use color in each latex cell
      colormap: str, color map name (matplotlib)
      colorscale: float, default 0.5, 
                  the color will be (0, colorscale)
      colorwrap: float, default 0, wrap the color-value mapping curve
                 colorwrap > 0 works like mels-scale curve
      col_sep: str, additional string to separate columns. 
                  You may use '\t' or ',' for CSV
      print_latex_table: bool, print the table as latex command (default True)
      print_text_table: bool, print the table as text format (default True)

    output
    ------
      None
      
    Tables will be printed to the screen.
    The latex table will be surrounded by begin{tabular}...end{tabular}
    It can be directly pasted to latex file.
    However, it requires usepackage{colortbl} to show color in table cell.
    
    """
    # color configuration
    color_func = cm.get_cmap(colormap)
    value_min = np.min(data_array[data_array != np.inf])
    value_max = np.max(data_array[data_array != np.inf])
    
    def get_latex_color(x):
        # return a color command for latex cell
        return return_latex_color_cell(x, value_min, value_max, 
                                       colorscale, colorwrap, color_func)
    
    # maximum width for tags in 1st column
    row_tag_max_len = max([len(x) for x in row_tag])
    # maximum width for data and tags for other columns
    col_tag_max_len = max(
        [len("{num:{form}}".format(num=x, form=print_format)) \
         for x in data_array.flatten()])
    col_tag_max_len = max([len(x) for x in column_tag] + [col_tag_max_len])
    
    # prepare buffer
    text_buffer = ""
    latex_buffer = ""
    
    # latex head
    latex_buffer += r"\begin{tabular}{" \
                    + ''.join(['c' for x in column_tag + ['']]) + r"}" + "\n"
    
    # head row
    #  for latex
    hrow = [fill_cell("", row_tag_max_len)] \
                + [fill_cell(x, col_tag_max_len) for x in column_tag]
    latex_buffer += return_one_row_latex(hrow)
    #  for plain text (add additional separator for each column)
    hrow = [fill_cell("", row_tag_max_len, col_sep)] \
           + [fill_cell(x, col_tag_max_len, col_sep) for x in column_tag]
    text_buffer += return_one_row_text(hrow)
    
    # contents
    row = data_array.shape[0]
    col = data_array.shape[1]
    for row_idx in np.arange(row):
        # row head
        row_content_latex = [fill_cell(row_tag[row_idx], row_tag_max_len)]
        row_content_text = [fill_cell(row_tag[row_idx],row_tag_max_len,col_sep)]
        
        # each column in the raw
        for col_idx in np.arange(col):
            if not np.isinf(data_array[row_idx,col_idx]):
                num_str = "{num:{form}}".format(num=data_array[row_idx,col_idx],
                                                form=print_format)
                latex_color_cell = get_latex_color(data_array[row_idx,col_idx])
            else:
                num_str = ''
                latex_color_cell = ''
                
            if not with_color_cell:
                latex_color_cell = ''
                
            row_content_text.append(
                fill_cell(num_str, col_tag_max_len, col_sep))

            row_content_latex.append(
                fill_cell(latex_color_cell + ' ' + num_str, col_tag_max_len))
            
        # latex table content
        latex_buffer += return_one_row_latex(row_content_latex)
        # text content
        text_buffer += return_one_row_text(row_content_text)
        
    latex_buffer += r"\end{tabular}" + "\n"

    if print_latex_table:
        print(latex_buffer)
    if print_text_table:
        print(text_buffer)
    return


if __name__ == "__main__":
    print("Tools for printing table for latex")
