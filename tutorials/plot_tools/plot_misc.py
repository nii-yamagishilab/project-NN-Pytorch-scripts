#!/usr/bin/env python
"""
Misc tools used before the start of plot_API and plot_lib

"""
from __future__ import absolute_import
from __future__ import print_function

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10, 5)

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

def plot_signal(data, fig, axis, xlabel='', ylabel='', title=''):
    """
    """
    axis.plot(data)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.set_xlim(0, data.shape[0]-1)
    return fig, axis

def plot_signal_stem(data, fig, axis, xlabel, ylabel):
    """
    """
    axis.stem(data, use_line_collection=True)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, data.shape[0]-1)
    return fig, axis

def plot_imshow(data, fig, axis, xlabel, ylabel):
    axis.imshow(data, aspect='auto', origin='lower', cmap='RdBu')
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return fig, axis

def plot_spectrogram(data, fig, axis, xlabel, ylabel, sampling_rate=None, title=''):
    axis.imshow(data, aspect='auto', origin='lower', cmap='jet')
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if sampling_rate:
        yticks = [data.shape[0]//2, data.shape[0]//4*3, data.shape[0]]
        ytickslabels = ["%1.1f" % (x / data.shape[0] * sampling_rate / 2.0 / 1000.0) for x in yticks]
        axis.set_yticks(yticks)
        axis.set_yticklabels(ytickslabels)
    if title:
        axis.set_title(title)
    return fig, axis

def plot_surface(data, fig, ax, xlabel='', ylabel='', zlabel='', angleX=30, angleY=30):
    
    X = np.arange(data.shape[0])
    Y = np.arange(data.shape[1])
    X, Y = np.meshgrid(X, Y)
    Z = data

    surf = ax.plot_surface(X, Y, Z.T, cmap='RdBu')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(angleX, angleY)
    return fig, ax

import numpy as np
import matplotlib

def plot_matrix(data, fig, axis_left=0.0, 
                axis_bottom=0.0, axis_length=1.0, colormap='Greys',
                color_norm = None,
                colorgrad_y=True, colorgrad_x=True, alpha=1.0, 
                int_width=1, deci_width=1):
    
    axis = fig.add_axes([axis_left, axis_bottom, axis_length, axis_length])
    
    # get colormap for each data point
    cmap = matplotlib.cm.get_cmap(colormap)
    color_num = colorgrad_x * data.shape[1] + colorgrad_y * (data.shape[0]) * (data.shape[1] ** colorgrad_x)
    color_num = int(color_num)
    n_more_color = int(np.sqrt(data.shape[0] * data.shape[1]))
    color_val = np.linspace(0, 1.0, color_num+n_more_color)
    
    # 
    bias = 0
    x_range = np.arange(data.shape[1]+1)
    y_range = np.arange(data.shape[0]+1)
    x_max = x_range[-1]
    y_max = y_range[-1]
    for x_idx in x_range:
        axis.plot([x_idx + bias, x_idx + bias], [bias, y_max+bias], 'k')
    for y_idx in y_range:
        axis.plot([bias, x_max + bias], [y_idx + bias, y_idx+bias], 'k')
    for x_idx in x_range[:-1]:
        for y_idx in y_range[:-1]:
            # plot from top to down
            y_idx_reverse = y_max - 1 - y_idx
            axis.text(x_idx+0.5+bias, y_idx_reverse+0.5+bias, 
                      "%*.*f" % (int_width, deci_width, data[y_idx, x_idx]),
                      color='k', horizontalalignment='center',
                      verticalalignment='center')
            x_tmp = np.array([x_idx, x_idx+1, x_idx+1, x_idx]) + bias
            y_tmp = np.array([y_idx_reverse, y_idx_reverse, 
                              y_idx_reverse+1, y_idx_reverse+1])+bias
            
            # get the color-idx
            color_idx = x_idx * colorgrad_x \
                + y_idx * colorgrad_y * (data.shape[1] ** colorgrad_x)
            if color_norm is not None:
                cell_color = cmap(color_norm(data[y_idx, x_idx]))
            else:
                cell_color = cmap(color_val[color_idx])
            axis.fill(x_tmp, y_tmp, color=cell_color, alpha=alpha)
    axis.axis('off')
    return axis

def plot_tensor(data_tensor, shift=0.1, colormap='Greys',
                color_on_value=False,
                colorgrad_y=True, colorgrad_x=True, alpha=1.0,
                title='', int_width=1, deci_width=1):
    """plot_tensor(data, shift=0.1, colormap='Greys',
                   color_on_value=False,
                   colorgrad_y=True, colorgrad_x=True, alpha=1.0,
                   title='', int_width=1, deci_width=1):
    Plot 3D tensor data. 
    
    data: tensor of shape (batchsize, length, dim)
    shift=0.1: space between different data in the batch
    colormap='Greys': color map for coloring
    color_on_value=False: select color of the cell based on data value
    colorgrad_y=True: when color_on_value=False, select color based on row index
    colorgrad_x=True: when color_on_value=False, select color based on column index
    alpha=1.0: alpha for matplotlib.plot
    title='': title of this data
    int_width=1: x in %x.yf, when displaying numbers in figure
    deci_width=1: y in %x.yf, when displaying numbers in figure
    """
    try:
        data = data_tensor.numpy()
    except AttributeError:
        data = data_tensor
    if data.ndim != 3:
        print("input data is not a 3d tensor ")
        return None,None
    
    fig_width = data.shape[2]/2 + (data.shape[0]-1)*shift
    fig_height = data.shape[1]/2 + (data.shape[0]-1)*shift
    fig = plt.figure(figsize=(fig_width, fig_height))
    axis_start = 0.0
    axis_end = 1.0
    axis_length = axis_end - shift * (data.shape[0] - 1) - axis_start
    
    
    if color_on_value:
        color_norm = lambda x: (x - data.min())/(data.max() - data.min()+0.0001)*0.6
    else:
        color_norm = None
    axis = []
    for idx in np.arange(data.shape[0]):
        axis.append(
            plot_matrix(
                data[idx], fig, 
                axis_start + shift * idx, 
                axis_start + shift * (data.shape[0]-1-idx),
                axis_length, 
                colormap, color_norm, 
                colorgrad_y, colorgrad_x, alpha, int_width, deci_width))
    if len(title):
        fig.text(0.5, 0.99, title, ha='center') 
    return fig, axis

if __name__ == "__main__":
    print("Misc tools from ../plot_lib.py")
