#!/usr/bin/env python
"""
Library of plotting functions
"""
from __future__ import absolute_import
from __future__ import print_function

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import copy

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

##################
## Basic functions
##################

def plot_signal(data, fig, axis, config_dic):
    """ plot signal as 1D sequence, using matplotlib.plot

    Args
    ----
      data: np.array, (L, 2) or (L, 1) or (L, )
      fig: matplotlib figure handle
      axis: matplotlib axis handle
      config: dictionary for axis.bar
    
    L: number of vertial bar groups
    When data is (L, 2), use data[:, 0] as x, data[:, 1] as y
    
    Optional field in config_dic['plot_bar']:
    6. other options for matplotlib.pyplot.bar

    """
    if type(data) is list:
        print("plot_signal no longer support list input")
        print("data will be converted to np.array")
        x = data[0]
        y = data[1]
        if "plot_signal" in config_dic:
            axis.plot(x, y, **config_dic["plot_signal"])
        else:
            axis.plot(x, y)
    elif type(data) is np.ndarray:
        if data.ndim == 2 and data.shape[1] == 2:
            x = data[:, 0]
            y = data[:, 1]
        elif data.ndim == 2 and data.shape[1] == 1:
            y = data[:, 0]
            x = np.arange(y.shape[0])
        elif data.ndim == 1:
            y = data
            x = np.arange(y.shape[0])
        else:
            print("dimension of data is not supported")
            sys.exit(1)
    
        if "plot_signal" in config_dic:
            axis.plot(x, y, **config_dic["plot_signal"])
        else:
            # default configuration
            axis.plot(x, y)
    else:
        print("Type of data is not np.array")
        sys.exit(1)
    return fig, axis

def plot_hist(data, fig, axis, config_dic):
    if "plot_hist" in config_dic:
        axis.hist(data, **config_dic["plot_hist"])
    else:
        # default configuration
        axis.hist(data, histtype='step', density=True, bins=100)
    return fig, axis

def plot_imshow(data, fig, axis, config_dic):
    """
    """
    plot_colorbar = False        
    if "plot_imshow" in config_dic:
        temp_dic = copy.deepcopy(config_dic["plot_imshow"])
        if "colorbar" in temp_dic:
            plot_colorbar = temp_dic['colorbar']
            temp_dic.pop("colorbar")
        pos = axis.imshow(data, **temp_dic)
    else:
        # default configuration
        pos = axis.imshow(data, cmap='jet', origin='lower', aspect='auto')
    if type(plot_colorbar) is dict:
        fig.colorbar(pos, **plot_colorbar)
    elif plot_colorbar:
        fig.colorbar(pos)
    else:
        pass
    return fig, axis

def plot_scatter(data, fig, axis, config_dic):
    """
    """
    if type(data) is list:
        x = data[0]
        y = data[1]
        # size
        s = data[2] if len(data) > 2 else None
        # color
        c = data[3] if len(data) > 3 else None
    else:
        x = data[:, 0]
        y = data[:, 1]
        s = data[:, 2] if data.shape[1] > 2 else None
        c = data[:, 3] if data.shape[1] > 3 else None
    
    if "plot_scatter" in config_dic:
        if "s" in config_dic["plot_scatter"]:
            if "c" in config_dic["plot_scatter"]:
                axis.scatter(x, y, **config_dic["plot_scatter"])
            else:
                axis.scatter(x, y, c = c, **config_dic["plot_scatter"])
        else:
            if "c" in config_dic["plot_scatter"]:
                axis.scatter(x, y, s, **config_dic["plot_scatter"])
            else:
                axis.scatter(x, y, s, c, **config_dic["plot_scatter"])
    else:
        # default configuration
        axis.scatter(x, y, s, c)
    return fig, axis


def plot_bar(data, fig, axis, config):
    """plot_bar(data, fig, axis, config_dic)
    
    Args
    ----
      data: np.array, (L, M)
      fig: matplotlib figure handle
      axis: matplotlib axis handle
      config: dictionary for axis.bar
    
    L: number of vertial bar groups
    M: number of sub-bar in each group
    
    If yerr is to be used, please save it in config_dic['plot_bar']['yerr']
    config_dic['plot_bar']['yerr'] should have the same shape as data.

    Optional field in config_dic['plot_bar']:
    1. ['x']: np.array or list, location of each bar group on x-axis
    2. ['show_number']: str, format of showing numbers on top of each var
         "{:{form}}".format(form=config_dic['plot_bar']['show_number'])
    4. ['color_list']: list of str, list of colors for each column of data
    5. ['edgecolor_list']: list of str, list of edge colors for each column of data
    6. other options for matplotlib.pyplot.bar
    
    """
    if type(data) is list:
        data = np.asarray(data)
    if data.ndim == 1:
        data = np.expand_dims(data, 1)

    width = 0.3
    x_pos = np.arange(data.shape[0])
    yerr = None
    show_number = None
    group_size = data.shape[1]
    config_dic = copy.deepcopy(config)
    color_list = [None for x in range(group_size)]
    edgecolor_list = [None for x in range(group_size)]
    if "plot_bar" in config_dic:
        if "x" in config_dic["plot_bar"]:
            x_pos = config_dic["plot_bar"]['x']
            config_dic['plot_bar'].pop('x')
            
        if "width" in config_dic['plot_bar']:
            width = config_dic['plot_bar']['width']
            config_dic['plot_bar'].pop('width')
            
        if "color" in config_dic["plot_bar"]:
            color_list = [config_dic["plot_bar"]['color'] for x in range(group_size)]
            config_dic["plot_bar"].pop('color')
        if "edgecolor" in config_dic['plot_bar']:
            edgecolor_list = [config_dic["plot_bar"]['edgecolor'] for x in range(group_size)]
            config_dic["plot_bar"].pop('edgecolor')
            
        if "yerr" in config_dic['plot_bar']:
            yerr = config_dic['plot_bar']['yerr']
            config_dic['plot_bar'].pop('yerr')
            
        if "show_number" in config_dic['plot_bar']:
            show_number = config_dic['plot_bar']['show_number']
            config_dic['plot_bar'].pop('show_number')
            
        if "color_list" in config_dic['plot_bar']:
            color_list =  config_dic['plot_bar']['color_list']
            config_dic['plot_bar'].pop('color_list')

        if "edgecolor_list" in config_dic['plot_bar']:
            edgecolor_list =  config_dic['plot_bar']['edgecolor_list']
            config_dic['plot_bar'].pop('edgecolor_list')
            
    if "color_list" in config_dic:
        color_list =  config_dic['color_list']
        config_dic.pop('color_list')
        print("color_list should be in dic['plot_bar']")
        
    for group_idx in range(group_size): 
        x_shift = group_idx * width - width * (group_size - 1) / 2
        if yerr is None:
            sub_yerr = None
        else:
            sub_yerr = yerr[:, group_idx]
            
        if "plot_bar" in config_dic:
            axis.bar(x_pos + x_shift, data[:, group_idx], width=width, 
                     color = color_list[group_idx],
                     edgecolor = edgecolor_list[group_idx],
                     yerr = sub_yerr, **config_dic["plot_bar"])
        else:
            axis.bar(x_pos + x_shift, data[:, group_idx], width=width,
                     color = color_list[group_idx],
                     edgecolor = edgecolor_list[group_idx],                     
                     yerr = sub_yerr)

        if show_number is not None:
            for x, y in zip(x_pos, data[:, group_idx]):
                axis.text(x + x_shift, y*1.01, 
                          "{num:{form}}".format(num=y, form=show_number),
                          horizontalalignment="center")
    return fig, axis


def plot_boxplot(data, fig, axis, config):
    """ 
    """
    # get config for obxplot
    bp_config = copy.deepcopy(config["plot_boxplot"]) \
                if "plot_boxplot" in config else {}

    if "plot_marker" in bp_config:
        marker_config = bp_config["plot_marker"]
        bp_config.pop("plot_marker")
    else:
        marker_config = {}
    
    # filter data
    data_for_plot = []
    data_mean = []
    if type(data) is list:
        for col in range(len(data)):
            idx = ~np.bitwise_or(np.isinf(data[col]), np.isnan(data[col]))
            data_for_plot.append(data[col][idx])
            data_mean.append(data[col][idx].mean())    
    else:
        for col in range(data.shape[1]):
            idx = ~np.bitwise_or(np.isinf(data[:, col]), np.isnan(data[:, col]))
            data_for_plot.append(data[idx, col])
            data_mean.append(data[idx, col].mean())
    
    # 
    axis.boxplot(data_for_plot, **bp_config)

    xpos = bp_config["positions"] if "positions" in bp_config \
           else np.arange(len(data_for_plot))+1
    
    if marker_config:
        axis.plot(xpos, data_mean, **marker_config)
    return fig, axis


def plot_err_bar(data, fig, axis, config_dic):
    """ plot_err_bar
    
    """
    if type(data) is list:
        if len(data) == 3:
            x = data[0]
            y = data[1]
            err = data[2]
        elif len(data) == 2:
            y = data[0]
            err = data[1]
            x = np.arange(y.shape[0])
        else:
            print("data must have 3 or 2 elements")
            sys.exit(1)
    else:
        if data.shape[1] == 3:
            x = data[:, 0]
            y = data[:, 1]
            err = data[:, 2]
        elif data.shape[1] == 2:
            x = np.arange(data.shape[0])
            y = data[:, 0]
            err = data[:, 1]
        else:
            print("data must have 3 or 2 columns")
            sys.exit(1)
            
    if "plot_err_bar" in config_dic:
        axis.errorbar(x, y, yerr=err, **config_dic["plot_err_bar"])
    else:
        axis.errorbar(x, y, yerr=err)
    return fig, axis



def plot_stacked_bar(data, fig, axis, config):
    """plot_bar(data, fig, axis, config_dic)
    
    Args
    ----
      data: np.array, (L, M)
      fig: matplotlib figure handle
      axis: matplotlib axis handle
      config: dictionary for axis.bar
    
    L: number of bars
    M: number of values to be stacked in each group
    
    If yerr is to be used, please save it in config_dic['plot_bar']['yerr']
    config_dic['plot_bar']['yerr'] should have the same shape as data.

    Optional field in config_dic['plot_bar']:
    1. ['x']: np.array or list, location of each bar group on x-axis
    2. ['show_number']: str, format of showing numbers on top of each var
         "{:{form}}".format(form=config_dic['plot_bar']['show_number'])
    4. ['color_list']: list of str, list of colors for each column of data
    5. ['edgecolor_list']: list of str, list of edge colors for each column of data
    6. other options for matplotlib.pyplot.bar
    
    """
    if type(data) is list:
        data = np.asarray(data)
    if data.ndim == 1:
        data = np.expand_dims(data, 1)

    width = 0.3
    x_pos = np.arange(data.shape[0])
    yerr = None
    show_number = None
    group_size = data.shape[1]
    config_dic = copy.deepcopy(config)
    color_list = [None for x in range(group_size)]
    edgecolor_list = [None for x in range(group_size)]
    if "plot_bar" in config_dic:
        if "x" in config_dic["plot_bar"]:
            x_pos = config_dic["plot_bar"]['x']
            config_dic['plot_bar'].pop('x')
            
        if "width" in config_dic['plot_bar']:
            width = config_dic['plot_bar']['width']
            config_dic['plot_bar'].pop('width')
            
        if "color" in config_dic["plot_bar"]:
            color_list = [config_dic["plot_bar"]['color'] for x in range(group_size)]
            config_dic["plot_bar"].pop('color')
        if "edgecolor" in config_dic['plot_bar']:
            edgecolor_list = [config_dic["plot_bar"]['edgecolor'] for x in range(group_size)]
            config_dic["plot_bar"].pop('edgecolor')
            
        if "yerr" in config_dic['plot_bar']:
            yerr = config_dic['plot_bar']['yerr']
            config_dic['plot_bar'].pop('yerr')
            
        if "show_number" in config_dic['plot_bar']:
            show_number = config_dic['plot_bar']['show_number']
            config_dic['plot_bar'].pop('show_number')
            
        if "color_list" in config_dic['plot_bar']:
            color_list =  config_dic['plot_bar']['color_list']
            config_dic['plot_bar'].pop('color_list')

        if "edgecolor_list" in config_dic['plot_bar']:
            edgecolor_list =  config_dic['plot_bar']['edgecolor_list']
            config_dic['plot_bar'].pop('edgecolor_list')
            
    if "color_list" in config_dic:
        color_list =  config_dic['color_list']
        config_dic.pop('color_list')
        print("color_list should be in dic['plot_bar']")
    
    stacked_accum = np.zeros([data.shape[0]])
    for group_idx in range(group_size): 
        if yerr is None:
            sub_yerr = None
        else:
            sub_yerr = yerr[:, group_idx]
            
        if "plot_bar" in config_dic:
            axis.bar(x_pos, data[:, group_idx], width=width, 
                     color = color_list[group_idx], bottom=stacked_accum,
                     edgecolor = edgecolor_list[group_idx],
                     yerr = sub_yerr, **config_dic["plot_bar"])
        else:
            axis.bar(x_pos, data[:, group_idx], width=width,
                     color = color_list[group_idx], bottom=stacked_accum,
                     edgecolor = edgecolor_list[group_idx],                     
                     yerr = sub_yerr)
        
        stacked_accum += data[:, group_idx]
        if show_number is not None:
            for x, y in zip(x_pos, data[:, group_idx]):
                axis.text(x, y*1.01, 
                          "{num:{form}}".format(num=y, form=show_number),
                          horizontalalignment="center")
    return fig, axis

def plot_table(data, fig, axis, config_dic):
    """plot_table(data, fig, axis, config_dic)
    Use plot_imshow to show table and print numbers
    """
    print_format = "1.2f"
    font_color = "r"

    if "plot_table" in config_dic:
        tmp_config = copy.deepcopy(config_dic["plot_table"])
        if "print_format" in tmp_config:
            print_format = tmp_config['print_format']
            tmp_config.pop('print_format')

        if "font_color" in tmp_config:
            font_color = tmp_config['font_color']
            tmp_config.pop('font_color')
    else:
        tmp_config = {'cmap':'jet', 'origin':'lower', 'aspect':'auto'}
        
    
    axis.imshow(data, **tmp_config)

    config_dic['xlim'] = (-0.5, data.shape[1]-0.5)
    config_dic['ylim'] = (-0.5, data.shape[0]-0.5)
    
    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            axis.text(col_idx, row_idx,
                      "{num:{form}}".format(
                          num=data[row_idx,col_idx],form=print_format),
                      ha='center', va='center', c=font_color)

    return fig, axis


def plot_barh(dataset, fig, axis, configs):
    """
    """
    max_bin_width = np.max([np.max(x) for x in dataset])
    
    if 'plot_barh' in configs:
        if 'show_percentage' in configs['plot_barh']:
            flag_show_per = configs['plot_barh']['show_percentage']
        if 'ad_max_bin_width' in configs['plot_barh']:
            max_bin_width *= configs['plot_barh']['ad_max_bin_width']
        if 'color' in configs['plot_barh']:
            color = configs['plot_barh']['color']
        else:
            color = 'Grey'
    
    if flag_show_per:
        dataset_tmp = dataset / np.sum(dataset, axis=1, keepdims=True) * 100
        max_bin_width = 100 * 1.05
    else:
        dataset_tmp = dataset
    
    # location of x
    x_locs = [x * max_bin_width for x in np.arange(len(dataset))]
    # temporary buffer
    max_length = np.max([x.shape[0] for x in dataset])
    # left most pos
    left_most = x_locs[-1]
    right_most = 0
    
    for idx, (x_loc, data) in enumerate(zip(x_locs, dataset_tmp)):
        
        # decide the left alignment position
        lefts = x_loc - 0 * data
        if 'plot_barh' in configs and 'align' in configs['plot_barh']:
            if  configs['plot_barh']['align'] == 'middle':
                lefts = x_loc - 0.5 * data
            elif configs['plot_barh']['align'] == 'right':
                lefts = x_loc - data
                
        # plot
        axis.barh(np.arange(len(data)), data, height=1.0, left=lefts, 
                  color=color)
        
        # show text
        if flag_show_per:
            for idx, (data_value, left_pos) in enumerate(zip(data, lefts)):
                axis.text(left_pos, idx,
                          '{:2.1f}'.format(data_value), 
                          ha='right', va='center')
                left_most = np.min([left_most, left_pos - max_bin_width * 0.4])
                #right_most = np.max([right_most, left_pos + max(data)])
            right_most = left_pos + max_bin_width * 1.05
    if flag_show_per:
        axis.set_xlim([left_most, right_most])
        
    if 'xticklabels' in configs:
        axis.set_xticks(x_locs)
        axis.set_xticklabels(configs['xticklabels'])
    if 'yticklabels' in configs:
        axis.set_yticks(np.arange(max_length))
        axis.set_yticklabels(configs['yticklabels'])
    return fig, axis

############################
## Specific functions
##  classification
############################

from scipy import special as scipy_special 
def probit(x):
    """ probit function to scale the axis
    based on __probit__(p) 
    https://
    projets-lium.univ-lemans.fr/sidekit/_modules/sidekit/bosaris/detplot.html
    """
    return np.sqrt(2) * scipy_special.erfinv(2.0 * x - 1)

def plot_det(data, fig, axis, config_dic):
    """ plot DET curves
    
    fig, axis = plot_det(data, fig, axis, config_dic)
    This function will do probit conversion
    
    input
    -----
      data: [frr, far] computed by compute_det_curve
      fig: fig handler
      axis: axis handler
      config_dic: configuration dictionary
    
    output
    ------
      fig: fig handler
      axis: axis handler
    """

    if type(data) is list:
        
        # warping through probit
        x = probit(data[1]) # far
        y = probit(data[0]) # frr
        
        # we will use plot_signal as back-end function for plotting DET curves
        tmp_config_dic = config_dic.copy()
        if "plot_det" in config_dic:
            tmp_config_dic["plot_signal"] = config_dic["plot_det"]

        # grid option
        if "grid" in config_dic and config_dic["grid"]["b"] is False:
            pass
        else:
            #axis.plot([probit(0.0001), probit(0.99)], 
            #          [probit(0.0001), probit(0.99)], 
            #          c='lightgrey', linestyle='--')
            #axis.plot([probit(0.5), probit(0.5)], 
            #          [probit(0.0001), probit(0.99)],     
            #          c='lightgrey', linestyle='--')
            #axis.plot([probit(0.0001), probit(0.99)], 
            #          [probit(0.5), probit(0.5)],      
            #          c='lightgrey', linestyle='--')
            pass

        # plot using the plot_signal function
        plot_signal(np.stack([x, y], axis=1), fig, axis, tmp_config_dic)

            
        # options on label
        if "xlabel" not in config_dic:
            axis.set_xlabel("False alarm probablity ({:s})".format("\%"))
        if "ylabel" not in config_dic:
            axis.set_ylabel("Miss probability ({:s})".format("\%"))
            
        # ticks
        if "xticks" not in config_dic:
            xticks_to_use = [0.005, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]            
        else:
            xticks_to_use = config_dic["xticks"]
            config_dic.pop("xticks", None)
        if "yticks" not in config_dic:            
            yticks_to_use = [0.005, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9]
        else:
            yticks_to_use = config_dic["yticks"]
            config_dic.pop("yticks", None)
        
        xticks_to_use_probit = [probit(x) for x in xticks_to_use]
        yticks_to_use_probit = [probit(x) for x in yticks_to_use] 
        
        axis.set_xticks(xticks_to_use_probit)
        axis.set_xticklabels(["%d" % (x * 100) for x in xticks_to_use])
        axis.set_yticks(yticks_to_use_probit)
        axis.set_yticklabels(["%d" % (x * 100) for x in yticks_to_use])
        
        if "xlim" in config_dic:
            config_dic["xlim"] = [probit(x) for x in config_dic["xlim"]]
        if "ylim" in config_dic:
            config_dic["ylim"] = [probit(x) for x in config_dic["ylim"]]
        
        if "grid" not in config_dic:
            axis.grid(True)
        
        # whether show EER on the figure
        if "eer" in config_dic and config_dic['eer']:
            abs_diffs = np.abs(data[1] - data[0])
            min_index = np.argmin(abs_diffs)
            eer = np.mean((data[1][min_index], data[0][min_index]))
            axis.text(probit(eer), probit(eer), 
                      "EER {:2.3}\%".format(eer * 100))
    else:
        print("plot_det requires input data = [far, frr]")
    return fig, axis

############################
## Specific functions
##  signal processing 
############################

import scipy
import scipy.signal
import numpy as np

def _spec(data, fft_bins=4096, frame_shift=40, frame_length=240):
    f, t, cfft = scipy.signal.stft(
        data, nfft=fft_bins, 
        noverlap=frame_length-frame_shift, nperseg=frame_length)
    return f,t,cfft

def _amplitude(cfft):
    mag = np.power(np.power(np.real(cfft),2) + np.power(np.imag(cfft),2), 0.5)
    return mag

def _amplitude_to_db(mag):
    return 20*np.log10(mag+ np.finfo(np.float32).eps)

def _spec_amplitude(data, fft_bins=4096, frame_shift=40, frame_length=240):
    _, _, cfft = _spec(data, fft_bins, frame_shift, frame_length)
    mag  = _amplitude(cfft)
    return _amplitude_to_db(mag)

def plot_spec(data, fig, axis, config_dic):
    """ 
    fig, axis = plot_spec(data, fig, axis, config_dic)
    This function will plot spectrogram, given configuration in config_dic
    
    input
    -----
      data: data
      fig: fig handler
      axis: axis handler
      config_dic: configuration dictionary
    
    output
    ------
      fig: fig handler
      axis: axis handler
    """

    if type(data) is list:
        print("plot_spectrogram only supports data array input, ")
        print("but it receives list of data")
        sys.exit(1)
    
    # default configuration
    tmp_dic = copy.deepcopy(config_dic["plot_spec"]) \
              if "plot_spec" in config_dic else {}
    
    if "sampling_rate" in tmp_dic:
        sr = tmp_dic["sampling_rate"]  
        tmp_dic.pop("sampling_rate")
    else:
        sr = None
    if "frame_shift" in tmp_dic:
        fs = tmp_dic["frame_shift"]
        tmp_dic.pop("frame_shift")
    else:
        fs = 80
    if "frame_length" in tmp_dic:
        fl = tmp_dic["frame_length"]
        tmp_dic.pop("frame_length")
    else:
        fl =  320
    if "fft_bins" in tmp_dic:
        fn = tmp_dic["fft_bins"] 
        fn.pop("fft_bins")
    else:
        fn = 1024
    
    # stft
    spec = _spec_amplitude(data, fn, fs, fl)
    
    tmp_config_dic = config_dic.copy()
    if tmp_dic:
        tmp_config_dic["plot_imshow"] = tmp_dic

    plot_imshow(spec, fig, axis, tmp_config_dic)
    
    # options on label
    if "xlabel" not in config_dic:
        axis.set_xlabel("Frame index")
        
    if "ylabel" not in config_dic:
        if sr is None:
            axis.set_ylabel("Frequency bins")
        else:
            axis.set_ylabel("Frequency (Hz)")
            
    # ticks
    if "yticks" not in config_dic:            
        yticks_to_use = [(fn//2+1)//2, fn//2+1]
    else:
        yticks_to_use = config_dic["yticks"]
        config_dic.pop("yticks", None)
    
    axis.set_yticks(yticks_to_use)
    if sr is not None:
        freq_str = ["{:4.1f}".format(x * sr // 2 // 1000 / (fn//2+1)) \
                    for x in yticks_to_use]
        axis.set_yticklabels(freq_str)
        
    return fig, axis


if __name__ == "__main__":
    print("Definition of plot_lib")
