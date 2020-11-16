#!/usr/bin/env python
"""
Library of utilities for figure plotting
"""
from __future__ import absolute_import
from __future__ import print_function

from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#################
## Basic function
#################

def plot_signal(data, fig, axis, config_dic):
    """ plot signal
    """
    if type(data) is list:
        x = data[0]
        y = data[1]
        if "plot_signal" in config_dic:
            axis.plot(x, y, **config_dic["plot_signal"])
        else:
            axis.plot(x, y)
    else:
        if "plot_signal" in config_dic:
            axis.plot(data, **config_dic["plot_signal"])
        else:
            # default configuration
            axis.plot(data)
    return fig, axis

def plot_hist(data, fig, axis, config_dic):
    if "plot_hist" in config_dic:
        axis.hist(data, **config_dic["plot_hist"])
    else:
        # default configuration
        axis.hist(data, histtype='step', density=True, bins=100)
    return fig, axis

def plot_imshow(data, fig, axis, config_dic):
    if "plot_imshow" in config_dic:
        axis.hist(data, **config_dic["plot_imshow"])
    else:
        # default configuration
        axis.imshow(data, cmap='jet', origin='lower', aspect='auto')
    return fig, axis

def plot_scatter(data, fig, axis, config_dic):
    """
    """
    if type(data) is list:
        x = data[0]
        y = data[1]
        if len(data) > 2:
            s = data[2]
        else:
            s = None
    else:
        x = data[:, 0]
        y = data[:, 1]
        if data.shape[1] > 2:
            s = data[:, 2]
        else:
            s = None
    
    if "plot_scatter" in config_dic:
        axis.scatter(x, y, s, **config_dic["plot_scatter"])
    else:
        # default configuration
        axis.scatter(x, y, s)
    return fig, axis


#####################
## Specific functions
#####################

from scipy import special as scipy_special 
def probit(x):
    """ probit function to scale the axis
    based on __probit__(p) 
    https://projets-lium.univ-lemans.fr/sidekit/_modules/sidekit/bosaris/detplot.html
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
        x = probit(data[1]) # far
        y = probit(data[0]) # frr
        
        tmp_config_dic = config_dic.copy()
        if "plot_det" in config_dic:
            tmp_config_dic["plot_signal"] = config_dic["plot_det"]

        if "grid" in config_dic and config_dic["grid"]["b"] is False:
            pass
        else:
            axis.plot([probit(0.0001), probit(0.99)], [probit(0.0001), probit(0.99)], 
                      c='lightgrey', linestyle='--')
            axis.plot([probit(0.5), probit(0.5)], [probit(0.0001), probit(0.99)],     
                      c='lightgrey', linestyle='--')
            axis.plot([probit(0.0001), probit(0.99)], [probit(0.5), probit(0.5)],      
                      c='lightgrey', linestyle='--')
            
            
        plot_signal([x, y], fig, axis, tmp_config_dic)

        if "xlabel" not in config_dic:
            axis.set_xlabel("False alarm rate (FAR {:s})".format("\%"))
        if "ylabel" not in config_dic:
            axis.set_ylabel("Miss probability (FRR {:s})".format("\%"))
            
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
        
        if "eer" in config_dic and config_dic['eer']:
            abs_diffs = np.abs(data[1] - data[0])
            min_index = np.argmin(abs_diffs)
            eer = np.mean((data[1][min_index], data[0][min_index]))
            axis.text(probit(eer), probit(eer), "EER {:2.3}\%".format(eer * 100))

    else:
        print("plot_det requires input data = [far, frr]")
    return fig, axis
