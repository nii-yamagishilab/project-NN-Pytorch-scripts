#!/usr/bin/env python
"""
API for plotting

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.pyplot import cm

# turn of latex and set font type
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

############
## Utilities
############
_marker_bag = ["*","o","v","^","<",">","x","s","p","P","h","H","+",".","D","d","|","_"]
_line_style_bag = ['-','--', '-.', ':']
_color_bag = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def get_marker(idx):
    return _marker_bag[idx % len(_marker_bag)]


def get_colors(total, colormap='jet'):
    color_bag = []
    color_func = None
    # pre-defined color bags
    if colormap == 'self_1':
        color_bag = np.array([[78, 78, 78], [132,145,252],
                              [253,134,141], [110,143,82],
                              [229,173,69], [139,139,139]])/255.0
    elif colormap == 'self_2':
        color_bag = np.array([[178, 180, 253], [253, 178, 179]])/255.0

    elif colormap == 'self_3':
        color_bag = np.array([[1.0, 1.0, 1.0], [0.95, 0.95, 0.95],
                              [0.90, 0.90, 0.90], [0.85, 0.85, 0.85],
                              [0.80, 0.80, 0.80]])
    elif colormap == 'self_4':
        color_bag = np.array([[174,174,174], [13,13,13], [11,36,251],
                              [252,13,27], [55,183,164], [189,27,189],
                              [26,120,148], [110,143,82]])/255.0;
    elif colormap == 'self_5':
        color_bag = np.array([[132,145,252], [253,134,141],
                              [110,143,82], [229,173,69], 
                              [139,139,139], [200,200,200]])/255.0;
    elif colormap == 'self_6':
        color_bag = np.array([[243,243,243],[202,202,202],[160,160,160]])/255.0
    else:
        color_func = cm.get_cmap(colormap)
    
    if color_func is not None:
        indices = np.linspace(0,1,total)
        return [color_func(idx) for idx in indices]        
    else:
        return [color_bag[x % len(color_bag)] for x in range(total)]

def get_color(idx, total, colormap='jet'):
    return get_colors(total, colormap)[idx]

#################
## initialization
#################
# figure size
default_fig_size = (10, 5)
matplotlib.rcParams['figure.figsize'] = default_fig_size



###############
## function API
###############
def plot_API(data, plot_func, split_mode='single', config_dic={}, verbose=False):
    """
    fig, axis = plot_API(data, plot_func, split_mode='single', config_dic={})
    
    Plot a figure on data, using plot_func with config_dic
    
    input
    -----
      data: either a np.array or a list of np.array
      plot_func: the function to plot a single figure. see plot_lib
      split_mode: 'single': plot data in a single figure 
                  'v': plot data in separate rows. data must be a list
                  'h': plot data in separate columns. data must be a list
                  'grid': plot in a grid. config_dic['ncols'] and config_dic['nrows']
                          are used to decide the number of rows and columns for gridspec
      config_dic: configuration dictionary
      verbose: whether print out the config_dic information for each figure (default: False)
    
    output
    ------
      fig: fig handler
      axis: axis handler of the last figure
      
    """
    
    # figure size
    if "figsize" not in config_dic or config_dic["figsize"] is None:
        figsize = default_fig_size
    else:
        figsize = config_dic["figsize"]
    fig = plt.figure(figsize=figsize)
    
    
    # 
    if split_mode == 'single':
        # plot in a single figure
        
        axis = fig.add_subplot(111)
        if type(data) is list:
            for idx, data_entry in enumerate(data):
                tmp_config = process_config_dic(config_dic, idx)
                fig, axis = plot_func(data_entry, fig, axis, tmp_config)
                util_options(fig, axis, tmp_config)
        else:
            fig, axis = plot_func(data, fig, axis, config_dic)
            util_options(fig, axis, config_dic)
        if verbose:
            print(str(config_dic))
            
    elif split_mode == 'grid' or split_mode == 'h' or split_mode == 'v':
        # plot in a grid
        
        # data must be a list
        if type(data) is not list:
            print("split_mode == h requires list of input data")
            sys.exit(1)
        
        # decide number of row and col
        if split_mode == 'h':
            # horizontol mode
            nrows = 1
            ncols = len(data)
        elif split_mode == 'v':
            # vertical mode
            ncols = 1
            nrows = len(data)
        else:
            # grid mode
            if "ncols" in config_dic and "nrows" not in config_dic:
                ncols = config_dic["ncols"]
                nrows = int(np.ceil(len(data) * 1.0 / ncols))    
            elif "ncols" not in config_dic and "nrows" in config_dic:
                nrows = config_dic["nrows"]
                ncols = int(np.ceil(len(data) * 1.0 / nrows))
            elif "ncols" not in config_dic and "nrows" not in config_dic:
                nrows = int(np.sqrt(len(data)))
                ncols = int(np.ceil(len(data) * 1.0 / nrows))
            else:
                nrows = config_dic["nrows"]
                ncols = config_dic["ncols"]
                if nrows * ncols < len(data):
                    print("nrows * ncols < len(data)")
                    sys.exit(1)
        
        # grid 
        wspace = config_dic["wspace"] if "wspace" in config_dic else None
        hspace = config_dic["hspace"] if "hspace" in config_dic else None
        gs = GridSpec(nrows, ncols, figure=fig, wspace=wspace, hspace=hspace)
        
        # buffers
        xlim_bag = [np.inf, -np.inf]
        ylim_bag = [np.inf, -np.inf]
        axis_bags = []
        
        # plot    
        for idx, data_entry in enumerate(data):     
            
            if "col_first" in config_dic and config_dic["col_first"]:
                col_idx = idx % ncols
                row_idx = idx // ncols
            else:
                row_idx = idx % nrows
                col_idx = idx // nrows
            
            axis = fig.add_subplot(gs[row_idx, col_idx])
            tmp_config = process_config_dic(config_dic, idx)
            fig, axis = plot_func(data_entry, fig, axis, tmp_config)
            util_options(fig, axis, tmp_config)
            if verbose:
                print(str(tmp_config))
            
            axis_bags.append(axis)
            xlim_bag = [min(xlim_bag[0], axis.get_xlim()[0]),
                        max(xlim_bag[1], axis.get_xlim()[1])]
            ylim_bag = [min(ylim_bag[0], axis.get_ylim()[0]),
                        max(ylim_bag[1], axis.get_ylim()[1])]
            
            if "sharey" in config_dic and config_dic["sharey"]:
                if col_idx > 0:
                    axis.set_yticks([])
                    axis.set_ylabel("")
            if "sharex" in config_dic and config_dic["sharex"]:
                if row_idx < nrows -1:
                    axis.set_xticks([])
                    axis.set_xlabel("")
                    
        for axis in axis_bags:
            if "sharex" in config_dic and config_dic["sharex"]:
                axis.set_xlim(xlim_bag)
            if "sharey" in config_dic and config_dic["sharey"]:
                axis.set_ylim(ylim_bag)
    else:
        print("Unknown split_mode {:s}".format(split_mode))
        axis = None
        
    return fig, axis


def util_options(fig, axis, config_dic):
    if "xlabel" in config_dic:
        axis.set_xlabel(config_dic["xlabel"])
    if "ylabel" in config_dic:
        axis.set_ylabel(config_dic["ylabel"])
    if "title" in config_dic:
        axis.set_title(config_dic["title"])
    if "xlim" in config_dic:
        axis.set_xlim(config_dic["xlim"])
    if "ylim" in config_dic:
        axis.set_ylim(config_dic["ylim"])
    if "legend" in config_dic:
        axis.legend(**config_dic["legend"])
    if "xticks" in config_dic:
        axis.set_xticks(config_dic["xticks"])
        if "xticklabels" in config_dic:
            axis.set_xticklabels(config_dic["xticklabels"])
    if "yticks" in config_dic:
        axis.set_yticks(config_dic["yticks"])
        if "yticklabels" in config_dic:
            axis.set_yticklabels(config_dic["yticklabels"])
    if "grid" in config_dic:
        axis.grid(**config_dic["grid"])
    return fig, axis


def process_config_dic(input_dic, idx):
    """ input_dic may contain global configuration and sub configuration
    for each sub-figures.
    
    >> config = {"xlabel": "time",
          "ylabel": "amplitude",
          "sharey": True,
          "sharex": True,
          "plot": {"alpha": 0.3},
          "sub1": [{"legend": {"labels": ["s1", "s2"], "loc":2}},
                   {"legend": {"labels": ["s3", "s4"], "loc":2}}]}

    >> plot_API.process_config_dic(config, 0)
    {'xlabel': 'time',
     'ylabel': 'amplitude',
     'sharey': True,
     'sharex': True,
     'plot': {'alpha': 0.3},
     'legend': {'labels': ['s1', 's2'], 'loc': 2}}
    """
    global_dic = {x:y for x, y in input_dic.items() if not x.startswith("sub")}
    for x, y in input_dic.items():
        if x.startswith("sub"):
            if type(y) is not list:
                print("{:s} is not list".format(str(y)))
                sys.exit(1)
            if idx < len(y):
                sub_dic = y[idx]
                global_dic.update(sub_dic)
    return global_dic
    

if __name__ == "__main__":
    print("Definition of plot_API")
