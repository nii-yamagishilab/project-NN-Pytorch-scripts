from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (10, 5)


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


import numpy as np
import matplotlib

def plot_matrix(data, fig, axis_left=0.0, 
                axis_bottom=0.0, axis_length=1.0, colormap='Greys',
                color_norm = None,
                colorgrad_y=True, colorgrad_x=True, alpha=1.0):
    
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
                      "%.2f" % (data[y_idx, x_idx]),
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

def plot_tensor(data, fig, shift=0.2, colormap='Greys',
                color_on_value=False,
                colorgrad_y=True, colorgrad_x=True, alpha=1.0):
    if data.ndim != 3:
        print("input data is not a 3d tensor ")
        return
    axis_start = 0.0
    axis_end = 1.0
    axis_length = axis_end - shift * (data.shape[0] - 1) - axis_start
    
    if color_on_value:
        color_norm = lambda x: (x - data.min())/(data.max() - data.min()+0.0001)*0.6
    else:
        color_norm = None
    for idx in np.arange(data.shape[0]):
        plot_matrix(data[idx], fig, 
                    axis_start + shift * idx, 
                    axis_start + shift * (data.shape[0]-1-idx),
                    axis_length, 
                    colormap, color_norm, colorgrad_y, colorgrad_x, alpha)