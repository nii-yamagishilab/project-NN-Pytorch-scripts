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