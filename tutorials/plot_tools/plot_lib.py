from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt

def plot_signal(data, fig, axis, config_dic):
    if "plot" in config_dic:
        axis.plot(data, **config_dic["plot"])
    else:
        axis.plot(data)
    return fig, axis

def plot_hist(data, fig, axis, config_dic):
    if "hist" in config_dic:
        axis.hist(data, **config_dic["hist"])
    else:
        axis.hist(data, histtype='step', density=True, bins=100)
    return fig, axis