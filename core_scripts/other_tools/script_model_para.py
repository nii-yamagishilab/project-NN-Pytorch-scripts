#!/usr/bin/env python
"""
A simple wrapper to show parameter of the model

Usage:
  # go to the model directory, then
  $: python script_model_para.py 
  
  We assume model.py and config.py are available in the project directory.

"""

from __future__ import print_function
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"


def f_model_show(pt_model):
    """                                                                      
    f_model_show(pt_model)                                                   
    Args: pt_model, a Pytorch model                                          
                                                                             
    Print the informaiton of the model                                       
    """
    #f_model_check(pt_model)
    
    print(pt_model)
    num = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    print("Parameter number: {:d}".format(num))
    for name, p in pt_model.named_parameters():
        if p.requires_grad:
            print("Layer: {:s}\tPara. num: {:<10d} ({:02.1f}%)\tShape: {:s}"\
                  .format(name, p.numel(), p.numel()*100.0/num, str(p.shape)))
    return


if __name__ == "__main__":
    
    sys.path.insert(0, os.getcwd())

    if len(sys.argv) == 3:
        prj_model = importlib.import_module(sys.argv[1])
        prj_conf = importlib.import_module(sys.argv[2])
    else:
        print("By default, load model.py and config.py")
        prj_model = importlib.import_module("model")
        prj_conf = importlib.import_module("config")

    input_dims = sum(prj_conf.input_dims)
    output_dims = sum(prj_conf.output_dims)

    model = prj_model.Model(input_dims, output_dims, None)
    f_model_show(model)
