#!/usr/bin/env python
"""
nn_manager

utilities used by nn_manager

"""
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import torch

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#############################################################

def f_state_dict_wrapper(state_dict, data_parallel=False):
    """ a wrapper to take care of state_dict when using DataParallism

    f_model_load_wrapper(state_dict, data_parallel):
    state_dict: pytorch state_dict
    data_parallel: whether DataParallel is used
    
    https://discuss.pytorch.org/t/solved-keyerror-unexpected-
    key-module-encoder-embedding-weight-in-state-dict/1686/3
    """
    if data_parallel is True:
        # if data_parallel is used
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('module'):
                # if key is not starting with module, add it
                name = 'module.' + k
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if not k.startswith('module'):
                name = k
            else:
                # remove module.
                name = k[7:] 
            new_state_dict[name] = v
        return new_state_dict

def f_process_loss(loss):
    """ loss, loss_value = f_process_loss(loss):
    Input:
      loss: returned by loss_wrapper.compute
      It can be a torch.tensor or a list of torch.tensor
      When it is a list, it should look like:
       [[loss_1, loss_2, loss_3],
        [true/false,   true/false,  true.false]]
      where true / false tells whether the loss should be taken into 
      consideration for early-stopping

    Output:
      loss: a torch.tensor
      loss_value: a torch number of a list of torch number
    """
    if type(loss) is list:
        loss_sum = loss[0][0]
        loss_list = [loss[0][0].item()]
        if len(loss[0]) > 1:
            for loss_tmp in loss[0][1:]:
                loss_sum += loss_tmp
                loss_list.append(loss_tmp.item())
        return loss_sum, loss_list, loss[1]
    else:
        return loss, [loss.item()], [True]


def f_load_pretrained_model_partially(model, model_paths, model_name_prefix):
    """ f_load_pretrained_model_partially(model, model_paths, model_name_prefix)
    
    Initialize part of the model with pre-trained models
    
    Input:
    -----
       model: torch model
       model_paths: list of path to pre-trained models
       model_prefix: list of model name prefix used by model
            for example, pre_trained_model.*** may be referred to as 
            model.m_part1.*** in the new model. The prefix is "m_part1."
    
    Output:
    ------
       None
    """
    if type(model_paths) is str:
        model_path_tmp = [model_paths]
    else:
        model_path_tmp = model_paths
    if type(model_name_prefix) is str:
        model_prefix_tmp = [model_name_prefix]
    else:
        model_prefix_tmp = model_name_prefix

    model_dict = model.state_dict()

    for model_path, prefix in zip(model_path_tmp, model_prefix_tmp):
        if prefix[-1] != '.':
            # m_part1. not m_part
            prefix += '.'
        
        pretrained_dict = torch.load(model_path)
        
        # 1. filter out unnecessary keys
        pretrained_dict = {prefix + k: v \
                           for k, v in pretrained_dict.items() \
                           if prefix + k in model_dict}
        print("Load model {:s} as {:s} ({:d} parameter buffers)".format(
            model_path, prefix, len(pretrained_dict.keys())))
        
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return


if __name__ == "__main__":
    print("nn_manager_tools")
