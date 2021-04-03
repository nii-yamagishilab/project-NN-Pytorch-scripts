#!/usr/bin/env python
"""
nn_manager

utilities used by nn_manager

"""
from __future__ import print_function

from collections import OrderedDict
import numpy as np
import torch

import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.other_tools.display as nii_display
import core_scripts.nn_manager.nn_manager_conf as nii_nn_manage_conf

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

def f_save_epoch_name(args, epoch_idx, suffix=''):
    """ str = f_save_epoch_name(args, epoch_idx)
    Return the name of the model file saved during training

    Args: 
      args: argument object by arg_parse, we will use
            args.save_epoch_name, args.save_model_dir, args.save_model_ext
      epoch_idx:, int, epoch index
      suffix: a suffix to the name (default '')

    Return: 
      str: name of epoch state file, str, e.g. epoch_001.pt
    """
    tmp_name = "{}_{:03d}".format(args.save_epoch_name, epoch_idx) + suffix
    return nii_str_tk.f_realpath(args.save_model_dir, tmp_name, \
                                 args.save_model_ext)

def f_save_trained_name(args, suffix=''):
    """ str = f_save_trained_name(args)
    Return the name of the best trained model file

    Args: 
      args: argument object by arg_parse
            args.save_trained_name, args.save_model_dir, args.save_model_ext
      suffix: a suffix added to the name (default '')

    Return: 
      str: name of trained network file, e.g., trained_network.pt
    """    
    return nii_str_tk.f_realpath(
        args.save_model_dir, args.save_trained_name + suffix, 
        args.save_model_ext)


def f_model_check(pt_model, model_type=None):
    """ f_model_check(pt_model)
    Check whether the model contains all the necessary keywords 
    
    Args: 
    ----
      pt_model: a Pytorch model
      model_type_flag: str or None, a flag indicating the type of network

    Return:
    -------
    """
    nii_display.f_print("Model check:")
    if model_type in nii_nn_manage_conf.nn_model_keywords_bags:
        keywords_bag = nii_nn_manage_conf.nn_model_keywords_bags[model_type]
    else:
        keywords_bag = nii_nn_manage_conf.nn_model_keywords_default
    
    for tmpkey in keywords_bag.keys():
        flag_mandatory, mes = keywords_bag[tmpkey]

        # mandatory keywords
        if flag_mandatory:
            if not hasattr(pt_model, tmpkey):
                nii_display.f_print("Please implement %s (%s)" % (tmpkey, mes))
                nii_display.f_die("[Error]: found no %s in Model" % (tmpkey))
            else:
                print("[OK]: %s found" % (tmpkey))
        else:
            if not hasattr(pt_model, tmpkey):
                print("[OK]: %s is ignored, %s" % (tmpkey, mes))
            else:
                print("[OK]: use %s, %s" % (tmpkey, mes))
        # done
    nii_display.f_print("Model check done\n")
    return

def f_model_show(pt_model, do_model_def_check=True, model_type=None):
    """ f_model_show(pt_model, do_model_check=True)
    Print the informaiton of the model

    Args: 
      pt_model, a Pytorch model
      do_model_def_check, bool, whether check model definition (default True)
      model_type: str or None (default None), what type of network

    Return:
      None
    """
    if do_model_def_check:
        f_model_check(pt_model, model_type)

    nii_display.f_print("Model infor:")
    print(pt_model)
    num = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    nii_display.f_print("Parameter number: {:d}\n".format(num), "normal")
    return


def f_loss_check(loss_module, model_type=None):
    """ f_loss_check(pt_model)
    Check whether the loss module contains all the necessary keywords 
    
    Args: 
    ----
      loss_module, a class
      model_type, a str or None
    Return:
    -------
    """
    nii_display.f_print("Loss check")
    
    if model_type in nii_nn_manage_conf.loss_method_keywords_bags:
        keywords_bag = nii_nn_manage_conf.loss_method_keywords_bags[model_type]
    else:
        keywords_bag = nii_nn_manage_conf.loss_method_keywords_default

    for tmpkey in keywords_bag.keys():
        flag_mandatory, mes = keywords_bag[tmpkey]

        # mandatory keywords
        if flag_mandatory:
            if not hasattr(loss_module, tmpkey):
                nii_display.f_print("Please implement %s (%s)" % (tmpkey, mes))
                nii_display.f_die("[Error]: found no %s in Loss" % (tmpkey))
            else:
                # no need to print other information here
                pass #print("[OK]: %s found" % (tmpkey))
        else:
            if not hasattr(loss_module, tmpkey):
                # no need to print other information here
                pass #print("[OK]: %s is ignored, %s" % (tmpkey, mes))
            else:
                print("[OK]: use %s, %s" % (tmpkey, mes))
        # done
    nii_display.f_print("Loss check done\n")
    return

def f_loss_show(loss_module, do_loss_def_check=True, model_type=None):
    """ f_model_show(pt_model, do_model_check=True)
    Print the informaiton of the model

    Args: 
      pt_model, a Pytorch model
      do_model_def_check, bool, whether check model definition (default True)
      model_type: str or None (default None), what type of network

    Return:
      None
    """
    # no need to print other information here
    # because loss is usually not a torch.Module

    #nii_display.f_print("Loss infor:")
    if do_loss_def_check:
        f_loss_check(loss_module, model_type)
    #print(loss_module)
    return

if __name__ == "__main__":
    print("nn_manager_tools")
