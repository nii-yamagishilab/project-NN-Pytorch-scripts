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


def f_load_checkpoint(checkpoint, args, flag_multi_device, pt_model, optimizer, 
                      monitor_trn, monitor_val, lr_scheduler):
    """ f_load_checkpoint(checkpoint, args, pt_model, optimizer, 
                      monitor_trn, monitor_val, lr_scheduler)
    Load checkpoint.
    
    Input:
      checkpoint: check point saved by the script. Either a dict or a pt model
      args: command line arguments when running the script
      flag_multi_device: bool, does this code uses multiple GPUs?

    Input (which will be modified by the function)
      pt_model: Pytorch model, this will load the model saved in checkpoint
      optimizer: optimizer, this will load the optimizer saved in checkpoint
      monitor_trn: log of loss on training set
      monitor_val: log of loss on validation set
      lr_scheduler: scheudler of learning rate

    Output:
      train_log: str, text log of training loss
    """
    #
    train_log = ''

    if checkpoint is None:
        # no checkpoint
        return train_log
    
    # checkpoint exist
    cp_names = nii_nn_manage_conf.CheckPointKey()

    if args.allow_mismatched_pretrained_model:
        if type(checkpoint) is dict:
            # if it is a epoch*.pt, ignore training histories
            # only load the model parameter
            nii_display.f_print("allow-mismatched-pretrained-model is on")
            nii_display.f_print("ignore training history in pre-trained model")
            pretrained_dict = f_state_dict_wrapper(
                    checkpoint[cp_names.state_dict], flag_multi_device)
        else:
            # it is only a dictionary of trained parameters
            pretrained_dict = f_state_dict_wrapper(
                    checkpoint, flag_multi_device)
                    
        # target model dict
        model_dict = pt_model.state_dict()
                
        # methods similar to f_load_pretrained_model_partially
        # 1. filter out mismatched keys
        pre_dict_tmp = {
            k: v for k, v in pretrained_dict.items() \
            if k in model_dict \
            and model_dict[k].numel() == pretrained_dict[k].numel()}
            
        mismatch_keys = [k for k in model_dict.keys() if k not in pre_dict_tmp]
                
        if mismatch_keys:
            print("Partially load model, ignoring buffers: {:s}".format(
                ' '.join(mismatch_keys)))
                    
            # 2. overwrite entries in the existing state dict
            model_dict.update(pre_dict_tmp)
        
            # 3. load the new state dict
            pt_model.load_state_dict(model_dict)
        else:
            # the usual case
            # only model status
            pt_model.load_state_dict(pretrained_dict)
            nii_display.f_print("Load pretrained model")
    else:
        if type(checkpoint) is dict:
            # checkpoint is a dict (trained model + optimizer + other logs)
            
            # load model parameter and optimizer state
            if cp_names.state_dict in checkpoint:
                # wrap the state_dic in f_state_dict_wrapper 
                # in case the model is saved when DataParallel is on
                pt_model.load_state_dict(
                    f_state_dict_wrapper(checkpoint[cp_names.state_dict], 
                                         flag_multi_device))

            # load optimizer state
            if cp_names.optimizer in checkpoint and \
               not args.ignore_optimizer_statistics_in_trained_model:
                optimizer.load_state_dict(checkpoint[cp_names.optimizer])
            
            # optionally, load training history
            if not args.ignore_training_history_in_trained_model:
                #nii_display.f_print("Load ")
                if cp_names.trnlog in checkpoint:
                    monitor_trn.load_state_dic(checkpoint[cp_names.trnlog])

                if cp_names.vallog in checkpoint and monitor_val:
                    monitor_val.load_state_dic(checkpoint[cp_names.vallog])

                if cp_names.info in checkpoint:
                    train_log = checkpoint[cp_names.info]
                    
                if cp_names.lr_scheduler in checkpoint and \
                   checkpoint[cp_names.lr_scheduler] and lr_scheduler.f_valid():
                    lr_scheduler.f_load_state_dict(
                        checkpoint[cp_names.lr_scheduler])
                    
                nii_display.f_print("Load check point, resume training")
            else:
                nii_display.f_print("Load pretrained model and optimizer")
        else:
            
            # the usual case
            # only model status
            pt_model.load_state_dict(
                f_state_dict_wrapper(checkpoint, flag_multi_device))
            nii_display.f_print("Load pretrained model")

    return train_log

def f_load_checkpoint_for_inference(checkpoint, pt_model):
    """ f_load_checkpoint_for_inference(checkpoint, pt_model)
    Load checkpoint for model inference
    
    No matter what is inside the checkpoint, only load the model parameters
    """
    cp_names = nii_nn_manage_conf.CheckPointKey()
    if type(checkpoint) is dict and cp_names.state_dict in checkpoint:
        pt_model.load_state_dict(checkpoint[cp_names.state_dict])
    else:
        pt_model.load_state_dict(checkpoint)
    return


def f_load_pretrained_model_partially(model, model_paths, model_name_prefix):
    """ f_load_pretrained_model_partially(model, model_paths, model_name_prefix)
    
    Initialize part of the model with pre-trained models.
    This function can be used directly. It is also called by nn_manager.py
    if model.g_pretrained_model_path and model.g_pretrained_model_prefix are 
    defined. 

    For reference: 
    https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    Input:
    -----
       model: torch model
       model_paths: list of str, list of path to pre-trained models (.pt files)
       model_prefix: list of str, list of model name prefix used by model
            
    Output:
    ------
       None

    For example, 
        case1: A module in a pretrained model may be called model.*** 
               This module will be model.m_part1.*** in the new model. 
               Then the prefix is "m_part1."
               new_model.m_part1.*** <- pre_trained_model.***
        case2: A module in a pretrained model may be called model.*** 
               This module will still be model..*** in the new model. 
               Then the prefix is ""
               new_model.*** <- pre_trained_model.***

    Call f(model, ['./asr.pt', './tts.pt'], ['asr.', 'tts.']), then
       model.asr <- load_dict(asr.pt)
       model.tts <- load_dict(tts.pt)
    """
    cp_names = nii_nn_manage_conf.CheckPointKey()

    # change string to list
    if type(model_paths) is str:
        model_path_tmp = [model_paths]
    else:
        model_path_tmp = model_paths
    if type(model_name_prefix) is str:
        model_prefix_tmp = [model_name_prefix]
    else:
        model_prefix_tmp = model_name_prefix

    # get the dictionary format of new model
    model_dict = model.state_dict()

    # for each pre-trained model
    for model_path, prefix in zip(model_path_tmp, model_prefix_tmp):
        if prefix == '':
            pass
        elif prefix[-1] != '.':
            # m_part1. not m_part
            prefix += '.'
        
        pretrained_dict = torch.load(model_path)
        
        # if this is a epoch***.pt, load only the network weight
        if cp_names.state_dict in pretrained_dict:
            pretrained_dict = pretrained_dict[cp_names.state_dict]

        # 1. filter out unnecessary keys
        pretrained_dict = {prefix + k: v \
                           for k, v in pretrained_dict.items() \
                           if prefix + k in model_dict}
        print("Load model {:s} as {:s} ({:d} parameter buffers, ".format(
            model_path, prefix, len(pretrained_dict.keys())), end=' ')
        print("{:d} parameters)".format(
            sum([pretrained_dict[x].numel() for x in pretrained_dict.keys()])))
        
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
