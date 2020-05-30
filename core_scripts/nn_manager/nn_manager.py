#!/usr/bin/env python
"""
nn_manager

A simple wrapper to run the training / testing process

"""
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import datetime

import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.display as nii_display
import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.op_manager.op_process_monitor as nii_monitor
import core_scripts.op_manager.op_display_tools as nii_op_display_tk

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#############################################################

# name for the checkpoint keys
class CheckPointKey:
    state_dict = 'state_dict'
    info = 'info'
    optimizer = 'optimizer' 
    trnlog = 'train_log'
    vallog = 'val_log'
    

def f_save_epoch_name(args, epoch_idx):
    """ 
    f_save_epoch_name(args, epoch_idx)
    Args: args, argument object by arg_parse
          epoch_idx, int, epoch index

    Return: name of epoch state file, str, e.g. epoch_001.pt
    """
    tmp_name = "{}_{:03d}".format(args.save_epoch_name, epoch_idx)
    return nii_str_tk.f_realpath(args.save_model_dir, tmp_name, \
                                 args.save_model_ext)

def f_save_trained_name(args):
    """ 
    f_save_trained_name(args)
    Args: args, argument object by arg_parse

    Return: name of trained network file, e.g., trained_network.pt
    """    
    return nii_str_tk.f_realpath(args.save_model_dir, \
                                 args.save_trained_name, \
                                 args.save_model_ext)

def f_model_show(pt_model):
    """ 
    f_model_show(pt_model)
    Args: pt_model, a Pytorch model
    
    Print the informaiton of the model
    """
    print(pt_model)
    num = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    nii_display.f_print("Parameter number: {:d}".format(num), "normal")
    return
    

def f_run_one_epoch(args,
                    pt_model, loss_wrapper, \
                    device, monitor,  \
                    data_loader, epoch_idx, optimizer = None):
    """
    f_run_one_epoch: 
       run one poech over the dataset (for training or validation sets)

    Args:
       args:         from argpase
       pt_model:     pytorch model (torch.nn.Module)
       loss_wrapper: a wrapper over loss function
                     loss_wrapper.compute(generated, target) 
       device:       torch.device("cuda") or torch.device("cpu")
       monitor:      defined in op_procfess_monitor.py
       data_loader:  pytorch DataLoader. 
       epoch_idx:    int, index of the current epoch
       optimizer:    torch optimizer or None
                     if None, the back propgation will be skipped
                     (for developlement set)
    """
    # timer
    start_time = time.time()
        
    # loop over samples
    for data_idx, (data_in, data_tar, data_info, idx_orig) in \
        enumerate(data_loader):

        # idx_orig is the original idx in the dataset
        # which can be different from data_idx when shuffle = True
        #idx_orig = idx_orig.numpy()[0]
        #data_seq_info = data_info[0]    
        
        # send data to device
        if optimizer is not None:
            optimizer.zero_grad()

        # compute
        data_in = data_in.to(device, dtype=nii_dconf.d_dtype)
        if args.model_forward_with_target:
            # if model.forward requires (input, target) as arguments
            # for example, for auto-encoder & autoregressive model
            if isinstance(data_tar, torch.Tensor):
                data_tar_tm = data_tar.to(device, dtype=nii_dconf.d_dtype)
                data_gen = pt_model(data_in, data_tar_tm)
            else:
                nii_display.f_print("--model-forward-with-target is set")
                nii_display.f_die("but no data_tar is not loaded")
        else:
            # normal case for model.forward(input)
            data_gen = pt_model(data_in)
        
        # compute loss and do back propagate
        loss_value = 0
        if isinstance(data_tar, torch.Tensor):
            data_tar = data_tar.to(device, dtype=nii_dconf.d_dtype)
            # there is no way to normalize the data inside loss
            # thus, do normalization here
            normed_target = pt_model.normalize_target(data_tar)
            loss = loss_wrapper.compute(data_gen, normed_target)
            loss_value = loss.item()            
            if optimizer is not None:
                loss.backward()
                optimizer.step()
                    
        # log down process information
        end_time = time.time()
        batchsize = len(data_info)
        for idx, data_seq_info in enumerate(data_info):
            monitor.log_loss(loss_value / batchsize, \
                             (end_time-start_time) / batchsize, \
                             data_seq_info, idx_orig.numpy()[idx], \
                             epoch_idx)
            # print infor for one sentence
            if args.verbose == 1:
                monitor.print_error_for_batch(data_idx*batchsize + idx,\
                                              idx_orig.numpy()[idx], \
                                              epoch_idx)
            # 
        # start the timer for a new batch
        start_time = time.time()
            
    # lopp done
    return
    

def f_train_wrapper(args, pt_model, loss_wrapper, device, \
                    optimizer_wrapper, \
                    train_dataset_wrapper, \
                    val_dataset_wrapper = None, \
                    checkpoint = None):
    """ 
    f_train_wrapper(args, pt_model, loss_wrapper, device, 
                    optimizer_wrapper
                    train_dataset_wrapper, val_dataset_wrapper = None,
                    check_point = None):
      A wrapper to run the training process

    Args:
       args:         argument information given by argpase
       pt_model:     pytorch model (torch.nn.Module)
       loss_wrapper: a wrapper over loss function
                     loss_wrapper.compute(generated, target) 
       device:       torch.device("cuda") or torch.device("cpu")

       optimizer_wrapper: 
           a wrapper over optimizer (defined in op_manager.py)
           optimizer_wrapper.optimizer is torch.optimizer
    
       train_dataset_wrapper: 
           a wrapper over training data set (data_io/default_data_io.py)
           train_dataset_wrapper.get_loader() returns torch.DataSetLoader
       
       val_dataset_wrapper: 
           a wrapper over validation data set (data_io/default_data_io.py)
           it can None.
       
       check_point:
           a check_point that stores every thing to resume training
    """        
    
    nii_display.f_print_w_date("Start model training")

    # get the optimizer
    optimizer_wrapper.print_info()
    optimizer = optimizer_wrapper.optimizer
    epoch_num = optimizer_wrapper.get_epoch_num()
    no_best_epoch_num = optimizer_wrapper.get_no_best_epoch_num()
    
    # get data loader for training set
    train_dataset_wrapper.print_info()
    train_data_loader = train_dataset_wrapper.get_loader()
    train_seq_num = train_dataset_wrapper.get_seq_num()

    # get the training process monitor
    monitor_trn = nii_monitor.Monitor(epoch_num, train_seq_num)

    # if validation data is provided, get data loader for val set
    if val_dataset_wrapper is not None:
        val_dataset_wrapper.print_info()
        val_data_loader = val_dataset_wrapper.get_loader()
        val_seq_num = val_dataset_wrapper.get_seq_num()
        monitor_val = nii_monitor.Monitor(epoch_num, val_seq_num)
    else:
        monitor_val = None

    # training log information
    train_log = ''

    # print the network
    pt_model.to(device, dtype=nii_dconf.d_dtype)
    f_model_show(pt_model)

    # resume training or initialize the model if necessary
    cp_names = CheckPointKey()
    if checkpoint is not None:
        if type(checkpoint) is dict:
            # checkpoint
            if cp_names.state_dict in checkpoint:
                pt_model.load_state_dict(checkpoint[cp_names.state_dict])
            if cp_names.optimizer in checkpoint:
                optimizer.load_state_dict(checkpoint[cp_names.optimizer])
            if cp_names.trnlog in checkpoint:
                monitor_trn.load_state_dic(checkpoint[cp_names.trnlog])
            if cp_names.vallog in checkpoint and monitor_val:
                monitor_val.load_state_dic(checkpoint[cp_names.vallog])
            if cp_names.info in checkpoint:
                train_log = checkpoint[cp_names.info]
            nii_display.f_print("Load check point and resume training")
        else:
            # only model status
            pt_model.load_state_dict(checkpoint)
            nii_display.f_print("Load pre-trained model")
            
    # other variables
    flag_early_stopped = False
    start_epoch = monitor_trn.get_epoch()
    epoch_num = monitor_trn.get_max_epoch()

    # print
    _ = nii_op_display_tk.print_log_head()
    nii_display.f_print_message(train_log, flush=True, end='')
        
        
    # loop over multiple epochs
    for epoch_idx in range(start_epoch, epoch_num):

        # training one epoch
        pt_model.train()
        f_run_one_epoch(args, pt_model, loss_wrapper, device, \
                        monitor_trn, train_data_loader, \
                        epoch_idx, optimizer)
        time_trn = monitor_trn.get_time(epoch_idx)
        loss_trn = monitor_trn.get_loss(epoch_idx)
        
        # if necessary, do validataion 
        if val_dataset_wrapper is not None:
            # set eval() if necessary 
            if args.eval_mode_for_validation:
                pt_model.eval()
            with torch.no_grad():
                f_run_one_epoch(args, pt_model, loss_wrapper, \
                                device, \
                                monitor_val, val_data_loader, \
                                epoch_idx, None)
            time_val = monitor_val.get_time(epoch_idx)
            loss_val = monitor_val.get_loss(epoch_idx)
        else:
            time_val, loss_val = 0, 0
                
        
        if val_dataset_wrapper is not None:
            flag_new_best = monitor_val.is_new_best()
        else:
            flag_new_best = True
            
        # print information
        train_log += nii_op_display_tk.print_train_info(epoch_idx, \
                                                        time_trn, \
                                                        loss_trn, \
                                                        time_val, \
                                                        loss_val, \
                                                        flag_new_best)
        # save the best model
        if flag_new_best:
            tmp_best_name = f_save_trained_name(args)
            torch.save(pt_model.state_dict(), tmp_best_name)
            
        # save intermediate model if necessary
        if not args.not_save_each_epoch:
            tmp_model_name = f_save_epoch_name(args, epoch_idx)
            if monitor_val is not None:
                tmp_val_log = monitor_val.get_state_dic()
            else:
                tmp_val_log = None
            # save
            tmp_dic = {
                cp_names.state_dict : pt_model.state_dict(),
                cp_names.info : train_log,
                cp_names.optimizer : optimizer.state_dict(),
                cp_names.trnlog : monitor_trn.get_state_dic(),
                cp_names.vallog : tmp_val_log
            }
            torch.save(tmp_dic, tmp_model_name)
            if args.verbose == 1:
                nii_display.f_eprint(str(datetime.datetime.now()))
                nii_display.f_eprint("Save {:s}".format(tmp_model_name),
                                     flush=True)
                
            
        # early stopping
        if monitor_val is not None and \
           monitor_val.should_early_stop(no_best_epoch_num):
            flag_early_stopped = True
            break
        
    # loop done        
    nii_op_display_tk.print_log_tail()
    if flag_early_stopped:
        nii_display.f_print("Training finished by early stopping")
    else:
        nii_display.f_print("Training finished")
    nii_display.f_print("Model is saved to", end = '')
    nii_display.f_print("{}".format(f_save_trained_name(args)))
    return

def f_inference_wrapper(args, pt_model, device, \
                        test_dataset_wrapper, checkpoint):
    """
    """
    test_data_loader = test_dataset_wrapper.get_loader()
    test_seq_num = test_dataset_wrapper.get_seq_num()
    test_dataset_wrapper.print_info()
    
    # print the network
    pt_model.to(device, dtype=nii_dconf.d_dtype)
    print(pt_model)
    
    cp_names = CheckPointKey()
    if type(checkpoint) is dict and cp_names.state_dict in checkpoint:
        pt_model.load_state_dict(checkpoint[cp_names.state_dict])
    else:
        pt_model.load_state_dict(checkpoint)

    
    pt_model.eval() 
    with torch.no_grad():
        for _, (data_in, data_tar, data_info, idx_orig) in \
            enumerate(test_data_loader):

            # send data to device
            data_in = data_in.to(device)
            if isinstance(data_tar, torch.Tensor):
                data_tar = data_tar.to(device, dtype=nii_dconf.d_dtype)
            
            
            # compute output
            start_time = time.time()
            if args.model_forward_with_target:
                # if model.forward requires (input, target) as arguments
                # for example, for auto-encoder
                data_gen = pt_model(data_in, data_tar)
            else:    
                data_gen = pt_model(data_in)
            data_gen = pt_model.denormalize_output(data_gen)
            time_cost = time.time() - start_time
            # average time for each sequence when batchsize > 1
            time_cost = time_cost / len(data_info)

            # save output (in case batchsize > 1, )
            data_gen_np = data_gen.to("cpu").numpy()
            for idx, seq_info in enumerate(data_info):
                _ = nii_op_display_tk.print_gen_info(seq_info, time_cost)
                test_dataset_wrapper.putitem(data_gen_np[idx:idx+1],\
                                             args.output_dir, \
                                             seq_info)
    # done
    return
            
if __name__ == "__main__":
    print("nn_manager")
