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
import core_scripts.nn_manager.nn_manager_tools as nii_nn_tools

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
                    data_loader, epoch_idx, optimizer = None, \
                    target_norm_method = None):
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
       target_norm_method: method to normalize target data
                           (by default, use pt_model.normalize_target)
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
                if args.model_forward_with_file_name:
                    data_gen = pt_model(data_in, data_tar_tm, data_info)
                else:
                    data_gen = pt_model(data_in, data_tar_tm)
            else:
                nii_display.f_print("--model-forward-with-target is set")
                nii_display.f_die("but data_tar is not loaded")
        else:
            if args.model_forward_with_file_name:
                # specifcal case when model.forward requires data_info
                data_gen = pt_model(data_in, data_info)
            else:
                # normal case for model.forward(input)
                data_gen = pt_model(data_in)
        
        # process target data
        if isinstance(data_tar, torch.Tensor):
            data_tar = data_tar.to(device, dtype=nii_dconf.d_dtype)
            # there is no way to normalize the data inside loss
            # thus, do normalization here
            if target_norm_method is None:
                normed_target = pt_model.normalize_target(data_tar)
            else:
                normed_target = target_norm_method(data_tar)
        else:
            normed_target = []

        # compute loss and do back propagate
        loss_vals = [0]

        # return the loss from loss_wrapper
        # loss_computed may be [[loss_1, loss_2, ...],[flag_1, flag_2,.]]
        #   which contain multiple loss and flags indicating whether
        #   the corresponding loss should be taken into consideration
        #   for early stopping
        # or 
        # loss_computed may be simply a tensor loss 
        loss_computed = loss_wrapper.compute(data_gen, normed_target)

        # To handle cases where there are multiple loss functions
        # when loss_comptued is [[loss_1, loss_2, ...],[flag_1, flag_2,.]]
        #   loss: sum of [loss_1, loss_2, ...], for backward()
        #   loss_vals: [loss_1.item(), loss_2.item() ..], for logging
        #   loss_flags: [True/False, ...], for logging, 
        #               whether loss_n is used for early stopping
        # when loss_computed is loss
        #   loss: loss
        #   los_vals: [loss.item()]
        #   loss_flags: [True]
        loss, loss_vals, loss_flags = nii_nn_tools.f_process_loss(
            loss_computed)

        # Back-propgation using the summed loss
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            
        # save the training process information to the monitor
        end_time = time.time()
        batchsize = len(data_info)
        for idx, data_seq_info in enumerate(data_info):
            # loss_value is supposed to be the average loss value
            # over samples in the the batch, thus, just loss_value
            # rather loss_value / batchsize
            monitor.log_loss(loss_vals, loss_flags, \
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

    # prepare for DataParallism if available
    # pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    if torch.cuda.device_count() > 1 and args.multi_gpu_data_parallel:
        flag_multi_device = True  
        nii_display.f_print("Use %d GPUs" % (torch.cuda.device_count()))
        # no way to call normtarget_f after pt_model is in DataParallel
        normtarget_f = pt_model.normalize_target
        pt_model = nn.DataParallel(pt_model)
    else:
        nii_display.f_print("Use single GPU: %s" % \
                            (torch.cuda.get_device_name(device)))
        flag_multi_device = False
        normtarget_f = None
    pt_model.to(device, dtype=nii_dconf.d_dtype)

    # print the network
    f_model_show(pt_model)

    # resume training or initialize the model if necessary
    cp_names = CheckPointKey()
    if checkpoint is not None:
        if type(checkpoint) is dict:
            # checkpoint

            # load model parameter and optimizer state
            if cp_names.state_dict in checkpoint:
                # wrap the state_dic in f_state_dict_wrapper 
                # in case the model is saved when DataParallel is on
                pt_model.load_state_dict(
                    nii_nn_tools.f_state_dict_wrapper(
                        checkpoint[cp_names.state_dict], 
                        flag_multi_device))

            # load optimizer state
            if cp_names.optimizer in checkpoint:
                optimizer.load_state_dict(checkpoint[cp_names.optimizer])
            
            # optionally, load training history
            if not args.ignore_training_history_in_trained_model:
                #nii_display.f_print("Load ")
                if cp_names.trnlog in checkpoint:
                    monitor_trn.load_state_dic(
                        checkpoint[cp_names.trnlog])
                if cp_names.vallog in checkpoint and monitor_val:
                    monitor_val.load_state_dic(
                        checkpoint[cp_names.vallog])
                if cp_names.info in checkpoint:
                    train_log = checkpoint[cp_names.info]
                nii_display.f_print("Load check point, resume training")
            else:
                nii_display.f_print("Load pretrained model and optimizer")
        else:
            # only model status
            #pt_model.load_state_dict(checkpoint)
            pt_model.load_state_dict(
                nii_nn_tools.f_state_dict_wrapper(
                    checkpoint, flag_multi_device))
            nii_display.f_print("Load pretrained model")
            
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
                        epoch_idx, optimizer, normtarget_f)
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
                                epoch_idx, None, normtarget_f)
            time_val = monitor_val.get_time(epoch_idx)
            loss_val = monitor_val.get_loss(epoch_idx)
        else:
            time_val, loss_val = 0, 0
                
        
        if val_dataset_wrapper is not None:
            flag_new_best = monitor_val.is_new_best()
        else:
            flag_new_best = True
            
        # print information
        train_log += nii_op_display_tk.print_train_info(
            epoch_idx, time_trn, loss_trn, time_val, loss_val, 
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
    
    if torch.cuda.device_count() > 1 and args.multi_gpu_data_parallel:
        nii_display.f_print(
            "DataParallel for inference is not implemented", 'warning')
    nii_display.f_print("Use single GPU: %s" % \
                        (torch.cuda.get_device_name(device)))

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
                if args.model_forward_with_file_name:
                    data_gen = pt_model(data_in, data_tar, data_info)
                else:
                    data_gen = pt_model(data_in, data_tar)
            else:    
                if args.model_forward_with_file_name:
                    data_gen = pt_model(data_in, data_info)
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
    # 
    nii_display.f_print("Generated data to %s" % (args.output_dir))
    # done
    return
            
if __name__ == "__main__":
    print("nn_manager")
