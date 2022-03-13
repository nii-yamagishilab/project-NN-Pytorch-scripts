#!/usr/bin/env python
"""
nn_manager_AL

A simple wrapper to run the training for active learning

"""
from __future__ import print_function

import time
import datetime
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.display as nii_display
import core_scripts.other_tools.str_tools as nii_str_tk
import core_scripts.op_manager.op_process_monitor as nii_monitor
import core_scripts.op_manager.op_display_tools as nii_op_display_tk
import core_scripts.nn_manager.nn_manager_tools as nii_nn_tools
import core_scripts.nn_manager.nn_manager_conf as nii_nn_manage_conf
import core_scripts.nn_manager.nn_manager as nii_nn_manager_base

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

#############################################################
def __print_new_sample_list(cycle_idx, data_idx, dataset_wrapper):
    """ print information on the newly added data 
    """
    mes = 'Active learning cycle {:d}, add samples: '.format(cycle_idx)
    mes += ', '.join(dataset_wrapper.get_seq_list())
    mes += '\nNumber of samples: {:d}'.format(len(set(data_idx)))
    nii_display.f_eprint(mes)
    return
   
def __print_cycle(cycle_idx, train_s, pool_s):
    """ print information added to the error log
    """
    return "AL cycle {:d}, {:d}, {:d}".format(cycle_idx, train_s, pool_s)

def __print_AL_info(num_al_cycle, epoch_per_cycle, num_sample_al_cycle, args):
    """ print head information to summarize the AL settings
    """
    mes = "\nActive learning (pool-based) settings:"
    nii_display.f_print(mes)
    
    mes = 'Number of active learning cycle: {:d}'.format(num_al_cycle)
    mes += '\nNumber of epochs per cycle:      {:d}'.format(epoch_per_cycle)
    mes += '\nNumber of new samples per cycle: {:d}'.format(num_sample_al_cycle)
    if args.active_learning_use_new_data_only:
        mes += '\nUse retrieved data for fine-tuning model'
    else:
        mes += '\nUse seed + retrieved data for model training'
    if args.active_learning_with_replacement:
        mes += '\nRetrieve data w/ replacement'
    else:
        mes += '\nRetrieve data w/o replacement'
    mes += '\n'
    nii_display.f_print(mes, 'normal')
    return 
    

def f_train_wrapper(args, pt_model, loss_wrapper, device, \
                    optimizer_wrapper, \
                    train_dataset_wrapper, \
                    pool_dataset_wrapper, \
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
           a wrapper over training data set
           train_dataset_wrapper.get_loader() returns torch.DataSetLoader

       pool_dataset_wrapper: 
           a wrapper over pool data set for AL
           train_dataset_wrapper.get_loader() returns torch.DataSetLoader
       
       val_dataset_wrapper: 
           a wrapper over validation data set (data_io/default_data_io.py)
           it can None.
       
       check_point:
           a check_point that stores every thing to resume training
    """        
    
    nii_display.f_print_w_date("Start model training")

    ##############
    ## Preparation
    ##############

    # get the optimizer
    if not args.active_learning_pre_train_epoch_num:
        # this information should have been printed during pre-training
        optimizer_wrapper.print_info()
    optimizer = optimizer_wrapper.optimizer
    lr_scheduler = optimizer_wrapper.lr_scheduler
    
    # total number of epoch = epoch per cycle * number of AL cycles
    total_epoch_num = optimizer_wrapper.get_epoch_num()
    num_al_cycle = np.abs(args.active_learning_cycle_num)
    if num_al_cycle == 0:
        nii_display.f_die("Number of active learning cycles must be > 0")
    epoch_per_cycle = total_epoch_num // num_al_cycle
    
    no_best_epoch_num = optimizer_wrapper.get_no_best_epoch_num()
    
    # get data loader for seed training set
    if not args.active_learning_pre_train_epoch_num:
        train_dataset_wrapper.print_info()
    train_data_loader = train_dataset_wrapper.get_loader()
    train_seq_num = train_dataset_wrapper.get_seq_num()
    
    # get pool data set for active learning
    pool_dataset_wrapper.print_info()
    pool_data_loader = pool_dataset_wrapper.get_loader()
    pool_seq_num = pool_dataset_wrapper.get_seq_num()

    # set the number of samples take per cycle
    num_sample_al_cycle = args.active_learning_new_sample_per_cycle
    # if not set, take batch-size of samples per cycle
    if num_sample_al_cycle < 1:
        num_sample_al_cycle = args.batch_size
    #nii_display.f_print("Add {:d} new samples per cycle".format(
    #    num_sample_al_cycle))

    # get the training process monitor
    monitor_trn = nii_monitor.Monitor(total_epoch_num, train_seq_num)

    # if validation data is provided, get data loader for val set
    if val_dataset_wrapper is not None:
        if not args.active_learning_pre_train_epoch_num:
            val_dataset_wrapper.print_info()
        val_data_loader = val_dataset_wrapper.get_loader()
        val_seq_num = val_dataset_wrapper.get_seq_num()
        monitor_val = nii_monitor.Monitor(total_epoch_num, val_seq_num)
    else:
        monitor_val = None
        
    # training log information
    train_log = ''

    # prepare for DataParallism if available
    # pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    if torch.cuda.device_count() > 1 and args.multi_gpu_data_parallel:
        nii_display.f_die("Not implemented for multiple GPU")
    else:
        nii_display.f_print("\nUse single GPU: %s\n" % \
                            (torch.cuda.get_device_name(device)))
        flag_multi_device = False
        normtarget_f = None
    pt_model.to(device, dtype=nii_dconf.d_dtype)

    # print the network
    if not args.active_learning_pre_train_epoch_num:
        nii_nn_tools.f_model_show(pt_model)
        nii_nn_tools.f_loss_show(loss_wrapper)

    # key names, used when saving *.epoch.pt
    cp_names = nii_nn_manage_conf.CheckPointKey()

    ###############################
    ## Resume training if necessary
    ###############################
    # resume training or initialize the model if necessary
    train_log = nii_nn_tools.f_load_checkpoint(
        checkpoint, args, flag_multi_device, pt_model, 
        optimizer, monitor_trn, monitor_val, lr_scheduler)
    
    ######################
    ### User defined setup 
    ######################
    if hasattr(pt_model, "other_setups"):
        nii_display.f_print("Conduct User-defined setup")
        pt_model.other_setups()
    
    # This should be merged with other_setups
    if hasattr(pt_model, "g_pretrained_model_path") and \
       hasattr(pt_model, "g_pretrained_model_prefix"):
        nii_display.f_print("Load pret-rained models as part of this mode")
        nii_nn_tools.f_load_pretrained_model_partially(
            pt_model, pt_model.g_pretrained_model_path, 
            pt_model.g_pretrained_model_prefix)
        
    ##############################
    ### Start active learning loop
    ##############################
    # other variables
    # initialize flag to save state of early stopping
    flag_early_stopped = False
    # initialize the starting epoch number 
    start_epoch = monitor_trn.get_epoch()
    # get the total number of epochs to run
    total_epoch_num = monitor_trn.get_max_epoch()    
    # a buf to store the path of trained models per cycle
    saved_model_path_buf = []

    # print information
    __print_AL_info(num_al_cycle, epoch_per_cycle, num_sample_al_cycle, args)
    # print training log
    _ = nii_op_display_tk.print_log_head()
    nii_display.f_print_message(train_log, flush=True, end='')
    
    # loop over cycles
    for cycle_idx in range(num_al_cycle):
        
        if pool_dataset_wrapper.get_seq_num() < 1:
            #nii_display.f_print("Pool data has been used up. Training ends.")
            break

        ########
        # select the best samples
        ########
        # There are many methods to select samples
        #  we require pt_model to define one of the method
        #  I. Pool-based, no knowedge on seed data:
        #     al_retrieve_data(pool_data_loader, 
        #                      num_sample_al_cycle)
        #     Only use model to score each data in pool_data_loader
        # 
        # II. Pool-based, w/ knowedge on seed data
        #     al_retrieve_data_knowing_train(train_data_loader,
        #                                    pool_data_loader, 
        #                                    num_sample_al_cycle)
        #     Select sample from pool given knowlege of train seed data
        
        # save current model flag
        tmp_train_flag = True if pt_model.training else False
        if args.active_learning_train_model_for_retrieval:
            pt_model.train()
        else:
            pt_model.eval()

        # retrieve data
        if hasattr(pt_model, 'al_retrieve_data'):
            data_idx = pt_model.al_retrieve_data(
                pool_data_loader, num_sample_al_cycle)
        elif hasattr(pt_model, 'al_retrieve_data_knowing_train'):
            data_idx = pt_model.al_retrieve_data_knowing_train(
                train_data_loader, pool_data_loader, num_sample_al_cycle)
        else:
            nii_display.f_die("model must define al_retrieve_data")
            
        # convert data index to int
        data_idx = [int(x) for x in data_idx]

        # set flag back
        if tmp_train_flag:
            pt_model.train()
        else:
            pt_model.eval()
        ########
        # Create Dataset wrapper for the new data add new data to train set
        tmp_data_wrapper = copy.deepcopy(pool_dataset_wrapper)
        tmp_data_wrapper.manage_data(data_idx, 'keep')
        
        # Delete data if we sample without replacement
        if not args.active_learning_with_replacement:
            pool_dataset_wrapper.manage_data(data_idx, 'delete')
            pool_data_loader = pool_dataset_wrapper.get_loader()

        if args.active_learning_use_new_data_only:
            # only augmented data
            nii_display.f_die("Not implemented yet")
        else:
            # base dataset + augmented data
            train_dataset_wrapper.add_dataset(tmp_data_wrapper)
            
        train_data_loader = train_dataset_wrapper.get_loader()
        train_seq_num = train_dataset_wrapper.get_seq_num()
        pool_seq_num = pool_dataset_wrapper.get_seq_num()
        tmp_monitor_trn = nii_monitor.Monitor(epoch_per_cycle, train_seq_num)
        
        
        if args.verbose == 1:
            __print_new_sample_list(cycle_idx, data_idx, tmp_data_wrapper)

        ########
        # training using the new training set
        for epoch_idx in range(start_epoch, epoch_per_cycle):

            # training one epoch
            pt_model.train()
            if hasattr(pt_model, 'flag_validation'):
                pt_model.flag_validation = False
            nii_nn_manager_base.f_run_one_epoch(
                args, pt_model, loss_wrapper, device, \
                tmp_monitor_trn, train_data_loader, \
                epoch_idx, optimizer, normtarget_f)
            time_trn = tmp_monitor_trn.get_time(epoch_idx)
            loss_trn = tmp_monitor_trn.get_loss(epoch_idx)
        
            # if necessary, do validataion 
            if val_dataset_wrapper is not None:
                # set eval() if necessary 
                if args.eval_mode_for_validation:
                    pt_model.eval()
                if hasattr(pt_model, 'flag_validation'):
                    pt_model.flag_validation = True

                with torch.no_grad():
                    nii_nn_manager_base.f_run_one_epoch(
                        args, pt_model, loss_wrapper, \
                        device, monitor_val, val_data_loader, \
                        epoch_idx, None, normtarget_f)
                time_val = monitor_val.get_time(epoch_idx)
                loss_val = monitor_val.get_loss(epoch_idx)
            
                # update lr rate scheduler if necessary
                if lr_scheduler.f_valid():
                    lr_scheduler.f_step(loss_val)
            else:
                time_val = monitor_val.get_time(epoch_idx)
                loss_val = monitor_val.get_loss(epoch_idx)
                #time_val, loss_val = 0, 0
                
        
            if val_dataset_wrapper is not None:
                flag_new_best = monitor_val.is_new_best()
            else:
                flag_new_best = True
            
            # print information
            info_mes = [optimizer_wrapper.get_lr_info(), 
                        __print_cycle(cycle_idx, train_seq_num, pool_seq_num)]
            train_log += nii_op_display_tk.print_train_info(
                epoch_idx, time_trn, loss_trn, time_val, loss_val, 
                flag_new_best, ', '.join([x for x in info_mes if x]))

            # save the best model
            if flag_new_best or args.force_save_lite_trained_network_per_epoch:
                tmp_best_name = nii_nn_tools.f_save_trained_name(args)
                torch.save(pt_model.state_dict(), tmp_best_name)
            
            # save intermediate model if necessary
            if not args.not_save_each_epoch:
                tmp_model_name = nii_nn_tools.f_save_epoch_name(args, epoch_idx)
                
                if monitor_val is not None:
                    tmp_val_log = monitor_val.get_state_dic()
                else:
                    tmp_val_log = None
                
                if lr_scheduler.f_valid():
                    lr_scheduler_state = lr_scheduler.f_state_dict()
                else:
                    lr_scheduler_state = None

                # save
                tmp_dic = {
                    cp_names.state_dict : pt_model.state_dict(),
                    cp_names.info : train_log,
                    cp_names.optimizer : optimizer.state_dict(),
                    cp_names.trnlog : monitor_trn.get_state_dic(),
                    cp_names.vallog : tmp_val_log,
                    cp_names.lr_scheduler : lr_scheduler_state
                }
                torch.save(tmp_dic, tmp_model_name)
                if args.verbose == 1:
                    nii_display.f_eprint(str(datetime.datetime.now()))
                    nii_display.f_eprint("Save {:s}".format(tmp_model_name),
                                         flush=True)

            # check whether early stopping
            if lr_scheduler.f_allow_early_stopping() and \
               monitor_val is not None and \
               monitor_val.should_early_stop(no_best_epoch_num):
                flag_early_stopped = True
                break

        # loop done for epoch per cycle
        # always save the trained model for each cycle
        suffix = '_al_cycle_{:03d}'.format(cycle_idx)
        tmp_best_name = nii_nn_tools.f_save_trained_name(args, suffix)
        torch.save(pt_model.state_dict(), tmp_best_name)
        saved_model_path_buf.append(tmp_best_name)

    # loop for AL cycle
    nii_op_display_tk.print_log_tail()
    nii_display.f_print("Training finished")
    nii_display.f_print("Models from each cycle are saved to:")
    for path in saved_model_path_buf:
        nii_display.f_print("{}".format(path), 'normal')
    return
            
if __name__ == "__main__":
    print("nn_manager_AL")
