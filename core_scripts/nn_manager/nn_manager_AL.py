#!/usr/bin/env python
"""
nn_manager_AL

A simple wrapper to run the training for active learning

Note: 
1. The mode to continue to training does not guanrantee exactly 
the same result because selection is based on random sampling. The
random seed for data selection differs.

"""
from __future__ import print_function

import os
import time
import datetime
import numpy as np
import copy
import re

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
__g_info_separator = ':'
__g_name_separator = ';'
__g_type_tags = ['add', 'remove']

def __print_new_sample_list(cycle_idx, dataset_wrapper, data_idx):
    """ print information on the newly added data 
    """
    mes = 'Active learning cycle {:d}, add samples'.format(cycle_idx)
    mes += __g_info_separator + ' '
    mes += __g_name_separator.join(dataset_wrapper.get_seq_info())
    mes += __g_info_separator + ' '
    mes += __g_name_separator.join([str(x) for x in data_idx])
    #mes += '\nNumber of samples: {:d}'.format(len(data_idx))
    nii_display.f_eprint(mes)
    return mes

def __print_excl_sample_list(cycle_idx, dataset_wrapper, data_idx):
    """ print information on the newly removed data 
    """
    mes = 'Before learning cycle {:d}, remove'.format(cycle_idx)
    mes += __g_info_separator + ' '
    mes += __g_name_separator.join(dataset_wrapper.get_seq_info())
    mes += __g_info_separator + ' '
    mes += __g_name_separator.join([str(x) for x in data_idx])
    #mes += '\nNumber of removed samples: {:d}'.format(len(data_idx))
    nii_display.f_eprint(mes)
    return mes

def __save_sample_list_buf(list_buff, cache_path):
    with open(cache_path, 'w') as file_ptr:
        for data_str in list_buff:
            file_ptr.write(data_str + '\n')
    return

def __cache_name(path, cycle_idx):
    return '{:s}_{:03d}.txt'.format(path, cycle_idx)
   
def __parse_sample_list(mes):
    """ 
    Active learning cycle K, add samples: file1, file2, file3
    ->
    K, add, [file1, file2, file3]
    """
    # cycle index
    cycle_id = re.findall("[0-9]+", mes.split(__g_info_separator)[0])
    cycle_id = int(cycle_id[0])
    
    # type of method
    if re.findall(__g_type_tags[1], mes.split(__g_info_separator)[0]):
        tag = __g_type_tags[1]
    else:
        tag = __g_type_tags[0]
    
    # assume that : is not included in the file name
    filepart = mes.split(__g_info_separator)[2]
    # return the sample list
    return cycle_id,  tag, \
        [int(x.rstrip().lstrip()) for x in filepart.split(__g_name_separator)]

def __load_cached_data_list_file(cache_path):
    with open(cache_path, 'r') as file_ptr:
        output = [__parse_sample_list(x) for x in file_ptr]
    return output
    

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

def _f_copy_subset(dataset_wrapper, data_idx):
    """ return a deepcopy of dataset that contains data specified by data_idx
    """
    # create data that contains selected data only
    #  the database only contains data index, so it is fast to do deepcopy  
    tmp_data_wrapper = copy.deepcopy(dataset_wrapper)
    tmp_data_wrapper.manage_data(data_idx, 'keep')
    return tmp_data_wrapper

def _f_add_data(pool_dataset_wrapper, train_dataset_wrapper, data_idx, args):
    """
    """
    # create a copy of the data to be selected from the pool
    #  the database only contains data index, so it is fast to do deepcopy  
    tmp_data_wrapper = _f_copy_subset(pool_dataset_wrapper, data_idx)

    # Delete data from original pool if we sample without replacement
    if not args.active_learning_with_replacement:
        pool_dataset_wrapper.manage_data(data_idx, 'delete')
        #pool_data_loader = pool_dataset_wrapper.get_loader()

    # 
    if args.active_learning_use_new_data_only:
        # only augmented data
        nii_display.f_die("Not implemented yet")
    else:
        # base dataset + augmented data
        train_dataset_wrapper.add_dataset(tmp_data_wrapper)
    return


def _f_remove_data(pool_dataset_wrapper, data_idx, args):
    """
    """
    pool_dataset_wrapper.manage_data(data_idx, 'delete')
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
      
    ##
    # Configurations
    ##
    # total number of epoch = epoch per cycle * number of AL cycles
    total_epoch_num = optimizer_wrapper.get_epoch_num()
    num_al_cycle = np.abs(args.active_learning_cycle_num)
    if num_al_cycle == 0:
        nii_display.f_die("Number of active learning cycles must be > 0")
    epoch_per_cycle = total_epoch_num // num_al_cycle
    
    # set the number of samples take per cycle
    num_sample_al_cycle = args.active_learning_new_sample_per_cycle
    # if not set, take batch-size of samples per cycle
    if num_sample_al_cycle < 1:
        num_sample_al_cycle = args.batch_size
    #nii_display.f_print("Add {:d} new samples per cycle".format(
    #    num_sample_al_cycle))

    # patience for early stopping on development set
    no_best_epoch_num = optimizer_wrapper.get_no_best_epoch_num()
    
    ##
    # data loader, optimizer, model, ...
    ## 
    # get the optimizer
    if not args.active_learning_pre_train_epoch_num:
        # this information should have been printed during pre-training
        optimizer_wrapper.print_info()
    optimizer = optimizer_wrapper.optimizer
    lr_scheduler = optimizer_wrapper.lr_scheduler

    # get data loader for seed training set
    if not args.active_learning_pre_train_epoch_num:
        train_dataset_wrapper.print_info()
    train_data_loader = train_dataset_wrapper.get_loader()
    train_seq_num = train_dataset_wrapper.get_seq_num()
    
    # get pool data set for active learning
    pool_dataset_wrapper.print_info()
    pool_data_loader = pool_dataset_wrapper.get_loader()
    pool_seq_num = pool_dataset_wrapper.get_seq_num()

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

    ##
    # misc
    ##
    # print the network
    if not args.active_learning_pre_train_epoch_num:
        nii_nn_tools.f_model_show(pt_model)
        nii_nn_tools.f_loss_show(loss_wrapper)

    # key names, used when saving *.epoch.pt
    cp_names = nii_nn_manage_conf.CheckPointKey()
    
    # training log information
    train_log = ''

    # buffer for selected data list
    al_mes_buff = []

    ###############################
    ## Resume training if necessary
    ###############################
    ##
    # load epoch*.pt (which contains no infor on previously selected data)
    ## 
    train_log = nii_nn_tools.f_load_checkpoint(
        checkpoint, args, flag_multi_device, pt_model, 
        optimizer, monitor_trn, monitor_val, lr_scheduler)
    
    ##
    # load selected or removed utterances in previous cycles,
    ##
    # index of the starting cycle (default 0)
    start_cycle = 0
    if len(args.active_learning_cache_dataname_path) \
       and os.path.isfile(args.active_learning_cache_dataname_path):
        nii_display.f_print("Load cache of selected (removed data)")
        
        # Load from file
        cached_data_status = __load_cached_data_list_file(
            args.active_learning_cache_dataname_path)
        
        # For each cycle, update the pool and training set
        for entry in cached_data_status:
            
            # retrieve the log
            cycle_id, method_type, data_idx = entry[0], entry[1], entry[2]
            
            if method_type == __g_type_tags[1]:
                # for removing
                # print the information
                mes = __print_excl_sample_list(
                    cycle_id, _f_copy_subset(pool_dataset_wrapper, data_idx), 
                    data_idx)
                # remove previously removed data from pool
                _f_remove_data(pool_dataset_wrapper, data_idx, args)
            else:
                # for selected data (either active or passive)
                # print the information
                mes = __print_new_sample_list(
                    cycle_id, _f_copy_subset(pool_dataset_wrapper, data_idx),
                    data_idx)
                # add previously selected from the pool to the training set
                _f_add_data(pool_dataset_wrapper, train_dataset_wrapper, 
                            data_idx, args)
            #
            pool_data_loader = pool_dataset_wrapper.get_loader()
            train_data_loader = train_dataset_wrapper.get_loader()
            # 
            al_mes_buff.append(mes)

        if len(cached_data_status):
            # data selectio and removing should have been done in cycle_id
            # thus, we start from the next cycle
            start_cycle = cycle_id + 1
    else:
        pass

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
    # this counts the epoch number across different cycles
    start_epoch = monitor_trn.get_epoch()
    epoch_counter = start_epoch
    # get the total number of epochs to run
    total_epoch_num = monitor_trn.get_max_epoch()    
    # a buf to store the path of trained models per cycle
    saved_model_path_buf = []
    
    # print active learining general information
    __print_AL_info(num_al_cycle, epoch_per_cycle, num_sample_al_cycle, args)
    # print training log (if available from resumed checkpoint)
    _ = nii_op_display_tk.print_log_head()
    nii_display.f_print_message(train_log, flush=True, end='')

    # sanity check
    if start_epoch // epoch_per_cycle != start_cycle:
        nii_display.f_print("Training cycle in {:s} != that in {:s}".format(
            args.trained_model, args.active_learning_cache_dataname_path))
        nii_display.f_print(" {:d} // {:d} != {:d}".format(
            start_epoch, epoch_per_cycle, start_cycle))
        nii_display.f_die("Fail to resume training")
    #
    # currently, we can only restat from the 1st epoch in each active learning
    # cycle. Note that, monitor_trn.get_epoch() returns the current epoch idx
    if start_epoch > 0 and start_epoch % epoch_per_cycle != 0:
        mes = "The checkpoint is not the last epoch in one cycle"
        nii_display.f_print(mes)
        nii_display.f_die("Fail to resume training")

    # loop over cycles
    for cycle_idx in range(start_cycle, num_al_cycle):
        
        # Pool data has been used up. Training ends.
        if pool_dataset_wrapper.get_seq_num() < 1:
            break

        ########
        # select the samples
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
        #
        # III. Pool-based, but to exclude data first
        #     al_exclude_data(train_data_loader,pool_data_loader, 
        #                                    num_sample_al_cycle)
        #     Exclude samples from the pool set
        #     If provided, this function will be called first before executing
        #     Pool-based II and I
        # 
        # save current model flag
        tmp_train_flag = True if pt_model.training else False
        if args.active_learning_train_model_for_retrieval:
            pt_model.train()
        else:
            pt_model.eval()

        # exclude data if necesary
        if hasattr(pt_model, 'al_exclude_data'):
            # select least useful data
            data_idx = pt_model.al_exclude_data(
                pool_data_loader, num_sample_al_cycle)
            # convert data index to int
            data_idx = [int(x) for x in data_idx]

            # print datat to be excluded
            mes = __print_excl_sample_list(
                cycle_idx, _f_copy_subset(pool_dataset_wrapper, data_idx), 
                data_idx)
            al_mes_buff.append(mes)
            
            # remove the pool
            _f_remove_data(pool_dataset_wrapper, data_idx, args)
            pool_data_loader = pool_dataset_wrapper.get_loader()
            
            
        # retrieve data from the pool to training set
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
        ########
        # _f_add_data alters pool_dataset_wrapper
        # thus, we print the list based on pool_dataset before _f_add_data
        mes = __print_new_sample_list(
            cycle_idx, _f_copy_subset(pool_dataset_wrapper, data_idx), data_idx)
        al_mes_buff.append(mes)
        
        _f_add_data(pool_dataset_wrapper, train_dataset_wrapper, data_idx, args)
        
        # prepare for training
        # get the data loader from the new training and pool sets
        pool_data_loader = pool_dataset_wrapper.get_loader()
        train_data_loader = train_dataset_wrapper.get_loader()
        # number of samples in current training and pool sets
        train_seq_num = train_dataset_wrapper.get_seq_num()
        pool_seq_num = pool_dataset_wrapper.get_seq_num()
        # because the number of training data changes in each cycle, we need to
        # create a temporary monitor for training
        tmp_monitor_trn = nii_monitor.Monitor(epoch_per_cycle, train_seq_num)
        
        ########
        # training using the new training set
        # epoch_counter: a global counter of training epoch, across all cycles
        # tmp_start_epoch: index of starting epoch within one cycle
        # tmp_epoch_idx: index of epoch within one cycle
        tmp_start_epoch = epoch_counter % epoch_per_cycle
        for tmp_epoch_idx in range(tmp_start_epoch, epoch_per_cycle):
            

            # If the model has a member for g_epoch_idx
            # save the index
            if hasattr(pt_model, 'g_epoch_idx'): 
                pt_model.g_epoch_idx = epoch_counter

            # If the model has a member for g_epoch_idx
            # save the index
            # cycle index should be updated after selecting the data
            if hasattr(pt_model, 'g_cycle_idx'): 
                pt_model.g_cycle_idx = cycle_idx

            # training one epoch
            pt_model.train()
            if hasattr(pt_model, 'flag_validation'):
                pt_model.flag_validation = False
            nii_nn_manager_base.f_run_one_epoch(
                args, pt_model, loss_wrapper, device, \
                tmp_monitor_trn, train_data_loader, \
                tmp_epoch_idx, optimizer, normtarget_f)
            # get the time and loss for this epoch
            time_trn = tmp_monitor_trn.get_time(tmp_epoch_idx)
            loss_trn = tmp_monitor_trn.get_loss(tmp_epoch_idx)
        
            # if necessary, forward pass on development set
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
                        epoch_counter, None, normtarget_f)
                time_val = monitor_val.get_time(epoch_counter)
                loss_val = monitor_val.get_loss(epoch_counter)
            
                # update lr rate scheduler if necessary
                if lr_scheduler.f_valid():
                    lr_scheduler.f_step(loss_val)
            else:
                time_val = monitor_val.get_time(epoch_counter)
                loss_val = monitor_val.get_loss(epoch_counter)
                #time_val, loss_val = 0, 0
            
            # wether this is the new best trained epoch?
            if val_dataset_wrapper is not None:
                flag_new_best = monitor_val.is_new_best()
            else:
                flag_new_best = True
            
            # print information
            info_mes = [optimizer_wrapper.get_lr_info(), 
                        __print_cycle(cycle_idx, train_seq_num, pool_seq_num)]
            train_log += nii_op_display_tk.print_train_info(
                tmp_epoch_idx, time_trn, loss_trn, time_val, loss_val, 
                flag_new_best, ', '.join([x for x in info_mes if x]))

            # save the best model if necessary
            if flag_new_best or args.force_save_lite_trained_network_per_epoch:
                tmp_best_name = nii_nn_tools.f_save_trained_name(args)
                torch.save(pt_model.state_dict(), tmp_best_name)
            
            # save intermediate model if necessary 
            # we only say the last epoch in each cycle
            if not args.not_save_each_epoch \
               and tmp_epoch_idx == (epoch_per_cycle - 1):
                
                # we save the global epoch counter into the checkpoint
                monitor_trn.log_epoch(epoch_counter)
                
                # name of the checkpoint
                tmp_model_name = nii_nn_tools.f_save_epoch_name(
                    args, cycle_idx, '_epoch_{:03d}'.format(tmp_epoch_idx),
                    '_al_cycle')
                
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

            # 
            epoch_counter += 1
                
        # loop done for epoch per cycle
        # always save the trained model for each cycle
        suffix = '_al_cycle_{:03d}'.format(cycle_idx)
        tmp_best_name = nii_nn_tools.f_save_trained_name(args, suffix)
        torch.save(pt_model.state_dict(), tmp_best_name)
        saved_model_path_buf.append(tmp_best_name)
        
        # save selected data for each cycle
        __save_sample_list_buf(
            al_mes_buff, 
            __cache_name(args.active_learning_cache_dataname_save, cycle_idx))

    # loop for AL cycle
    nii_op_display_tk.print_log_tail()
    nii_display.f_print("Training finished")
    nii_display.f_print("Models from each cycle are saved to:")
    for path in saved_model_path_buf:
        nii_display.f_print("{}".format(path), 'normal')

    return
            
if __name__ == "__main__":
    print("nn_manager_AL")
