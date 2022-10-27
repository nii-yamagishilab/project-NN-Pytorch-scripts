#!/usr/bin/env python
"""
main.py for project-NN-pytorch/projects

The training/inference process wrapper for active learning.
The base is on main_mergedataset.py

Requires model.py and config.py (config_merge_datasets.py)

Usage: $: python main.py [options]
"""
from __future__ import absolute_import
import os
import sys
import copy
import torch
import importlib

import core_scripts.other_tools.display as nii_warn
import core_scripts.data_io.default_data_io as nii_default_dset
import core_scripts.data_io.customize_dataset as nii_dset
import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.config_parse.config_parse as nii_config_parse
import core_scripts.config_parse.arg_parse as nii_arg_parse
import core_scripts.op_manager.op_manager as nii_op_wrapper
import core_scripts.nn_manager.nn_manager_AL as nii_nn_wrapper
import core_scripts.nn_manager.nn_manager as nii_nn_wrapper_base
import core_scripts.startup_config as nii_startup

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"


def main():
    """ main(): the default wrapper for training and inference process
    Please prepare config.py and model.py
    """
    # arguments initialization
    args = nii_arg_parse.f_args_parsed()

    # 
    nii_warn.f_print_w_date("Start program", level='h')
    nii_warn.f_print("Load module: %s" % (args.module_config))
    nii_warn.f_print("Load module: %s" % (args.module_model))
    prj_conf = importlib.import_module(args.module_config)
    prj_model = importlib.import_module(args.module_model)

    # initialization
    nii_startup.set_random_seed(args.seed, args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # prepare data io    
    if not args.inference:
        params = {'batch_size':  args.batch_size,
                  'shuffle':  args.shuffle,
                  'num_workers': args.num_workers, 
                  'sampler': args.sampler}
        
        in_trans_fns = prj_conf.input_trans_fns \
                       if hasattr(prj_conf, 'input_trans_fns') else None
        out_trans_fns = prj_conf.output_trans_fns \
                        if hasattr(prj_conf, 'output_trans_fns') else None
        
        # Load file list and create data loader
        trn_lst = prj_conf.trn_list
        trn_set = nii_dset.NII_MergeDataSetLoader(
            prj_conf.trn_set_name, \
            trn_lst,
            prj_conf.input_dirs, \
            prj_conf.input_exts, \
            prj_conf.input_dims, \
            prj_conf.input_reso, \
            prj_conf.input_norm, \
            prj_conf.output_dirs, \
            prj_conf.output_exts, \
            prj_conf.output_dims, \
            prj_conf.output_reso, \
            prj_conf.output_norm, \
            './', 
            params = params,
            truncate_seq = prj_conf.truncate_seq, 
            min_seq_len = prj_conf.minimum_len,
            save_mean_std = True,
            wav_samp_rate = prj_conf.wav_samp_rate,
            way_to_merge = args.way_to_merge_datasets,
            global_arg = args,
            dset_config = prj_conf,
            input_augment_funcs = in_trans_fns,
            output_augment_funcs = out_trans_fns)


        # Load data pool and create data loader
        pool_lst = prj_conf.al_pool_list
        pool_set = nii_dset.NII_MergeDataSetLoader(
            prj_conf.al_pool_set_name, \
            pool_lst,
            prj_conf.al_pool_in_dirs, \
            prj_conf.input_exts, \
            prj_conf.input_dims, \
            prj_conf.input_reso, \
            prj_conf.input_norm, \
            prj_conf.al_pool_out_dirs, \
            prj_conf.output_exts, \
            prj_conf.output_dims, \
            prj_conf.output_reso, \
            prj_conf.output_norm, \
            './', 
            params = params,
            truncate_seq = prj_conf.truncate_seq, 
            min_seq_len = prj_conf.minimum_len,
            save_mean_std = True,
            wav_samp_rate = prj_conf.wav_samp_rate,
            way_to_merge = args.way_to_merge_datasets,
            global_arg = args,
            dset_config = prj_conf,
            input_augment_funcs = in_trans_fns,
            output_augment_funcs = out_trans_fns)

        if hasattr(prj_conf, 'val_input_dirs'):
            val_input_dirs = prj_conf.val_input_dirs
        else:
            val_input_dirs = prj_conf.input_dirs

        if hasattr(prj_conf, 'val_output_dirs'):
            val_output_dirs = prj_conf.val_output_dirs
        else:
            val_output_dirs = prj_conf.output_dirs

        if prj_conf.val_list is not None:
            val_lst = prj_conf.val_list
            val_set = nii_dset.NII_MergeDataSetLoader(
                prj_conf.val_set_name,
                val_lst,
                val_input_dirs, \
                prj_conf.input_exts, \
                prj_conf.input_dims, \
                prj_conf.input_reso, \
                prj_conf.input_norm, \
                val_output_dirs, \
                prj_conf.output_exts, \
                prj_conf.output_dims, \
                prj_conf.output_reso, \
                prj_conf.output_norm, \
                './', \
                params = params,
                truncate_seq= prj_conf.truncate_seq, 
                min_seq_len = prj_conf.minimum_len,
                save_mean_std = False,
                wav_samp_rate = prj_conf.wav_samp_rate,
                way_to_merge = args.way_to_merge_datasets,
                global_arg = args,
                dset_config = prj_conf,
                input_augment_funcs = in_trans_fns,
                output_augment_funcs = out_trans_fns)
        else:
            val_set = None

        # initialize the model and loss function
        model = prj_model.Model(trn_set.get_in_dim(), \
                                trn_set.get_out_dim(), \
                                args, prj_conf, trn_set.get_data_mean_std())
        loss_wrapper = prj_model.Loss(args)
        
        # initialize the optimizer
        optimizer_wrapper = nii_op_wrapper.OptimizerWrapper(model, args)

        # if necessary, resume training
        if args.trained_model == "":
            checkpoint = None 
        else:
            checkpoint = torch.load(args.trained_model)
            
        # pre-training using standard procedure
        # change args
        args_tmp = copy.deepcopy(args)
        args_tmp.epochs = args.active_learning_pre_train_epoch_num
        args_tmp.not_save_each_epoch = True
        args_tmp.save_trained_name += '_pretrained'
        args_tmp.active_learning_cycle_num = 0
        pretraind_name = args_tmp.save_trained_name + args_tmp.save_model_ext
        if args.active_learning_pre_train_epoch_num:
            nii_warn.f_print_w_date("Normal training (warm-up) phase",level='h')
            nii_warn.f_print("Normal training for {:d} epochs".format(
                args.active_learning_pre_train_epoch_num))
            op_wrapper_tmp = nii_op_wrapper.OptimizerWrapper(model, args_tmp)
            loss_wrapper_tmp = prj_model.Loss(args_tmp)
            nii_nn_wrapper_base.f_train_wrapper(
                args_tmp, model, loss_wrapper, device, op_wrapper_tmp,
                trn_set, val_set, checkpoint)
            checkpoint = torch.load(pretraind_name)
        elif checkpoint is None:
            if os.path.isfile(pretraind_name):
                checkpoint = torch.load(pretraind_name)
                nii_warn.f_print("Use pretrained model before active learning")
        else:
            nii_warn.f_print("Use seed model to initialize")

        nii_warn.f_print_w_date("Active learning phase",level='h')
        # start training
        nii_nn_wrapper.f_train_wrapper(
            args, model, 
            loss_wrapper, device,
            optimizer_wrapper,
            trn_set, pool_set, val_set, checkpoint)
        # done for traing

    else:
        
        # for inference
        
        # default, no truncating, no shuffling
        params = {'batch_size':  args.batch_size,
                  'shuffle': False,
                  'num_workers': args.num_workers,
                  'sampler': args.sampler}

        in_trans_fns = prj_conf.test_input_trans_fns \
                       if hasattr(prj_conf, 'test_input_trans_fns') else None
        out_trans_fns = prj_conf.test_output_trans_fns \
                        if hasattr(prj_conf, 'test_output_trans_fns') else None
        
        if type(prj_conf.test_list) is list:
            t_lst = prj_conf.test_list
        else:
            t_lst = nii_list_tool.read_list_from_text(prj_conf.test_list)

        test_set = nii_dset.NII_MergeDataSetLoader(
            prj_conf.test_set_name, \
            t_lst, \
            prj_conf.test_input_dirs,
            prj_conf.input_exts, 
            prj_conf.input_dims, 
            prj_conf.input_reso, 
            prj_conf.input_norm,
            prj_conf.test_output_dirs, 
            prj_conf.output_exts, 
            prj_conf.output_dims, 
            prj_conf.output_reso, 
            prj_conf.output_norm,
            './',
            params = params,
            truncate_seq= None,
            min_seq_len = None,
            save_mean_std = False,
            wav_samp_rate = prj_conf.wav_samp_rate,
            way_to_merge = args.way_to_merge_datasets,
            global_arg = args,
            dset_config = prj_conf,
            input_augment_funcs = in_trans_fns,
            output_augment_funcs = out_trans_fns)
        
        # initialize model
        model = prj_model.Model(test_set.get_in_dim(), \
                                test_set.get_out_dim(), \
                                args, prj_conf)

        if args.trained_model == "":
            print("No model is loaded by ---trained-model for inference")
            print("By default, load %s%s" % (args.save_trained_name,
                                              args.save_model_ext))
            checkpoint = torch.load("%s%s" % (args.save_trained_name,
                                              args.save_model_ext))
        else:
            checkpoint = torch.load(args.trained_model)
            
        # do inference and output data
        nii_nn_wrapper_base.f_inference_wrapper(
            args, model, device, test_set, checkpoint)
    # done
    return

if __name__ == "__main__":
    main()

