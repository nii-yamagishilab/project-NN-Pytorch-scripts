#!/usr/bin/env python
"""
main.py for project-NN-pytorch/projects

The default training/inference process wrapper
Requires model.py and config.py

Usage: $: python main.py [options]
"""
from __future__ import absolute_import
import os
import sys
import torch
import importlib

import core_scripts.data_io.default_data_io as nii_default_dset
import core_scripts.data_io.customize_dataset as nii_dset
import core_scripts.other_tools.display as nii_warn
import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.config_parse.config_parse as nii_config_parse
import core_scripts.config_parse.arg_parse as nii_arg_parse
import core_scripts.op_manager.op_manager as nii_op_wrapper
import core_scripts.nn_manager.nn_manager as nii_nn_wrapper
import core_scripts.nn_manager.nn_manager_GAN as nii_nn_wrapper_GAN
import core_scripts.startup_config as nii_startup


__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


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
    if not args.inference and not args.epoch2pt:
        params = {'batch_size':  args.batch_size,
                  'shuffle':  args.shuffle,
                  'num_workers': args.num_workers,
                  'sampler': args.sampler,
                  'pin_memory': True}

        in_trans_fns = prj_conf.input_trans_fns \
                       if hasattr(prj_conf, 'input_trans_fns') else None
        out_trans_fns = prj_conf.output_trans_fns \
                        if hasattr(prj_conf, 'output_trans_fns') else None
        inout_trans_fns = prj_conf.input_output_trans_fn \
                        if hasattr(prj_conf, 'input_output_trans_fn') else None
        
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
            output_augment_funcs = out_trans_fns,
            inoutput_augment_func = inout_trans_fns)

        if prj_conf.val_list is not None:
            val_lst = prj_conf.val_list
            val_set = nii_dset.NII_MergeDataSetLoader(
                prj_conf.val_set_name,
                val_lst,
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
                output_augment_funcs = out_trans_fns,
                inoutput_augment_func = inout_trans_fns)
        else:
            val_set = None

        # initialize the model and loss function
        model_G = prj_model.ModelGenerator(
            trn_set.get_in_dim(), trn_set.get_out_dim(), \
            args, prj_conf, trn_set.get_data_mean_std())
        model_D = prj_model.ModelDiscriminator(
            trn_set.get_in_dim(), trn_set.get_out_dim(), 
            args, prj_conf, trn_set.get_data_mean_std())
        loss_wrapper = None
        
        # initialize the optimizer
        optimizer_G_wrap = nii_op_wrapper.OptimizerWrapper(model_G, args)
        optimizer_D_wrap = nii_op_wrapper.OptimizerWrapper(model_D, args)

        # if necessary, resume training
        if args.trained_model == "":
            checkpoint_G = None 
            checkpoint_D = None
        else:
            tmp_str = args.trained_model.split(",")
            checkpoint_G = torch.load(tmp_str[0])
            if len(tmp_str) > 1:
                checkpoint_D = torch.load(tmp_str[1])
            else:
                checkpoint_D = None

        # start training
        nii_nn_wrapper_GAN.f_train_wrapper_GAN(
            args, model_G, model_D, 
            loss_wrapper, device,
            optimizer_G_wrap, optimizer_D_wrap,
            trn_set, val_set, 
            checkpoint_G, checkpoint_D)
        # done for traing

    elif args.inference:
        
        # for inference
        
        # default, no truncating, no shuffling
        params = {'batch_size':  args.batch_size,
                  'shuffle': False,
                  'num_workers': args.num_workers}

        in_trans_fns = prj_conf.input_trans_fns \
                       if hasattr(prj_conf, 'test_input_trans_fns') else None
        out_trans_fns = prj_conf.output_trans_fns \
                        if hasattr(prj_conf, 'test_output_trans_fns') else None
        inout_trans_fns = prj_conf.output_trans_fns \
                        if hasattr(prj_conf, 'test_input_output_trans_fn') \
                           else None
        
        
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
            truncate_seq = None,
            min_seq_len = None,
            save_mean_std = False,
            wav_samp_rate = prj_conf.wav_samp_rate, 
            way_to_merge = args.way_to_merge_datasets,
            global_arg = args,
            dset_config = prj_conf,
            input_augment_funcs = in_trans_fns,
            output_augment_funcs = out_trans_fns,
            inoutput_augment_func = inout_trans_fns)
        
        # initialize model
        model = prj_model.ModelGenerator(
            test_set.get_in_dim(), test_set.get_out_dim(), args, prj_conf)

        if args.trained_model == "":
            print("Please provide ---trained-model")
            sys.exit(1)
        else:
            checkpoint = torch.load(args.trained_model)
            
        # do inference and output data
        nii_nn_wrapper.f_inference_wrapper(
            args, model, device, test_set, checkpoint)

    elif args.epoch2pt:
        # for model conversion from epoch.pt to trained_network.pt
        
        # initialize model
        model = prj_model.ModelGenerator(
            sum(prj_conf.input_dims), sum(prj_conf.output_dims), args, prj_conf)

        if args.trained_model == "":
            print("Please provide ---trained-model")
            sys.exit(1)
        else:
            checkpoint = torch.load(args.trained_model)
            
        # do inference and output data
        nii_nn_wrapper.f_convert_epoch_to_trained(
            args, model, device, checkpoint)

    else:
        print("Fatal error in main.py")
        sys.exit(1)
        
    # done
    return

if __name__ == "__main__":
    main()

