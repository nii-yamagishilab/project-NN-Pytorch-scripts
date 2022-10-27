#!/usr/bin/env python
"""
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

def self_defined_transfer(model_src, model_tar):
    """ A self defined function to transfer the weights from model_src 
    to model_tar
    """
    
    # load SSL front-end
    model_tar.m_front_end.ssl_model.load_state_dict(
        model_src.m_ssl.state_dict())
    
    # load SSL front-end linear layer
    model_tar.m_front_end.m_front_end_process.load_state_dict(
        model_src.m_frontend[0].state_dict())

    # load the linear output layer
    model_tar.m_back_end.m_utt_level.load_state_dict(
        model_src.m_output_act[0].state_dict())

    return


def main():
    """ main(): the default wrapper for training and inference process
    Please prepare config.py and model.py
    """
    # arguments initialization
    args = nii_arg_parse.f_args_parsed()

    # 
    nii_warn.f_print_w_date("Start program", level='h')
    nii_warn.f_print("Load module: %s" % (args.module_config))
    nii_warn.f_print("Load 1st module: %s" % (args.module_model))
    nii_warn.f_print("Load 2nd module: %s" % (args.module_model_aux))
    
    prj_conf = importlib.import_module(args.module_config)
    prj_model_src = importlib.import_module(args.module_model)
    prj_model_tar = importlib.import_module(args.module_model_aux)

    # initialization
    nii_startup.set_random_seed(args.seed, args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # 
    checkpoint = torch.load(args.trained_model)


    model_src = prj_model_src.Model(sum(prj_conf.input_dims), 
                                    sum(prj_conf.output_dims),
                                    args, prj_conf)
    
    model_tar = prj_model_tar.Model(sum(prj_conf.input_dims), 
                                    sum(prj_conf.output_dims),
                                    args, prj_conf)

    model_src.load_state_dict(checkpoint)

    self_defined_transfer(model_src, model_tar)
    
    torch.save(model_tar.state_dict(), 'temp.pt')
    return


if __name__ == "__main__":
    main()
