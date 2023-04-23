#!/usr/bin/env python
"""
config_parse

Argument parse

"""
from __future__ import absolute_import

import os
import sys
import argparse

import core_scripts.other_tools.list_tools as nii_list_tools
import core_scripts.other_tools.display as nii_display

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"



#############################################################
# argparser
#
def f_args_parsed(argument_input = None):
    """ Arg_parse
    """
    
    parser = argparse.ArgumentParser(
        description='General argument parse')
    
    ######
    # lib
    mes = 'module of model definition (default model, model.py will be loaded)'
    parser.add_argument('--module-model', type=str, default="model", help=mes)

    mes = 'module of configuration (default config, config.py will be loaded)'
    parser.add_argument('--module-config', type=str, default="config", 
                        help=mes)

    mes = 'module of auxiliary model definition (in case this is needed)'
    parser.add_argument('--module-model-aux', type=str, default="", help=mes)
    
    ######
    # Training settings    
    mes = 'batch size for training/inference (default: 1)'
    parser.add_argument('--batch-size', type=int, default=1, help=mes)


    mes = 'number of mini-batches to accumulate (default: 1)'
    parser.add_argument('--size-accumulate-grad', type=int, default=1, help=mes)
    
    mes = 'number of epochs to train (default: 50)'
    parser.add_argument('--epochs', type=int, default=50, help=mes)
    
    mes = 'number of no-best epochs for early stopping (default: 5)'
    parser.add_argument('--no-best-epochs', type=int, default=5, help=mes)

    mes = 'force to save trained-network.pt per epoch, '
    mes += 'no matter whether the epoch is currently the best.'
    parser.add_argument('--force-save-lite-trained-network-per-epoch', 
                        action='store_true', default=False, help=mes)

    mes = 'sampler (default: None). Default sampler is random shuffler. '
    mes += 'Option 1: block_shuffle_by_length, shuffle data by length'
    parser.add_argument('--sampler', type=str, default='None', help=mes)

    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='learning rate (default: 0.0001)')
    
    mes = 'learning rate decaying factor, using '
    mes += 'torch.optim.lr_scheduler.ReduceLROnPlateau(patience=no-best-epochs,'
    mes += ' factor=lr-decay-factor). By default, no decaying is used.'
    mes += ' Training stopped after --no-best-epochs.'
    parser.add_argument('--lr-decay-factor', type=float, default=-1.0, help=mes)
    
    mes = 'lr scheduler: 0: ReduceLROnPlateau (default); 1: StepLR; '
    mes += 'this option is set on only when --lr-decay-factor > 0. '
    mes += 'Please check core_scripts/op_manager/lr_scheduler.py '
    mes += 'for detailed hyper config for each type of lr scheduler'
    parser.add_argument('--lr-scheduler-type', type=int, default=0, help=mes)

    mes = 'lr patience: patience for torch_optim_steplr.ReduceLROnPlateau '
    mes += 'this option is used only when --lr-scheduler-type == 0. '
    parser.add_argument('--lr-patience', type=int, default=5, help=mes)

    mes = 'lr step size: step size for torch.optim.lr_scheduler.StepLR'
    mes += 'this option is used only when --lr-scheduler-type == 1. '
    parser.add_argument('--lr-steplr-size', type=int, default=5, help=mes)

    mes = 'L2 penalty on weight (default: not use). '
    mes += 'It corresponds to the weight_decay option in Adam'
    parser.add_argument('--l2-penalty', type=float, default=-1.0, help=mes)

    mes = 'gradient norm (torch.nn.utils.clip_grad_norm_ of Pytorch)'
    mes += 'default (-1, not use)'
    parser.add_argument('--grad-clip-norm', type=float, default=-1.0,
                        help=mes)
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    
    parser.add_argument('--seed', type=int, default=1, 
                        help='random seed (default: 1)')
    
    mes = 'turn model.eval() on validation set (default: false)'
    parser.add_argument('--eval-mode-for-validation', \
                        action='store_true', default=False, help=mes)

    mes = 'if model.forward(input, target), please set this option on. '
    mes += 'This is used for autoregressive model, auto-encoder, and so on. '
    mes += 'When --model-forward-with-file-name is also on, '
    mes += 'model.forward(input, target, file_name) should be defined'
    parser.add_argument('--model-forward-with-target', \
                        action='store_true', default=False, help=mes)

    mes = 'if model.forward(input, file_name), please set option on. '
    mes += 'This is used with forward requires file name of the data. '
    mes += 'When --model-forward-with-target is also on, '
    mes += 'model.forward(input, target, file_name) should be defined'
    parser.add_argument('--model-forward-with-file-name', \
                        action='store_true', default=False, help=mes)
    
    mes = 'shuffle data? (default true). Set --shuffle will turn off shuffling'
    parser.add_argument('--shuffle', action='store_false', \
                        default=True, help=mes)

    mes = 'number of parallel workers to load data (default: 0)'
    parser.add_argument('--num-workers', type=int, default=0, help=mes)

    mes = 'use DataParallel to levarage multiple GPU (default: False)'
    parser.add_argument('--multi-gpu-data-parallel', \
                        action='store_true', default=False, help=mes)

    mes = 'way to concatenate multiple datasets: '
    mes += 'concatenate: simply merge two datasets as one large dataset. '
    mes += 'batch_merge: make a minibatch by drawing one sample from each set. '
    mes += '(default: concatenate)'
    parser.add_argument('--way-to-merge-datasets', type=str, \
                        default='concatenate', help=mes)

    mes = "Ignore invalid data? the length of features does not match"
    parser.add_argument('--ignore-length-invalid-data', 
                        action='store_true', default=False, help=mes)


    mes = "Ignore existing cache file dic"
    parser.add_argument('--ignore-cached-file-infor', 
                        action='store_true', default=False, help=mes)

    mes = "External directory to store cache file dic"
    parser.add_argument('--path-cache-file', type=str, default="", help=mes)

    mes = "Skip scanning data directories (by default False)"
    parser.add_argument('--force-skip-datadir-scanning', 
                        action='store_true', default=False, help=mes)
    
    
    ######
    # options to save model / checkpoint
    parser.add_argument('--save-model-dir', type=str, \
                        default="./", \
                        help='save model to this direcotry (default ./)')
    
    mes = 'do not save model after every epoch (default: False)'
    parser.add_argument('--not-save-each-epoch', action='store_true', \
                        default=False, help=mes)

    mes = 'name prefix of saved model (default: epoch)'
    parser.add_argument('--save-epoch-name', type=str, default="epoch", \
                        help=mes)

    mes = 'name of trained model (default: trained_network)'
    parser.add_argument('--save-trained-name', type=str, \
                        default="trained_network", help=mes)
    
    parser.add_argument('--save-model-ext', type=str, default=".pt",
                        help='extension name of model (default: .pt)')

    mes = 'save model after every N mini-batches (default: 0, not use)'
    parser.add_argument('--save-model-every-n-minibatches', type=int, 
                        default=0, help=mes)

    
    #######
    # options for active learning
    mes = 'Number of active leaning cycles'
    parser.add_argument('--active-learning-cycle-num', type=int, default=0, 
                        help = mes)
    
    mes = 'Whetehr use base traing set with new samples? (default True)'
    parser.add_argument('--active-learning-use-new-data-only', 
                        action='store_true', default=False, help = mes)

    mes = 'Number of samples selected per cycle? (default =batch size)'
    parser.add_argument('--active-learning-new-sample-per-cycle', type=int, 
                        default=0, help = mes)

    mes = 'Use model.train() during data retrieval (defaul False)'
    parser.add_argument('--active-learning-train-model-for-retrieval', 
                        action='store_true', default=False, help = mes)

    mes = 'Retrieve data with replacement (defaul True)'
    parser.add_argument('--active-learning-with-replacement', 
                        action='store_true', default=False, help = mes)

    mes = 'Number of pre-trainining epochs before active learniing (defaul 0)'
    parser.add_argument('--active-learning-pre-train-epoch-num', type=int, 
                        default=0, help=mes)

    mes = 'Name of the cache file to store names of selected or removed data'
    parser.add_argument('--active-learning-cache-dataname-save', type=str, 
                        default="cache_al_data_log", help=mes)

    mes = 'Path to the cache file that stores names of selected or removed data'
    parser.add_argument('--active-learning-cache-dataname-path', type=str, 
                        default="", help=mes)


    #######
    # options to load model
    mes = 'a trained model for inference or resume training '
    parser.add_argument('--trained-model', type=str, \
                        default="", help=mes + "(default: '')")

    mes = 'do not load previous training error information.'
    mes += " Load only model para. and optimizer state  (default: false)"
    parser.add_argument('--ignore-training-history-in-trained-model', 
                        action='store_true', \
                        default=False, help=mes)    

    mes = 'do not load previous training statistics in optimizer.'
    mes += " (default: false)"
    parser.add_argument('--ignore-optimizer-statistics-in-trained-model', 
                        action='store_true', \
                        default=False, help=mes)    


    mes = 'load pre-trained model even if there is mismatch on the number of'
    mes += " parameters. Mismatched part will not be loaded (default: false)"
    parser.add_argument('--allow-mismatched-pretrained-model', 
                        action='store_true', \
                        default=False, help=mes)    

    mes = 'run inference mode (default: False, run training script)'
    parser.add_argument('--inference', action='store_true', \
                        default=False, help=mes)    

    mes = 'run model conversion script (default: False)'
    parser.add_argument('--epoch2pt', action='store_true', \
                        default=False, help=mes)    


    mes = 'inference only on data whose minibatch index is within the range of '
    mes = mes + '[--inference-sample-start-index, --inference-sample-end-index)'
    mes = mes + 'default: 0, starting from the 1st data'
    parser.add_argument('--inference-sample-start-index', type=int, default=0,
                        help=mes)

    mes = 'inference only on data whose minibatch index is within the range of '
    mes = mes + '[--inference-sample-start-index, --inference-sample-end-index)'
    mes = mes + 'default: -1, until the end of all data'
    parser.add_argument('--inference-sample-end-index', type=int, default=-1,
                        help=mes)
    
    mes = 'inference data list. A list of file names that should '
    mes = mes + 'be processed during the inference stage. '
    mes = mes + 'If such a data list is provided, only data listed will '
    mes = mes + 'be processed.'
    parser.add_argument('--inference-data-list', type=str, default="",
                        help=mes)
    
    #######
    # options to output
    mes = 'path to save generated data (default: ./output)'
    parser.add_argument('--output-dir', type=str, default="./output", \
                        help=mes)

    # options to output
    mes = 'prefix added to file name (default: no string)'
    parser.add_argument('--output-filename-prefix', type=str, default="", \
                        help=mes)
    
    mes = 'truncate input data sequences so that the max length < N.'
    mes += ' (default: -1, not do truncating at all)'
    parser.add_argument('--trunc-input-length-for-inference', type=int,
                        default=-1, help=mes)


    mes = 'truncate input data overlap length (default: 5)'
    parser.add_argument('--trunc-input-overlap', type=int, default=5, help=mes)


    mes = 'which optimizer to use (Adam | SGD, default: Adam)'
    parser.add_argument('--optimizer', type=str, default='Adam', help=mes)
    
    mes = 'verbose level 0: nothing; 1: print error per utterance'
    mes = mes + ' (default: 1)'
    parser.add_argument('--verbose', type=int, default=1,
                        help=mes)


    #######
    # options for debug mode
    mes = 'debug mode, each epoch only uses a specified number of mini-batches'
    mes += ' (default: 0, not used)'
    parser.add_argument('--debug-batch-num', type=int, default=0, help=mes)

    #######
    # options for user defined 
    mes = 'a temporary flag without specific purpose.'
    mes += 'User should define args.temp_flag only for temporary usage.'
    parser.add_argument('--temp-flag', type=str, default='', help=mes)


    mes = 'reverse the order when loading data from the dataset.'
    mes += 'This should not not used if --sampler block_shuffle_by_length '
    parser.add_argument('--flag-reverse-data-loading-order', 
                        action='store_true', default=False, help=mes)

    #######
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)')    

    
    #######
    # profile options
    mes = "options to setup Pytorch profile. It must be a string like A-B-C-D"
    mes += ' where A, B, C, D are integers. Meanining of these options are in'
    mes += ' torch.profiler.schedule. Default 1-1-3-2.'
    parser.add_argument('--wait-warmup-active-repeat', type=str, 
                        default='1-1-3-2', 
                        help=mes)
    mes = "directory to save profiling output. Default ./log_profile"
    parser.add_argument('--profile-output-dir', type=str, 
                        default='./log_profile')
    

    #######
    # data options
    mes = 'option to set silence_handler on waveform data.\n'
    mes += ' 0: do nothing, use the data as it is (default) \n'
    mes += ' 1: remove segments with small energy, use other segments\n'
    mes += ' 2: keep only segments with small energy, remove other segments\n'
    mes += ' 3: remove segments with small energy only at begining and end\n'
    mes += 'Code in core_scripts.data_io.wav_tools.silence_handler. '
    mes += 'This option is used when input or output contains only waveform. '
    mes += 'It only processes waveform. Other features will not be trimmed.'
    parser.add_argument('--opt-wav-silence-handler', type=int, 
                        default=0, help=mes)


    mes = 'update data length in internal buffer if data length is changed '
    mes += 'by augmentation method. This is useful, for example, when using '
    mes += '--sampler block_shuffle_by_length --opt-wav-silence-handler 3 '
    mes += 'or using other data augmentation method changes data length.'
    parser.add_argument('--force-update-seq-length', action='store_true', \
                        default=False, help=mes)


    #
    # done
    if argument_input is not None:
        return parser.parse_args(argument_input)
    else:
        return parser.parse_args()


if __name__ == "__main__":
    pass
    
