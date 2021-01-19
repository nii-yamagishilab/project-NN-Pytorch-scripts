#!/usr/bin/env python
"""
nn_manager_conf

A few definitions of nn_manager

"""
from __future__ import print_function

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

####
# Name of keys in checkpoint
# Epochs saved as checkpoint will have these fields
#  state_dict: network weights
#  info: printed information from the nn_manager
#  optimizer: optimizer state
#  trnlog: log of training error on training set
#  vallog: log of validation error on validation set
#  lr_scheduler: status for learning rate scheduler
####
class CheckPointKey:
    state_dict = 'state_dict'
    info = 'info'
    optimizer = 'optimizer' 
    trnlog = 'train_log'
    vallog = 'val_log'
    lr_scheduler = 'lr_scheduler'

####
# Methods that a Model should have 
#  name: (flag_mandatory, comment)
#   name: str, name of the method
#   flag_mandatory: bool, whether this method is mandatory to have
#   comment: str, comment string on the method
# 
####
nn_model_keywords = {
    #
    # mandatory methods that a model must define
    'prepare_mean_std': (True, "method to initialize mean/std"),
    'normalize_input': (True, "method to normalize input features"),
    'normalize_target': (True, "method to normalize target features"),
    'denormalize_output': (True, "method to de-normalize output features"),
    'forward': (True, "main method for forward"),
    #
    # optional methods that a model can define
    # inference:
    #  mainly for AR model. In training, forward() is used while in generation
    #  inference() is used. For other models, inference may be unnecessary
    'inference': (False, "alternative method for inference"),
    #
    # loss:
    #  A model can define loss inside its module. This is convenient when
    #  loss requires some additional information stored in the model.
    #  If model.loss() is not defined, Loss defined in model.py will be used.
    'loss': (False, 'loss defined within model module'),
    #
    # other_setups:
    #  other setups functions that should be applied before training
    'other_setups': (False, "other setup functions before training"),
    #
    # flag_validation:
    #  model.training tells whether this is under trainining or not
    #  model.flag_validation tells whether the model is on train set or val set
    'flag_validation': (False, 'flag to indicate train or validation set'),
    'validation': (False, 'please use model.flag_validation')
    #
}


if __name__ == "__main__":
    print("Configurations for nn_manager")
