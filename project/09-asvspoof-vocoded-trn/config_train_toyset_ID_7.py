#!/usr/bin/env python
"""
config.py for project-NN-pytorch/projects

Usage: 
 For training, change Configuration for training stage
 For inference,  change Configuration for inference stage
"""
import os
import pandas as pd

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2022, Xin Wang"

#########################################################
## Configuration for training stage
#########################################################
# Name of datasets
#  this will be used as the name of cache files created for each set
# 
# Name for the seed training set, in case you merge multiple data sets as
#  a single training set, just specify the name for each subset.
#  Here we only have 1 training subset
trn_set_name = ['asvspoof2019_toyset_vocoded_trn_bona']
val_set_name = ['asvspoof2019_toyset_vocoded_val']


# For convenience, specify a path to the toy data set
#  because config*.py will be copied into model-*/config_AL_train_toyset/NN
#  we need to use ../../../ 
tmp = os.path.dirname(__file__) + '/../../../DATA/toy_example_vocoded'

# File list for training and development sets
#   (text file, one file name per line, without name extension)
#   we need to provide one lst for each subset
#   trn_list[n] will correspond to trn_set_name[n]
# for training set, we only load bonafide through default data IO
# then, (paired) spoofed data will be loaded through data augmentation function
trn_list = [tmp + '/scp/train_bonafide.lst']
# for development set
val_list = [tmp + '/scp/dev.lst']

# Directories for input data
#   We need to provide the path to the directory that saves the input data.
#   We assume waveforms for training and development of one subset 
#   are stored in the same directory.
#   Hence, input_dirs[n] is for trn_set_name[n] and val_set_name[n]
#   
#   If you need to specify a separate val_input_dirs
#   val_input_dirs = [[PATH_TO_DEVELOPMENT_SET]]
# 
#   Each input_dirs[n] is a list, 
#   for example, input_dirs[n] = [wav, speaker_label, augmented_wav, ...]
#   
#   Here, input for each file is a single waveform
input_dirs = [[tmp + '/train_dev']]

val_input_dirs = [[tmp + '/train_dev']]

# Dimensions of input features
#   What is the dimension of the input feature 
#   len(input_dims) should be equal to len(input_dirs[n]) 
#
#   Here, input for each file is a single waveform, dimension is 1
input_dims = [1]

# File name extension for input features
#   input_exts = [name_extention_of_feature_1, ...]
#   len(input_exts) should be equal to len(input_dirs[n]) 
# 
#   Here, input file extension is .wav
#   We use .wav not .flac
input_exts = ['.wav']

# Temporal resolution for input features
#   This is not relevant for CM but for other projects
#   len(input_reso) should be equal to len(input_dirs[n]) 
#   Here, it is 1 for waveform
input_reso = [1]

# Whether input features should be z-normalized
#   This is not relevant for CM but for other projects
#   len(input_norm) should be equal to len(input_dirs[n]) 
#   Here, it is False for waveform
#   We don't normalize the waveform
input_norm = [False]
    

# Similar configurations for output features
#  Here, we set output to empty because we will load
#  the target labels from protocol rather than output feature
#  '.bin' is also a place holder
output_dirs = [[] for x in input_dirs]
val_output_dirs = [[]]
output_dims = [1]
output_exts = ['.bin']
output_reso = [1]
output_norm = [False]

# ===
# Waveform configuration
# ===

# Waveform sampling rate
#  wav_samp_rate can be None if no waveform data is used
wav_samp_rate = 16000

# Truncating input sequences so that the maximum length = truncate_seq
#  When truncate_seq is larger, more GPU mem required
#  If you don't want truncating, please truncate_seq = None
# Here, we don't use truncate_seq included in data_io, but we will do 
# truncation in data_augmentation functions
truncate_seq = None

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = None
    

# Optional argument
#  This used to load protocol(s)
#  Multiple protocol files can be specified in the list
# 
#  Note that these protocols should cover all the 
#  training, development, and pool set data.
#  Otherwise, the code will raise an error
#  
#  Here, this protocol will cover all the data in the toy set
optional_argument = [tmp + '/protocol.txt']

# ===
# pre-trained SSL model
# ===
# We will load this pre-trained SSL model as the front-end
# 
# path to the SSL model (it is downloaded by 01_download.sh)
ssl_front_end_path = os.path.dirname(__file__) \
                     + '/../../../SSL_pretrained/xlsr_53_56k.pt'
# dimension of the SSL model output
#  this must be provided.
ssl_front_end_out_dim = 1024


# ===
# data augmentation option
# ===
# for training with aligned bonafide-spoofed mini-batches,
# we have to use this customized function to make sure that
# we can load the aligned files

# We will use function in data_augment.py to process the loaded mini-batch
import data_augment
# path to the waveform directory (the same as input_dirs above)
wav_path = tmp + '/train_dev'
# path to the protocol of spoofed data ( the same as optional_argument)
protocol_path = tmp + '/protocol.txt'
# configuration to use Pandas to parse the protocol
protocol_cols = ['voice', 'trial', '-', 'attack', 'label']
# length to truncate the waveform
trim_len = 64000

# load protocol
protocol_pd = pd.read_csv(protocol_path, sep = ' ', names = protocol_cols)
# how many number of spoofed trials exist for every bonafide
# this toy dataset has four vocoded (spoofed) versions of each bonafide trial
num_spoofed_trials = 4
# where this database provided bonafide and spoofed pairs?
flag_database_aligned = True

def spoof_name_loader(bona_name, 
                      wav_path=wav_path, 
                      data_pd = protocol_pd,
                      num_spoofed = num_spoofed_trials,
                      flag_database_aligned = flag_database_aligned):
    """
    """
    # get the spoofed data list that contains the name of the bona fide
    # spoofs can be empty for non-aligned database, in that case,
    if flag_database_aligned:
        # if aligned we assume the spoofed and bonafide audio file names have 
        # common part, thus, we can select those trials from the list
        # in toy dataset
        # bonafide trial has name LA_X_YYYY
        # spoofed trials are named as vocoder1_LA_X_YYYY, vocoder2_LA_X_YYYY
        idx = data_pd.query('label == "spoof"')['trial'].str.contains(bona_name)
        spoofs = data_pd.query('label == "spoof"')[idx]['trial'].to_list()
        if len(spoofs) == 0:
            print("Warn: cannot find spoofed data for {:s}".format(bona_name))
            print("Possible reason: {:s} is spoofed trial".format(bona_name))
            print("How-to-resolve: make sure that trn_list={:s}".format(
                str(trn_list)
            ))
            print(" in config only contains bona fide data")
    else:
        # if not aligned, we randomply sample spoofed trials
        idx = data_pd.query('label == "spoof"')['trial'].str.contains('LA_T')
        tmp_pd = data_pd.query('label == "spoof"')[idx].sample(n=num_spoofed)
        spoofs = tmp_pd['trial'].to_list()
    
    spoof_paths = []
    for spoof_name in spoofs:
        filepath = wav_path + '/' + spoof_name + '.wav'
        if not os.path.isfile(filepath):
            print("Cannot find {:s}".format(filepath))
        else:
            spoof_paths.append(filepath)
    return spoof_paths


#### 
# wrapper of data augmentation functions
# the wrapper calls the data_augmentation function defined in
# data_augment.py.
# these wrapper will be called in data_io when loading the data
# from disk
####

# wrapper for training set
input_trans_fns = [
    [lambda x, y: data_augment.wav_aug_wrapper(
        x, y, wav_samp_rate, trim_len, spoof_name_loader)],
    [lambda x, y: data_augment.wav_aug_wrapper(
        x, y, wav_samp_rate, trim_len, spoof_name_loader)]]

output_trans_fns = [[], []]

# wrapper for development set
# development does nothing but simply truncate the waveforms
val_input_trans_fns = [
    [lambda x, y: data_augment.wav_aug_wrapper_val(
        x, y, wav_samp_rate, trim_len, spoof_name_loader)],
    [lambda x, y: data_augment.wav_aug_wrapper_val(
        x, y, wav_samp_rate, trim_len, spoof_name_loader)]]

val_output_trans_fns = [[], []]

#########################################################
## Configuration for inference stage
#########################################################
# This part is not used in this project
# They are place holders

test_set_name = trn_set_name + val_set_name

# List of test set data
# for convenience, you may directly load test_set list here
test_list = trn_list + val_list

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
test_input_dirs = input_dirs * 2

# Directories for output features, which are []
test_output_dirs = [[]] * 2

