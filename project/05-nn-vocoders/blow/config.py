#!/usr/bin/env python
"""
config.py

This configuration file specifiess the input and output data
"""

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2020, Xin Wang"

#########################################################
## Configuration for training stage
#########################################################

# Name of datasets
#  after data preparation, trn/val_set_name are used to save statistics 
#  about the data sets
trn_set_name = ['vctk_blow_trn']
val_set_name = ['vctk_blow_val']

# for convenience
tmp1 = '../DATA/vctk-blow'

# File lists (text file, one data name per line, without name extension)
# trin_file_list: list of files for training set
trn_list = [tmp1 + '/scp/train.lst']
# val_file_list: list of files for validation set. It can be None
val_list = [tmp1 + '/scp/val.lst']

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
input_dirs = [[tmp1 + '/vctk_wav']]

# Dimensions of input features
# input_dims = [dimension_of_feature_1, dimension_of_feature_2, ...]
input_dims = [1]

# File name extension for input features
# input_exts = [name_extention_of_feature_1, ...]
# Please put ".f0" as the last feature
input_exts = ['.wav']

# Temporal resolution for input features
# input_reso = [reso_feature_1, reso_feature_2, ...]
#  for waveform modeling, temporal resolution of input acoustic features
#  may be = waveform_sampling_rate * frame_shift_of_acoustic_features
#  for example, 80 = 16000 Hz * 5 ms 
input_reso = [1]

# Whether input features should be z-normalized
# input_norm = [normalize_feature_1, normalize_feature_2]
input_norm = [False]
    
# Similar configurations for output features
#  for convenience, simply load the data as target too
output_dirs = input_dirs
output_dims = [1]
output_exts = ['.wav']
output_reso = [1]
output_norm = [False]


# Waveform sampling rate
#  wav_samp_rate can be None if no waveform data is used
wav_samp_rate = 16000

# Truncating input sequences so that the maximum length = truncate_seq
#  When truncate_seq is larger, more GPU mem required
# If you don't want truncating, please truncate_seq = None
truncate_seq = None

# Minimum sequence length
#  If sequence length < minimum_len, this sequence is not used for training
#  minimum_len can be None
minimum_len = None
    

# Other optional arguments, the definition of which depends on a specific model
#  here we define a speaker ID manager
class VCTKSpeakerMap:
    def __init__(self):
        speaker_list = tmp1 + '/scp/spk.lst'
        self.m_speaker_map = {}
        with open(speaker_list, 'r') as file_ptr:
            for idx, line in enumerate(file_ptr):
                line = line.rstrip('\n')
                self.m_speaker_map[line] = idx
        return

    def num(self):
        # leave one for unseen
        return len(self.m_speaker_map) + 1

    def parse(self, filename, return_idx=True):
        # 
        # filename will be in this format: '8758,p***_***,1,4096,16000'
        # we need to get the p*** part
        spk = filename.split('_')[0].split(',')[-1]
        if return_idx:
            return self.get_idx(spk)
        else:
            return spk
        
    def get_idx(self, spk):
        if spk in self.m_speaker_map:
            return self.m_speaker_map[spk]  
        else:
            return len(self.m_speaker_map)

# conversion_map defines the conversion pairs: p361 -> p245, p278 -> p287 ...
options = {'speaker_map': VCTKSpeakerMap(),
           'conversion_map': {'p361': 'p245',
                              'p278': 'p287',
                              'p302': 'p298',
                              'p361': 'p345',
                              'p260': 'p267',
                              'p273': 'p351',
                              'p245': 'p273',
                              'p304': 'p238',
                              'p297': 'p283',
                              'p246': 'p362'}}

# data pre-processing functions
#  input_trans_fns must have the same shape as input_dirs
#  output_trans_fns must have the same shape as output_dirs
#  Thus, input_trans_fns[x][y] defines the transformation function
#  for the y-th feature of the x-th sub-database.
#  
#  This function is called when DataLoader loads the data from the disk
#  (see f_post_data_process in core_scripts.data_io.default_io.py)
#  
#  4096 denotes frame length
#  0.2 is the reference coefficient for waveform emphasis
from sandbox import block_blow
input_trans_fns = [[lambda x: block_blow.wav_aug(x, 4096, 0.2, wav_samp_rate)]]
output_trans_fns = [[lambda x: block_blow.wav_aug(x, 4096, 0.2, wav_samp_rate)]]


#########################################################
## Configuration for inference stage
#########################################################
# similar options to training stage

test_set_name = ['vctk_blow_test']

# List of test set data
# for convenience, you may directly load test_set list here
test_list = [tmp1 + '/scp/test_tiny.lst']

# Directories for input features
# input_dirs = [path_of_feature_1, path_of_feature_2, ..., ]
#  we assume train and validation data are put in the same sub-directory
test_input_dirs = [[tmp1 + '/vctk_wav_test_tiny']]

# Directories for output features, which are []
test_output_dirs = [[]]


