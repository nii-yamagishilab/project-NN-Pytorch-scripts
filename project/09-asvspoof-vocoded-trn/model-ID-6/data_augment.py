#!/usr/bin/env python
"""Functions for data augmentation

These functions are written using Numpy.
They should be used before data are casted into torch.tensor.

For example, use them in config.py:input_trans_fns or output_trans_fns


"""
import os
import sys
import numpy as np

from core_scripts.other_tools import debug as nii_debug
from core_scripts.data_io import wav_tools as nii_wav_tools
from core_scripts.data_io import wav_augmentation as nii_wav_aug

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2021, Xin Wang"


def load_spoof_trials(bona_name, spoof_loader):
    """spoof_audios = load_spoof_utterance
    """

    # load the spoofed data
    spoofs = spoof_loader(bona_name)
    spoof_list = []
    for filepath in spoofs:
        if os.path.isfile(filepath):
            _, data = nii_wav_tools.waveReadAsFloat(filepath)
        else:
            print("Cannot find {:s}".format(filepath))
            sys.exit(1)
        # make the shape (length, 1)
        spoof_list.append(np.expand_dims(data, axis=1))
    return spoof_list


def mixup_wav(input_wav_list, target_list, alpha, beta, mixup_method='wav'):
    """
    """
    # output buffer
    out_buf, out_tar1, out_tar2, out_gamma = [], [], [], []

    # assume input_wav_list[0] is bonafide
    bona_idx = 0
    bona_wav = input_wav_list[bona_idx]

    # mixed bonafide and spoofed
    for spoof_idx, spoof_wav in enumerate(input_wav_list):
        if spoof_idx == 0:
            # this is bonafide
            continue
        else:
            # mixup 
            gamma = np.random.beta(alpha, beta)
            mixed_wav = nii_wav_aug.morph_wavform(
                bona_wav, spoof_wav, gamma, method=mixup_method)
            out_buf.append(mixed_wav)
            out_tar1.append(target_list[bona_idx])
            out_tar2.append(target_list[spoof_idx])
            out_gamma.append(gamma)

    # return
    return out_buf, out_tar1, out_tar2, out_gamma


def wav_aug_wrapper(input_data, 
                    name_info, 
                    wav_samp_rate, 
                    length,
                    spoof_loader):

    # load spoofed trials
    spoof_trials = load_spoof_trials(
        name_info.seq_name, spoof_loader)

    # use this API to randomly trim the input length
    batch_data = [input_data] + spoof_trials
    batch_data = nii_wav_aug.batch_pad_for_multiview(
        batch_data, wav_samp_rate, length, random_trim_nosil=True)

    
    # first target is for bonafide
    # the rest are for spoofed
    orig_target = [0] * len(batch_data)
    orig_target[0] = 1

    # rawboost to create multiple view data
    aug_list = [nii_wav_aug.RawBoostWrapper12(x) for x in batch_data]
    # to be compatible with the rest of the code, 
    tar_list_1, tar_list_2 = orig_target, orig_target
    # gamma_list=1.0 will be equivalent to normal CE
    gamma_list = [1.0] * len(batch_data)
    
    # assign bonafide and spoofed data to different feature classes
    # for supervised contrastive learning loss
    new_feat_class = [2] * len(batch_data)
    new_feat_class[0] = 1

    
    # merge the original data and mixed data
    new_wav_list = [np.concatenate(batch_data, axis=1), 
                    np.concatenate(aug_list, axis=1)]
    new_tar_list_1 = [np.array(orig_target), np.array(tar_list_1)]
    new_tar_list_2 = [np.array(orig_target), np.array(tar_list_2)]
    new_gamma_list = [np.array([1.0] * len(batch_data)), np.array(gamma_list)]
        
    # waveform stack to (length, num_of_wav, num_of_aug)
    output = np.stack(new_wav_list, axis=2)
    # label stack to (num_of_wav, num_of_aug)
    new_tar_1 = np.stack(new_tar_list_1, axis=1)
    new_tar_2 = np.stack(new_tar_list_2, axis=1)
    # gamma the same shape as label
    new_gamma = np.stack(new_gamma_list, axis=1)
    # (num_of_wav)
    new_feat_class = np.array(new_feat_class)
    
    return [output, new_tar_1, new_tar_2, new_gamma, 
            new_feat_class, new_feat_class]

def wav_aug_wrapper_val(input_data, 
                        name_info, 
                        wav_samp_rate, 
                        length,
                        spoof_name_loader):
    # use this API to randomly trim the input length
    input_data = nii_wav_aug.batch_pad_for_multiview(
        [input_data], wav_samp_rate, length, random_trim_nosil=False)[0]
    return input_data

if __name__ == "__main__":
    print("Tools for data augmentation")
