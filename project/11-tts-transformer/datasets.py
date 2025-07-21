#!/usr/bin/python3
"""
Modules for create datasets for SASV.

Two types of modules are available
1. those for loading waveforms
2. those for loading pre-extracted embeddings
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
from logging import getLogger


import numpy as np
import pickle
from pathlib import Path
import speechbrain as sb

import tools.wav_tools as wav_tools
import tools.dsp_tools as dsp_tools

import tools.protocol_tools as protocol_tools
from tools.text_tools import text_io

from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.sampler import DynamicBatchSampler

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"

logger = getLogger(__name__)

#####
# Utils
#####
def get_wav_path(root_dir, filename, flag_flac=False):
    """ path = get_wav_path(root_dir, filename, flag_flac)
    Return full path to the file.
    
    
    input
    -----
      root_dir: str, path to the root directory of ASVspoof2019 LA
      filename: str, name of the file
      flag_flac: bool, whether audio file is in flac or wav

    output
    ------
      filepath: str, full path to the audio file
    """
    if flag_flac:
        # this is for loading flac 
        filepath = Path(root_dir) / filename
        return filepath.with_suffix('.flac')
    else:
        # this is for loading wav
        filepath = Path(root_dir) / filename
        return filepath.with_suffix('.wav')
        

def get_wav_data(filepath):
    """ sr, data = get_wav_data(filepath)
    Load wav audio data

    input
    -----
      filepath: str, path to the audio file
    
    output
    ------
      sr: int, sampling rate
      data: np.array, (N, ), audio data
    """
    if filepath.suffix == '.flac':
        # speechbrain uses torchaudio to load flac
        # this requires that flac-related libs
        # https://pytorch.org/audio/0.7.0/backend.html#sox-io-backend
        data = read_audio(str(filepath)).numpy()
        sr = None
    elif filepath.suffix == '.wav':
        # use the in-house tool 
        sr, data = wav_tools.waveReadAsFloat(str(filepath))
        data = np.squeeze(data)
    return sr, data


def pad_data(data, expected_len):
    """data_new = pad_data(data, expected_len)
    Pad or trim input data to a fixed length
    
    input
    -----
      data: np.array, (N, )
      expected_len: int, expected length of the data
                    if it is not a int, expected_len = N
    output
    ------
      data_new: np.array, (expected_len, )
    """
    if type(expected_len) is not int:
        data_new = data
    else:
        if data.shape[0] >= expected_len:
            data_new = data[:expected_len]
        else:
            num_repeats = int(expected_len / data.shape[0]) + 1
            data_new = np.tile(data, (num_repeats))[:expected_len]
    return data_new


def pad_pack_data(data_list, expected_len = None):
    """data_new = pad_pack_data(data_list, expected_len)
    Pad or trim input data in a list to a fixed length
    
    input
    -----
      data_list: list of np.array
      expected_len: int, expected length of the data
                    None, it will be max(data length) in data_list
    output
    ------
      data_new: np.array, (K, expected_len), where K=len(data_list)
    """
    maxlen = max([x.size for x in data_list])
    buflen = maxlen if expected_len is None else expected_len
    
    batch_data = np.zeros([len(data_list), buflen], dtype=data_list[0].dtype)
    for idx, data in enumerate(data_list):
        batch_data[idx] = pad_data(data, buflen)
    return batch_data


########
# Datasets for loading waveforms
########

def create_dataset(csv_file, hparams, require_output=True):
    """
    """

    # ID, speaker, utterance
    train_dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=csv_file)

    # function to sample data
    def __sample_trn_trials(speaker, filename, hparams):

        # path to text file
        txt_ext = hparams['txt_ext'] if 'txt_ext' in hparams else '.txt'
        txt_path = (Path(hparams['path_text_root']) / filename).with_suffix(txt_ext)
        text = text_io.textloader(txt_path, 'EN')

        # path to waveform file
        wav_path = get_wav_path(hparams['path_audio_root'], filename)
        
        if wav_path.is_file():
            sr, wav_data = get_wav_data(wav_path)

            if hparams['sample_rate'] != sr:
                logger.info("{:s} is {:d} Hz".format(str(wav_path), sr))
                logger.info("configuration says {:d} Hz".format(hparams['sample_rate']))
                sys.exit(1)
            if hparams['tar_sample_rate'] != sr:
                logger.info("{:s} is {:d} Hz".format(str(wav_path), sr))
                logger.info("Target sr is {:d} Hz".format(hparams['tar_sample_rate']))
                sys.exit(1)

            if 'spec_config' in hparams:
                spec_config = hparams['spec_config']
            else:
                spec_config = dsp_tools.default_spec_config(sr)
                
            tar_mel, tar_mag, tar_len = dsp_tools.get_spec(wav_data, sr, spec_config, filename)

            #if tar_mel.shape[0] >= hparams['max_frame_len']:
            #    logger.info("{:s} is too long. Set it to dummy".format(str(wav_path)))
            #    tar_len = 10
            #    tar_mel = tar_mel[0:tar_len]
            #    tar_mag = tar_mag[0:tar_len]
            #    text = text_io._text2code('', 'EN', None)
                
            
        elif require_output:
            logger.info("Cannot find output file {:s}".format(str(wav_path)))
            sys.exit(1)
            
        else:
            tar_mel = np.zeros([0])
            tar_mag = np.zeros([0])
        
        return text, tar_mel, tar_mag, filename, wav_data

    # add item 
    train_dataset.add_dynamic_item(
        lambda x, y: __sample_trn_trials(x, y, hparams),
        takes=['speaker', 'filename'],
        provides=['text', 'tar_mel', 'tar_mag', 'filename', 'tar_wav'])
    
    # set output keys
    train_dataset.set_output_keys(['text', 'tar_mel', 'tar_mag', 'filename', 'tar_wav'])

    return train_dataset


def get_dynamic_batch_sampler(dataset, dynamic_hparams, key='duration'):
    #dynamic_hparams = hparams["dynamic_batch_sampler_train"]
    train_sampler = DynamicBatchSampler(
        dataset,
        length_func=lambda x: float(x[key]),
        **dynamic_hparams,
    )
    return train_sampler
    


def recover_wav(mag, sr, spec_config):
    # recover wave from stft magnitude using griffin-lim
    #sr = hparams['tar_sample_rate']
    
    if spec_config is None:
        spec_config = dsp_tools.default_spec_config(sr)

    wav = dsp_tools.mag2wav_gl(mag, sr, spec_config)
    return wav

def save_output(data_tensor, filenames, sr, spec_config, save_folder):
    # function to save output
    # here, waveform needs to be recovered from the stft magnitude
    assert data_tensor.shape[0] == len(filenames), \
        "Number of file mismatch in save_output"

    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    for data, filename in zip(data_tensor, filenames):
        wav = recover_wav(data, sr, spec_config)
        save_path = (Path(save_folder) / filename).with_suffix('.wav')

        logger.info("generate wav {:s}".format(str(save_path)))
        wav_tools.waveFloatToPCMFile(wav, str(save_path), sr=sr)
        
    return
        


if __name__ == "__main__":
    print(__doc__)
