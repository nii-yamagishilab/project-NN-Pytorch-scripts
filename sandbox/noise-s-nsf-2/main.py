from __future__ import absolute_import
import os
import sys
import torch

import core_scripts.data_io.default_data_io as nii_data_io
import core_scripts.data_io.conf as nii_dconf
import core_scripts.other_tools.list_tools as nii_list_tool
import core_scripts.config_parse.config_parse as nii_config_parse
import core_scripts.config_parse.arg_parse as nii_arg_parse
import core_scripts.op_manager.op_manager as nii_op_wrapper
import core_scripts.nn_manager.nn_manager as nii_nn_wrapper

import model as self_model

def main():

    # arguments initialization
    args = nii_arg_parse.f_args_parsed()

    # initialization
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # prepare data io
    input_dirs = ['/home/smg/wang/WORK/WORK/WORK/music-model-2020/DATA/maestro-v2.0/mel/mel24k/L-Channel']
    input_dims = [80]
    input_exts = ['.mfbsp']
    input_reso = [288]
    input_norm = [True]
    
    output_dirs = ['/home/smg/wang/WORK/WORK/WORK/music-model-2020/DATA/maestro-v2.0/wav/wav24k/L-Channel']
    output_dims = [1]
    output_exts = ['.wav']
    output_reso = [1]
    output_norm = [False]

    wav_samp_rate = 24000
    truncate_seq = 48000 // 288 * 288
    minimum_len = 288 * 50
    
    if True:
        params = {'batch_size':  1,
                  'shuffle': True,
                  'num_workers': 10,
                  'pin_memory': True}

        file_list = '/home/smg/wang/WORK/WORK/WORK/music-model-2020/DATA/maestro-v2.0/scp/scp_400/train.scp'
        file_list = nii_list_tool.read_list_from_text(file_list)
        trnset_wrapper = nii_data_io.NIIDataSetLoader("maestro_400_trn", file_list,
                                                      input_dirs, input_exts, input_dims, input_reso, input_norm,
                                                      output_dirs, output_exts, output_dims, output_reso, output_norm,
                                                      './', params = params,
                                                      truncate_seq= truncate_seq, min_seq_len = minimum_len,
                                                      save_mean_std = True,
                                                      wav_samp_rate = wav_samp_rate)

        file_list = '/home/smg/wang/WORK/WORK/WORK/music-model-2020/DATA/maestro-v2.0/scp/scp_400/val.scp'
        file_list = nii_list_tool.read_list_from_text(file_list)
        valset_wrapper = nii_data_io.NIIDataSetLoader("maestro_400_val", file_list,
                                                      input_dirs, input_exts, input_dims, input_reso, input_norm,
                                                      output_dirs, output_exts, output_dims, output_reso, output_norm,
                                                      './', params = params,
                                                      truncate_seq= truncate_seq, min_seq_len = minimum_len,
                                                      save_mean_std = False,
                                                      wav_samp_rate = wav_samp_rate)
        
        model = self_model.Model(trnset_wrapper.get_in_dim(), \
                                 trnset_wrapper.get_out_dim(), \
                                 args, \
                                 trnset_wrapper.get_data_mean_std())
        loss_wrapper = self_model.Loss(args)
        optimizer_wrapper = nii_op_wrapper.OptimizerWrapper(model, args)
        
        checkpoint = torch.load('epoch_004.pt')
        nii_nn_wrapper.f_train_wrapper(args, model, loss_wrapper, device,
                                       optimizer_wrapper,
                                       trnset_wrapper, valset_wrapper, checkpoint)
    else:

        input_dirs = ['/home/smg/wang/WORK/WORK/WORK/music-model-2020/DATA/maestro-v2.0/TESTDATA/24k/split_short']
        output_dirs = input_dirs

        wav_samp_rate = 24000
        truncate_seq = None
        minimum_len = None 
        
        params = {'batch_size':  1,
                  'shuffle': False,
                  'num_workers': 0}
        tmp_path = '/home/smg/wang/WORK/WORK/WORK/music-model-2020/DATA/maestro-v2.0/TESTDATA/24k/split_short/file.lst'
        file_list = nii_list_tool.read_list_from_text(tmp_path)
        dataset_wrapper = nii_data_io.NIIDataSetLoader("maestro_test_tiny", 
                                                       file_list, input_dirs,
                                                       input_exts, input_dims, input_reso, input_norm,
                                                       output_dirs, output_exts, 
                                                       output_dims, output_reso, output_norm,
                                                       './',
                                                       params = params,
                                                       truncate_seq = None,
                                                       min_seq_len = minimum_len,
                                                       save_mean_std = False,
                                                       wav_samp_rate = wav_samp_rate)
        
        model = self_model.Model(dataset_wrapper.get_in_dim(), \
                                 dataset_wrapper.get_out_dim(), \
                                 args)
        checkpoint = torch.load('epoch_006.pt')
        nii_nn_wrapper.f_inference_wrapper(args, model, device, dataset_wrapper, checkpoint)
        
if __name__ == "__main__":
    main()
    #config_new = nii_config_parse.ConfigParse('./temp.config')
    #print(config_new.f_retrieve('Compression', None, 'bool'))

