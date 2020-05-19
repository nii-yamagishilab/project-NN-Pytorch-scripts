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
    input_dirs = ['/home/smg/wang/WORK/WORK/DATA/CMU/all/5ms/melspec',
                  '/home/smg/wang/WORK/WORK/DATA/CMU/all/5ms/f0']
    input_dims = [80, 1]
    input_exts = ['.mfbsp', '.f0']
    input_reso = [80, 80]
    input_norm = [True, True]
    
    output_dirs = ['/home/smg/wang/WORK/WORK/DATA/CMU/all/wav_16k']
    output_dims = [1]
    output_exts = ['.wav']
    output_reso = [1]
    output_norm = [False]

    wav_samp_rate = 16000
    truncate_seq = 16000 * 3
    minimum_len = 80 * 50
    
    if True:
        params = {'batch_size':  1,
                  'shuffle': True,
                  'num_workers': 2}

        file_list = '/home/smg/wang/WORK/WORK/DATA/CMU/all/scp/train.lst'
        file_list = nii_list_tool.read_list_from_text(file_list)
        trnset_wrapper = nii_data_io.NIIDataSetLoader("cmu_all_trn", file_list,
                                                      input_dirs, input_exts, input_dims, 
                                                      input_reso, input_norm,
                                                      output_dirs, output_exts, output_dims, 
                                                      output_reso, output_norm,
                                                      './', params = params,
                                                      truncate_seq= truncate_seq, 
                                                      min_seq_len = minimum_len,
                                                      save_mean_std = True,
                                                      wav_samp_rate = wav_samp_rate)

        file_list = '/home/smg/wang/WORK/WORK/DATA/CMU/all/scp/val.lst'
        file_list = nii_list_tool.read_list_from_text(file_list)
        valset_wrapper = nii_data_io.NIIDataSetLoader("cmu_all_val", file_list,
                                                      input_dirs, input_exts, input_dims, 
                                                      input_reso, input_norm,
                                                      output_dirs, output_exts, output_dims, 
                                                      output_reso, output_norm,
                                                      './', params = params,
                                                      truncate_seq= truncate_seq, 
                                                      min_seq_len = minimum_len,
                                                      save_mean_std = False,
                                                      wav_samp_rate = wav_samp_rate)
        
        model = self_model.Model(trnset_wrapper.get_in_dim(), \
                                 trnset_wrapper.get_out_dim(), \
                                 args, \
                                 trnset_wrapper.get_data_mean_std())

        loss_wrapper = self_model.Loss(args)
        optimizer_wrapper = nii_op_wrapper.OptimizerWrapper(model, args)
        checkpoint = None 
        nii_nn_wrapper.f_train_wrapper(args, model, loss_wrapper, device,
                                       optimizer_wrapper,
                                       trnset_wrapper, valset_wrapper, checkpoint)
    else:

        wav_samp_rate = 16000
        truncate_seq = None
        minimum_len = None 
        
        params = {'batch_size':  1,
                  'shuffle': False,
                  'num_workers': 0}
        file_list = '/home/smg/wang/WORK/WORK/DATA/CMU/all/scp/test.lst'
        file_list = ['slt_arctic_b0474', 'slt_arctic_b0475', 'slt_arctic_b0476',
                     'bdl_arctic_b0474', 'bdl_arctic_b0475', 'bdl_arctic_b0476',
                     'rms_arctic_b0474', 'rms_arctic_b0475', 'rms_arctic_b0476',
                     'clb_arctic_b0474', 'clb_arctic_b0475', 'clb_arctic_b0476']
        dataset_wrapper = nii_data_io.NIIDataSetLoader("cmu_all_test_tiny", 
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
        checkpoint = torch.load('trained_network.pt')
        nii_nn_wrapper.f_inference_wrapper(args, model, device, dataset_wrapper, checkpoint)
        
if __name__ == "__main__":
    main()
    #config_new = nii_config_parse.ConfigParse('./temp.config')
    #print(config_new.f_retrieve('Compression', None, 'bool'))

