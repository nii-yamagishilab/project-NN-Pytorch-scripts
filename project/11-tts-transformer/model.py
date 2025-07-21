#!/usr/bin/python3
"""
Model wrapper

"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.utils.logger import get_logger

from tqdm.contrib import tqdm

from tools import metric_sasv
from tools import io_tools
from tools import util_torch

from submodules import transformer
from submodules import postnet
from submodules import spk_emb
import datasets

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, Xin Wang"

logger = get_logger(__name__)

# Loss function can be part of model definition
# TO DO: move to transformer.py?
class TransformerLoss(torch.nn.Module):
    def __init__(self,
                 compute_mel_loss,
                 compute_stop_loss,
                 compute_guided_att_loss,
                 compute_spec_loss):
        super(TransformerLoss, self).__init__()

        self.compute_mel_loss = compute_mel_loss
        self.compute_stop_loss = compute_stop_loss
        self.compute_guided_att_loss = compute_guided_att_loss
        self.compute_spec_loss = compute_spec_loss

        return
    
    def forward(self,
                gen_mel, post_mel, gen_mag, stop, align_mat,
                tar_mel, tar_mag, olens, ilens):
        
        # mel loss
        mel_loss, mel_loss_post = self.compute_mel_loss(gen_mel, post_mel, tar_mel, olens)

        # stop token loss
        stop_loss = self.compute_stop_loss(stop, olens)

        # guided attention loss
        att_loss = self.compute_guided_att_loss(align_mat, olens, ilens)

        # linear spec loss
        # detach and don't propagate back 
        spec_loss = self.compute_spec_loss(gen_mag, tar_mag)

        return [mel_loss, mel_loss_post, stop_loss, att_loss, spec_loss]



class Model(sb.core.Brain):
    """Main definition of model wrapper
    """
    def __init__(self, *args, **kargs):
        """
        """
        # initialization of the parent class
        super(Model, self).__init__(*args, **kargs)
        
        # checkpointer will be initialized from yaml
        # if case the model is not defined in yaml, add them here
        # to the checkpointer recoverable settings
        self.checkpointer.add_recoverables(
            {
                'model': self.modules.model,
                'postnet': self.modules.postnet,
                'counter': self.hparams.epoch_counter,
            }
        )
        return


    @staticmethod
    def create_modules(config, run_opts):
        """create_modules(config)

        Speechbrain defines model in yaml. When initializing a sb.score.Brain 
        instance (a model), we can use this method to return modules required
        by sb.score.Brain. By doing this, we can define the model in *.py,
        not in yaml.
        """
        # check whether speaker embedding is needed
        if 'spkemb_config' in config and config['spkemb_config']['output_dim'] > 0:
            flag_spkemb = True
        else:
            flag_spkemb = False

        modules = {
            'model': transformer.TransformerTTS(**config['transformer_config']),
            'postnet': postnet.PostNet(config['postnet_config']), 
            'loss_compute': TransformerLoss(
                transformer.TransformerTTS.compute_mel_loss,
                transformer.TransformerTTS.compute_stop_loss,
                transformer.TransformerTTS.compute_guided_att_loss,
                postnet.PostNet.compute_spec_loss),
            'spkemb': spk_emb.Pretrained_speechbrain(
                config['spkemb_config'], run_opts) if flag_spkemb else None
        }

        # remove None
        modules = {x:y for x, y in modules.items() if y is not None }
        
        return modules

    def load_checkpoint(self):
        """load_best_ckpt()
        
        Load the best checkpoint from save directory specified in 
        checkpointer.checkpoints_dir
        """
        # find the ckpt
        if hasattr(self.hparams, "pick_checkpoint"):
            min_key = self.hparams.pick_checkpoint
            best_ckpt = self.checkpointer.find_checkpoint(min_key=min_key)
        else:
            # for this model, we check the latest checkpoint
            best_ckpt = self.checkpointer.find_checkpoint()

        if best_ckpt:
            logger.info("Load checkpoint: {:s}".format(str(best_ckpt.path)))
            # load models (back and front ends)
            best_paramfile = best_ckpt.paramfiles["model"]
            sb.utils.checkpoints.torch_parameter_transfer(
                self.modules.model, best_paramfile)
            
            best_paramfile = best_ckpt.paramfiles["postnet"]
            sb.utils.checkpoints.torch_parameter_transfer(
                self.modules.postnet, best_paramfile)
        else:
            logger.info("Cannot find checkpoint")
            sys.exit(1)
            
        return

    def compute_forward(self, batch, stage):
        """
        """

        # load data
        batch = batch.to(self.device)
        # text input data (converted to indx)
        text, _ = batch.text
        # target mel and length
        tar_mel, tar_len = batch.tar_mel
        # target stft magnitude
        tar_mag, _ = batch.tar_mag
        # target waveform
        tar_wav, _ = batch.tar_wav
        
        # output length in list 
        olens = [int(np.round(x * tar_mel.shape[1])) for x in tar_len.cpu().numpy()]

        # prepare feedback feature (shift by one step)
        if hasattr(self.modules.model, 'return_rec_fac'):
            rec_fac = self.modules.model.return_rec_fac()
        elif hasattr(self.modules.model, 'module'):
            rec_fac = self.modules.model.module.return_rec_fac()
        else:
            rec_fac = 1
            
        fb_mel = torch.roll(tar_mel, rec_fac, dims=1)
        fb_mel[:, 0:rec_fac] *= 0

        # prepare speaker embedding
        if 'spkemb' in self.modules:
            spk_emb = self.modules.spkemb.encode_batch(tar_wav)
        else:
            spk_emb = None
            
        # transformer text -> mel
        gen_mel, post_mel, stop, _, align_mat, _ = self.modules.model(
            text, fb_mel, olens, spk_emb)
        
        # mel -> stft magnitude
        gen_mag = self.modules.postnet(post_mel.detach())
            
        return gen_mel, post_mel, gen_mag, stop, align_mat

    def compute_objectives(self, model_output, batch, stage):
        """
        """
        # get input length
        text, txt_len = batch.text
        tar_mel, tar_len = batch.tar_mel
        tar_mag, _ = batch.tar_mag
        
        # compute the length of input and output sequences
        #  length returned by sb is a ratio against the longest utt in the minibatch
        olens = [int(np.round(x * tar_mel.shape[1])) for x in tar_len.cpu().numpy()]
        ilens = [int(np.round(x * text.shape[1])) for x in txt_len.cpu().numpy()]

        # compute loss
        loss_tuple = self.modules.loss_compute(
            *model_output,
            tar_mel,
            tar_mag,
            olens,
            ilens)

        # save the generated output for dev set
        if stage != sb.Stage.TRAIN:
            # dump loss and intermediate features
            self._stage_dump(loss_tuple, model_output, olens, ilens, batch)

        # return the loss
        return torch.sum(torch.stack(loss_tuple))

    def inference(self, batch):
        """inference(batch)
        """
        # load text data
        batch = batch.to(self.device)
        text, _ = batch.text

        # load ref waveform
        # (currently target waveform)
        ref_wav, _ = batch.tar_wav
        # prepare speaker embedding
        if hasattr(self.modules, 'spkemb') and self.modules.spkemb is not None:
            spk_emb = self.modules.spkemb.encode_batch(ref_wav)
        else:
            spk_emb = None
        
        # wrapper over ddp
        # transformer
        if hasattr(self.modules.model, 'module'):
            gen_mel, _, _ = self.modules.model.module.inference(text, spk_emb=spk_emb)
        else:
            gen_mel, _, _ = self.modules.model.inference(text, spk_emb=spk_emb)

        # post-net
        gen_mag = self.modules.postnet(gen_mel)
        
        return gen_mel, gen_mag

        
    def _stage_dump(self, loss_tuple, model_output, olens, ilens, batch):
        ##
        # this part needs to be customized
        ##
        # the structure of the data depends on the model API definition
        loss_tensor = torch.stack(loss_tuple).detach().cpu()

        # detatch the output from model output
        model_output_ = [x.detach().cpu() for x in model_output[:-1]]
        # the attention matrix is a list of tensor
        model_output_.append([x.detach().cpu() for x in model_output[-1]])
        
        # dump the data into list buffer
        self.dev_loss_bag.append(loss_tensor)
        # just dump actual output of one mini-batch
        if len(self.dev_out_bag) == 0:
            self.dev_out_bag.append(model_output_)
            self.dev_name_bag.append(batch.filename)
            self.dev_olens_bag.append(olens)
            self.dev_ilens_bag.append(ilens)

            # dumpy input text for inference
            text, _ = batch.text
            tar_wav, _ = batch.tar_wav
            self.dev_text_bag.append(text)
            self.dev_wav_bag.append(tar_wav)
            
        return

    def _stage_end_dump_tb(self, epoch):
        """dump features to tensorboard during

        This part needs to be customized
        """

        ###
        # log meta information
        ###
        # dump loss
        # [(5), (5) ...] -> stack -> [N, 5] -> mean [5] 
        loss_tuple = torch.stack(self.dev_loss_bag, dim=0).mean(dim=0)
        # name of the losses
        loss_names = ['loss_mel', 'loss_mel_post',
                      'loss_stop', 'loss_align', 'loss_post']
        valid_stats = {loss_names[idx] : y.item() \
                       for idx, y in enumerate(loss_tuple)}
        # tensorboard meta information
        #  to resume step
        self.hparams.train_tensorboard_logger.global_step['meta'] = epoch-1
        self.hparams.train_tensorboard_logger.global_step['valid'] = {
            x: epoch-1 for x in loss_names}
        # log
        self.hparams.train_tensorboard_logger.log_stats(
            stats_meta={"Epoch": epoch,
                        'LR': self.optimizer.param_groups[0]["lr"]},
            valid_stats=valid_stats)

        ###
        # dumpy feature
        ###
        # only dump two files if there are more
        dump_num = self.hparams.valid_dump_num
        
        # dump mel and others, just pick one mini-batch
        gen_mel = self.dev_out_bag[0][0]
        post_mel = self.dev_out_bag[0][1]
        gen_mag = self.dev_out_bag[0][2]
        stop = self.dev_out_bag[0][3]
        align_mat = self.dev_out_bag[0][4]
        filenames = self.dev_name_bag[0]
        olens = self.dev_olens_bag[0]
        ilens = self.dev_ilens_bag[0]
        text = self.dev_text_bag[0]
        tar_wav = self.dev_wav_bag[0]

        # dump from longest sentences
        sorted_idx = np.argsort(ilens)[::-1]
        
        for idx in sorted_idx[:dump_num]:
            filename = filenames[idx]
            data_len = olens[idx]
            text_len = ilens[idx]
            
            # mel spectrogram
            self.hparams.train_tensorboard_logger.log_figure(
                'gen_mel/{:s}'.format(filename), gen_mel[idx, :data_len].T)
            self.hparams.train_tensorboard_logger.log_figure(
                'post_mel/{:s}'.format(filename), post_mel[idx, :data_len].T)
            self.hparams.train_tensorboard_logger.log_figure(
                'gen_mag/{:s}'.format(filename), gen_mag[idx, :data_len].T)

            # attention
            bs = gen_mel.shape[0]
            for layer_idx in range(len(align_mat)):
                # shape [4, olen, ilen] 
                align_mat_ = align_mat[layer_idx][idx::bs]
                # concate along input len dimension: [olen, ilen * 4]
                align_mat_ = torch.concat(
                    [x[:data_len, :text_len] for x in align_mat_], dim=-1)
                # save
                self.hparams.train_tensorboard_logger.log_figure(
                    'attn_mat_{:d}/{:s}'.format(layer_idx, filename),
                    align_mat_)
            
            # create an audio
            mag = gen_mag[idx, :data_len].detach().cpu().numpy()
            wav = datasets.recover_wav(mag, self.hparams.tar_sample_rate, self.hparams.spec_config)
            self.hparams.train_tensorboard_logger.log_audio(
                "audio/{:s}".format(filename), wav,
                self.hparams.tar_sample_rate)


            # only run inference after a few training epochs
            if epoch >= self.hparams.valid_dump_after_epoch:
                # run inference
                with torch.no_grad():
                    # prepare speaker embedding
                    if hasattr(self.modules, 'spkemb') and self.modules.spkemb is not None:
                        spk_emb = self.modules.spkemb.encode_batch(tar_wav)
                    else:
                        spk_emb = [None for x in range(text.shape[0])]
                    
                    if hasattr(self.modules.model, 'module'):
                        inf_mel, attn, stop = self.modules.model.module.inference(
                            text[idx:idx+1, :ilens[idx]], spk_emb=spk_emb[idx:idx+1])
                    else:
                        inf_mel, attn, stop = self.modules.model.inference(
                            text[idx:idx+1, :ilens[idx]], spk_emb=spk_emb[idx:idx+1])
                    inf_mag = self.modules.postnet(inf_mel)
                    # if due to any reason, the data is too short (10 frames)
                    if inf_mag.shape[1] < 10:
                        inf_mel = inf_mel.repeat(1, 10, 1)
                        inf_mag = inf_mag.repeat(1, 10, 1)
            else:
                # just create dummpy inference output
                inf_mel = torch.zeros_like(gen_mel[:, :data_len])
                inf_mag = torch.zeros_like(gen_mag[:, :data_len])
                attn = [torch.zeros_like(x) for x in align_mat]
                stop = torch.zeros_like(gen_mel)
                
            mag = inf_mag[0].cpu().numpy()
            wav = datasets.recover_wav(mag, self.hparams.tar_sample_rate, self.hparams.spec_config)
            self.hparams.train_tensorboard_logger.log_audio(
                "inf_audio/{:s}".format(filename), wav,
                self.hparams.tar_sample_rate)

            for layer_idx in range(len(attn)):
                align_mat_ = attn[layer_idx]
                # concate along input len dimension: [olen, ilen * 4]
                align_mat_ = torch.concat([x for x in align_mat_], dim=-1)
                # save
                self.hparams.train_tensorboard_logger.log_figure(
                    'inf_attn_mat_{:d}/{:s}'.format(layer_idx, filename),
                    align_mat_)
                
            self.hparams.train_tensorboard_logger.log_figure(
                'inf_post_mel/{:s}'.format(filename),inf_mel[0].T)
            self.hparams.train_tensorboard_logger.log_figure(
                'inf_gen_mag/{:s}'.format(filename), inf_mag[0].T)
            self.hparams.train_tensorboard_logger.log_figure(
                'inf_stop/{:s}'.format(filename),
                stop[0].repeat(1, 2))
            
            
        return
        
    def on_stage_start(self, stage, epoch=None):

        if epoch is not None \
           and epoch > (self.hparams.number_of_epochs - self.hparams.stoptoken_epochs):
            # let model only tune the stop token
            self.modules.model.setflag_tune_stoptoken()

        
        if stage != sb.Stage.TRAIN:
            # clean the buffer whenever the new validation around begins
            self.dev_loss_bag = []
            self.dev_out_bag = []
            self.dev_olens_bag = []
            self.dev_ilens_bag = []
            self.dev_name_bag = []
            self.dev_text_bag = []
            self.dev_wav_bag = []
        return
    
    def on_stage_end(self, stage, stage_loss, epoch=None):
        
        stage_stats = {"loss": stage_loss}
        
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        if stage == sb.Stage.VALID:

            ###
            # dump to log
            ###
            tmp_loss = torch.stack(self.dev_loss_bag, dim=0).mean(dim=0)
            
            for idx, loss in enumerate(tmp_loss):
                stage_stats['subloss_{:d}'.format(idx+1)] = loss.item()
            
            self.hparams.train_logger.log_stats(
	        stats_meta={"epoch": epoch},
                train_stats=None,
                valid_stats=stage_stats,
            )
            
            self.checkpointer.save_checkpoint(meta=stage_stats)
            #self.checkpointer.save_and_keep_only(
            #    meta=stage_stats, min_keys=["sasv_eer"])

            ###
            # dump to tensorboard
            ###
            self._stage_end_dump_tb(epoch)
            
        return            

    def test(self, eval_dataset, eval_loader_para):
        """test(eval_daaset, eval_loader_para)
        
        """
        # prepare data loader
        eval_dataloader = sb.dataio.dataloader.make_dataloader(
            eval_dataset, **eval_loader_para)
        # pus hmodel to eval mode
        self.modules.eval()
        
        # inference
        with torch.no_grad():
            for batch in eval_dataloader: 
                # call Inference API
                batch = batch.to(self.device)
                gen_mel, gen_mag = self.inference(batch)

                # save output
                datasets.save_output(
                    gen_mag.cpu().numpy(),
                    batch.filename,
                    self.hparams.tar_sample_rate,
                    self.hparams.spec_config,
                    self.hparams.path_audio_save)
                
        return


    def analysis(self, dev_dataset, dev_loader_para, save_folder):
        return
        

if __name__ == "__main__":
    print(__doc__)
