#!/bin/bash
# Download pre-trained SSL models from fairseq
# See https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md
# https://github.com/pytorch/fairseq/tree/main/examples/hubert
# Please check their webpage for the latest link

TAR=SSL_pretrained

# XLSR-53 Large
# W2V-XLSR 
if [ ! -e ./${TAR}/xlsr_53_56k.pt ]; 
then
    wget -q --show-progress -O ./${TAR}/xlsr_53_56k.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_53_56k.pt 
fi

# Wav2Vec 2.0 Base No finetuning
# W2V-small
if [ ! -e ./${TAR}/wav2vec_small.pt ];
then
    wget -q --show-progress -O ./${TAR}/wav2vec_small.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt
fi

# Wav2Vec 2.0 Large (LV-60 + CV + SWBD + FSH) **  No finetuning
# W2V-Large2
if [ ! -e ./${TAR}/w2v_large_lv_fsh_swbd_cv.pt ];
then
   wget -q --show-progress -O ./${TAR}/w2v_large_lv_fsh_swbd_cv.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/w2v_large_lv_fsh_swbd_cv.pt
fi

# Wav2Vec 2.0 Large No finetuning
# W2V-Large1
if [ ! -e ./${TAR}/libri960_big.pt ];
then
    wget -q --show-progress -O ./${TAR}/libri960_big.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt
fi

# HuBERT Extra Large (~1B params)	Libri-Light 60k hr	No finetuning (Pretrained Model)
# Hubert
if [ ! -e ./${TAR}/hubert_xtralarge_ll60k.pt ];
then
    wget -q --show-progress -O ./${TAR}/hubert_xtralarge_ll60k.pt https://dl.fbaipublicfiles.com/hubert/hubert_xtralarge_ll60k.pt
fi
