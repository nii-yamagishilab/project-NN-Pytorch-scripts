#!/usr/bin/python

# baseline
SRCDIR=~/WORK/WORK/WORK/project-ood2021/energy/c1_baseline
mkdir -p model-LFCC-LLGF/config_train_asvspoof2019/01/__pretrained
cp ${SRCDIR}/model.py model-LFCC-LLGF
cp ${SRCDIR}/config_train_asvspoof2019/01/trained_network.pt model-LFCC-LLGF/config_train_asvspoof2019/01/__pretrained

# Models

SRCDIR=~/WORK/WORK/WORK/pytorch-project-2020/proj-asvspoof-2/self-super-emb
TRAIN=config_train_asvspoof2019

SRCS=("ssl-trial01-1" "ssl-trial01-2" "ssl-trial01-3" "ssl-trial01-4-2" "ssl-trial01-6-2" "ssl-trial01-5-2" "ssl-trial02-1" "ssl-trial05-1" "ssl-trial04-1" "ssl-trial03-1")
TARS=("W2V-XLSR-fix-LLGF" "W2V-XLSR-fix-LGF" "W2V-XLSR-fix-GF" "W2V-XLSR-ft-LLGF" "W2V-XLSR-ft-LGF" "W2V-XLSR-ft-GF" "HuBERT-XL-fix-LLGF" "W2V-Large2-fix-LLGF" "W2V-Large1-fix-LLGF" "W2V-Small-fix-LLGF")

for idx in $(seq 0 9);
do
    TAR=model-${TARS[${idx}]}
    SRC=${SRCS[${idx}]}

    if [ -d ${SRCDIR}/${SRC} ];
    then
	echo ${TAR} ${SRC}
	mkdir -p ${TAR}/${TRAIN}/01/__pretrained
	cp ${SRCDIR}/${SRC}/model.py ${TAR}
	cp ${SRCDIR}/${SRC}/${TRAIN}/01/trained_network.pt ${TAR}/${TRAIN}/01/__pretrained
    fi
done

