#!/bin/bash


SRC=./project-ood2021

# AM-softmax models
TARDIR=AM-softmax-maxprob/config_train_asvspoof2019/01
SRCDIR=${SRC}/energy/c1_amsoftmax_softmaxscore/config_train_asvspoof2019/01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained


# AM-softmax energy-based
TARDIR=AM-softmax-energy/config_train_asvspoof2019/01
SRCDIR=${SRC}/energy/c1_amsoftmax/config_train_asvspoof2019/01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained

# AM-softmx confidence 
TARDIR=AM-softmax-conf/config_train_asvspoof2019/01
SRCDIR=${SRC}/confidence/c1_lambda_update_4/config_train_asvspoof2019/01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained


# plain-softmax Maxprob
TARDIR=Softmax-maxprob/config_train_asvspoof2019/01
SRCDIR=${SRC}/energy/c1_softmaxscore/config_train_asvspoof2019//01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained


# plain-softmax energy
TARDIR=Softmax-energy/config_train_asvspoof2019/01
SRCDIR=${SRC}/energy/c1_baseline/config_train_asvspoof2019/01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained


# plain-softmax confidence
TARDIR=Softmax-conf/config_train_asvspoof2019/01
SRCDIR=${SRC}/confidence/c1_lambda_update_3/config_train_asvspoof2019/01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained


# plain-softmax energy on train set T2
TARDIR=Softmax-energy/config_train_asvspoof2019_esp/01
SRCDIR=${SRC}/energy/c1_inout_1/config_train_asvspoof2019_esp/01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained

# plain-softmax energy on train set T3
TARDIR=Softmax-energy/config_train_asvspoof2019_bc10/01
SRCDIR=${SRC}/energy/c1_inout_1/config_train_asvspoof2019_bc10/01

mkdir -p ${TARDIR}/__pretrained
cp ${SRCDIR}/*.py ${TARDIR}/
cp ${SRCDIR}/*.sh ${TARDIR}/
cp ${SRCDIR}/trained_network.pt ${TARDIR}/__pretrained
