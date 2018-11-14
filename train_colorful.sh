#!/bin/bash

source activate zhang
#echo "Creating prior probabilities..."
#CUDA_VISIBLE_DEVICES=1 python tools/create_prior_probs.py --file datum/imglist/list.train.vae.txt

echo "Starting training now..."

CONF="conf/train.cfg"
RESULT_DIR="/work/ne63wog/zhang_noPriors180627/"
# PRIORS_FILE="resources/prior_probs_imgNet.npy"

mkdir -p $RESULT_DIR
rm results
ln -s $RESULT_DIR ./results

cp conf/train.cfg ./results/train.cfg
cp train_colorful.sh ./results/train_colorful.sh

CUDA_VISIBLE_DEVICES=1 python tools/train.py -c $CONF

echo "Done."
