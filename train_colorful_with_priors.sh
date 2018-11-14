#!/bin/bash

source activate zhang
#echo "Creating prior probabilities..."
#CUDA_VISIBLE_DEVICES=1 python tools/create_prior_probs.py --file datum/imglist/list.train.vae.txt

echo "Starting training now..."

CONF="conf/train.cfg"
RESULT_DIR="/work/ne63wog/zhang_SEN12_Priors180706/"
PRIORS_FILE="resources/prior_probs_michael.npy"

mkdir -p $RESULT_DIR
rm results
rm resources/prior_probs.npy

ln -s $RESULT_DIR ./results
ln $PRIORS_FILE ./resources/prior_probs.npy

cp conf/train.cfg ./results/train.cfg
cp train_colorful_with_priors.sh ./results/train_colorful.sh
cp utils.py ./results/utils.py

CUDA_VISIBLE_DEVICES=1 python tools/train.py -c $CONF

echo "Done."
