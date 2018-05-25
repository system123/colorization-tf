#!/bin/bash

source activate zhang
echo "Creating prior probabilities..."
#CUDA_VISIBLE_DEVICES=1 python tools/create_prior_probs.py --file datum/imglist/list.train.vae.txt

echo "Starting training now..."

CONF="conf/train.cfg"

CUDA_VISIBLE_DEVICES=1 python tools/train.py -c $CONF

echo "Done."
