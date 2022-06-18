#!/usr/bin/env bash

# train
python main.py --gpus=0,1 --num_bins=1+4 --test=False --alpha=4 --beta=4 --log_dir=debug
# test
python main.py --gpus=0,1 --num_bins=1+4 --test=True --alpha=4 --beta=4 --weights=mstp
