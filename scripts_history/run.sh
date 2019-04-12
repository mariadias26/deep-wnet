#!/bin/bash
nohup python3 train_unet_generator.py &> train_resize.out&
nohup python3 predict.py &> predict.out&

nohup sh -c 'python3 train_unet.py >train.out && python3 predict.py >predict.out' &
