# ------------------------------------------------------------
# Real-Time Style Transfer Implementation
# @misc{chengbinjin2018realtimestyletransfer,
#     author = {Cheng-Bin Jin},
#     title = {Real-Time Style Transfer},
#     year = {2018},
#     howpublished = {\url{https://github.com/ChengBinJin/Real-time-style-transfer/}},
#     note = {commit xxxxxxx}
#   }
# Written by Logan Engstrom
# ------------------------------------------------------------
#! /bin/bash

mkdir data
cd data
wget http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
unzip train2014.zip
unzip val2014.zip
