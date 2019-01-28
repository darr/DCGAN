#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : etc.py
# Create date : 2019-01-24 17:02
# Modified date : 2019-01-28 23:37
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch

config = {}

config["dataset"] = "celeba"
config["batch_size"] = 128
config["image_size"] = 64
config["num_epochs"] = 5
config["data_path"] = "data/%s" % config["dataset"]
config["workers"] = 2
config["print_every"] = 200
config["save_every"] = 500
config["manual_seed"] = 999
config["train_load_check_point_file"] = False
#config["manual_seed"] = random.randint(1, 10000) # use if you want new results

config["number_channels"] = 3
config["size_of_z_latent"] = 100
config["number_gpus"] = 1
config["number_of_generator_feature"] = 64
config["number_of_discriminator_feature"] = 64
config["learn_rate"] = 0.0002
config["beta1"] =0.5
config["real_label"] = 1
config["fake_label"] = 0
config["device"] = torch.device("cuda:0" if (torch.cuda.is_available() and config["number_gpus"] > 0) else "cpu")


