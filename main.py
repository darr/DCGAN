#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : main.py
# Create date : 2019-01-25 14:07
# Modified date : 2019-01-27 22:36
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import celeba_dataset
from etc import config
from graph import NNGraph

def run():
    dataloader = celeba_dataset.get_dataloader(config)
    g = NNGraph(dataloader, config)
    g.train()

run()
