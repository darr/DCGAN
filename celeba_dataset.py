#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : celeba_dataset.py
# Create date : 2019-01-24 18:02
# Modified date : 2019-01-26 22:57
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_dataloader(config):
    image_size = config["image_size"]
    batch_size = config["batch_size"]
    dataroot = config["data_path"]
    workers = config["workers"]

    tf = transform=transforms.Compose([
           transforms.Resize(image_size),
           transforms.CenterCrop(image_size),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ])

    dataset = dset.ImageFolder(root=dataroot, transform=tf)
    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=workers)
    return dataloader
