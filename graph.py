#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : graph.py
# Create date : 2019-01-24 17:17
# Modified date : 2019-01-28 17:46
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os
import time
import torch
import torchvision.utils as vutils
import model
import show
import record

class NNGraph(object):
    def __init__(self, dataloader, config):
        super(NNGraph, self).__init__()
        self.config = config
        self.train_model = self._get_train_model(config)
        record.record_dict(self.config, self.train_model["config"])
        self.config = self.train_model["config"]
        self.dataloader = dataloader

    def _get_train_model(self, config):
        train_model = model.init_train_model(config)
        train_model = self._load_train_model(train_model)
        return train_model

    def _save_train_model(self):
        model_dict = model.get_model_dict(self.train_model)
        file_full_path = record.get_check_point_file_full_path(self.config)
        torch.save(model_dict, file_full_path)

    def _load_train_model(self, train_model):
        file_full_path = record.get_check_point_file_full_path(self.config)
        if os.path.exists(file_full_path) and self.config["train_load_check_point_file"]:
            checkpoint = torch.load(file_full_path)
            train_model = model.load_model_dict(train_model, checkpoint)
        return train_model

    def _train_step(self, data):
        netG = self.train_model["netG"]
        optimizerG = self.train_model["optimizerG"]
        netD = self.train_model["netD"]
        optimizerD = self.train_model["optimizerD"]
        criterion = self.train_model["criterion"]
        device = self.config["device"]

        real_data = data[0].to(device)

        noise = model.get_noise(real_data, self.config)
        fake_data = netG(noise)
        label = model.get_label(real_data, self.config)

        errD, D_x, D_G_z1 = model.get_Discriminator_loss(netD, optimizerD, real_data, fake_data.detach(), label, criterion, self.config)
        errG, D_G_z2 = model.get_Generator_loss(netG, netD, optimizerG, fake_data, label, criterion, self.config)

        return errD, errG, D_x, D_G_z1, D_G_z2

    def _train_a_step(self, data, i, epoch):
        start = time.time()
        errD, errG, D_x, D_G_z1, D_G_z2 = self._train_step(data)
        end = time.time()
        step_time = end - start

        self.train_model["take_time"] = self.train_model["take_time"] + step_time

        print_every = self.config["print_every"]
        if i % print_every == 0:
            record.print_status(step_time*print_every,
                                self.train_model["take_time"],
                                epoch,
                                i,
                                errD,
                                errG,
                                D_x,
                                D_G_z1,
                                D_G_z2,
                                self.config,
                                self.dataloader)
        return errD, errG

    def _DCGAN_eval(self):
        fixed_noise = self.train_model["fixed_noise"]
        with torch.no_grad():
            netG = self.train_model["netG"]
            fake = netG(fixed_noise).detach().cpu()
            return fake

    def _save_generator_images(self, iters, epoch, i):
        num_epochs = self.config["num_epochs"]
        save_every = self.config["save_every"]
        img_list = self.train_model["img_list"]

        if (iters % save_every == 0) or ((epoch == num_epochs-1) and (i == len(self.dataloader)-1)):
            fake = self._DCGAN_eval()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            self._save_train_model()

    def _train_iters(self):
        num_epochs = self.config["num_epochs"]
        G_losses = self.train_model["G_losses"]
        D_losses = self.train_model["D_losses"]
        iters = self.train_model["current_iters"]
        start_epoch = self.train_model["current_epoch"]

        for epoch in range(start_epoch, num_epochs):
            self.train_model["current_epoch"] = epoch
            for i, data in enumerate(self.dataloader, 0):
                errD, errG = self._train_a_step(data, i, epoch)
                G_losses.append(errG.item())
                D_losses.append(errD.item())
                iters += 1
                self.train_model["current_iters"] = iters
                self._save_generator_images(iters, epoch, i)

    def train(self):
        self._train_iters()
        show.show_images(self.train_model, self.config, self.dataloader)
