#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : model.py
# Create date : 2019-01-24 17:00
# Modified date : 2019-01-29 00:43
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import random
import torch
import torch.nn as nn
import torch.optim as optim

from DCGAN_architecture import Generator, Discriminator

import record

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def _random_init(config):
    manualSeed = config["manual_seed"]
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

def _get_a_net(Net, config):
    ngpu = config["number_gpus"]
    device = config["device"]
    net = Net(config).to(device)
    if (device.type == 'cuda') and (ngpu > 1):
        net = nn.DataParallel(net, list(range(ngpu)))
    net.apply(_weights_init)

    record.save_status(config, net)
    return net

def _get_optimizer(net, config):
    lr = config["learn_rate"]
    beta1 = config["beta1"]
    opt = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
    return opt

def _get_fixed_noise(config):
    nz = config["size_of_z_latent"]
    device = config["device"]
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    return fixed_noise

def load_model_dict(train_model, checkpoint):
    train_model["netG"].load_state_dict(checkpoint["netG"])
    train_model["netD"].load_state_dict(checkpoint["netD"])
    train_model["criterion"].load_state_dict(checkpoint["criterion"])
    train_model["optimizerD"].load_state_dict(checkpoint["optimizerD"])
    train_model["optimizerG"].load_state_dict(checkpoint["optimizerG"])

    train_model["fixed_noise"] = checkpoint["fixed_noise"]
    train_model["G_losses"] = checkpoint["G_losses"]
    train_model["D_losses"] = checkpoint["D_losses"]
    train_model["img_list"] = checkpoint["img_list"]
    train_model["current_iters"] = checkpoint["current_iters"]
    train_model["current_epoch"] = checkpoint["current_epoch"]
    train_model["config"] = checkpoint["config"]
    train_model["take_time"] = checkpoint["take_time"]
    return train_model

def get_model_dict(train_model):
    model_dict = {}
    model_dict["netG"] = train_model["netG"].state_dict()
    model_dict["netD"] = train_model["netD"].state_dict()
    model_dict["criterion"] = train_model["criterion"].state_dict()
    model_dict["optimizerD"] = train_model["optimizerD"].state_dict()
    model_dict["optimizerG"] = train_model["optimizerG"].state_dict()

    model_dict["fixed_noise"] = train_model["fixed_noise"]
    model_dict["G_losses"] = train_model["G_losses"]
    model_dict["D_losses"] = train_model["D_losses"]
    model_dict["img_list"] = train_model["img_list"]
    model_dict["current_iters"] = train_model["current_iters"]
    model_dict["current_epoch"] = train_model["current_epoch"]
    model_dict["config"] = train_model["config"]
    model_dict["take_time"] = train_model["take_time"]

    return model_dict


def init_train_model(config):
    _random_init(config)
    netG = _get_a_net(Generator, config)
    netD = _get_a_net(Discriminator, config)
    criterion = nn.BCELoss()
    optimizerD = _get_optimizer(netD, config)
    optimizerG = _get_optimizer(netG, config)

    fixed_noise = _get_fixed_noise(config)

    train_model = {}
    train_model["netG"] = netG
    train_model["netD"] = netD
    train_model["criterion"] = criterion
    train_model["optimizerD"] = optimizerD
    train_model["optimizerG"] = optimizerG
    train_model["fixed_noise"] = fixed_noise

    train_model["G_losses"] = []
    train_model["D_losses"] = []
    train_model["img_list"] = []
    train_model["current_iters"] = 0
    train_model["current_epoch"] = 0
    train_model["config"] = config
    train_model["take_time"] = 0.0

    return train_model

def _run_Discriminator(netD, data, label, loss):
    output = netD(data).view(-1)
    err = loss(output, label)
    err.backward()
    m = output.mean().item()
    return err, m

def get_Discriminator_loss(netD, optimizerD, real_data, fake_data, label, criterion, config):
    netD.zero_grad()
    errD_real, D_x = _run_Discriminator(netD, real_data, label, criterion)
    label.fill_(config["fake_label"])
    errD_fake, D_G_z1 = _run_Discriminator(netD, fake_data, label, criterion)
    errD = errD_real + errD_fake
    optimizerD.step()
    return errD, D_x, D_G_z1

def get_Generator_loss(netG, netD, optimizerG, fake_data, label, criterion, config):
    netG.zero_grad()
    label.fill_(config["real_label"])  # fake labels are real for generator cost
    errG, D_G_z2 = _run_Discriminator(netD, fake_data, label, criterion)
    optimizerG.step()
    return errG, D_G_z2

def get_label(data, config):
    b_size = data.size(0)
    real_label = config["real_label"]
    device = config["device"]
    label = torch.full((b_size, ), real_label, device=device)
    return label

def get_noise(data, config):
    b_size = data.size(0)
    device = config["device"]
    nz = config["size_of_z_latent"]
    noise = torch.randn(b_size, nz, 1, 1, device=device)
    return noise
