#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : record.py
# Create date : 2019-01-28 15:51
# Modified date : 2019-01-28 18:07
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os

def _get_param_str(config):
    # pylint: disable=bad-continuation
    param_str = "%s_%s_%s_%s_%s_%s_%s" % (
                                config["dataset"],
                                config["image_size"],
                                config["batch_size"],
                                config["number_of_generator_feature"],
                                config["number_of_discriminator_feature"],
                                config["size_of_z_latent"],
                                config["learn_rate"],
                                )
    # pylint: enable=bad-continuation
    return param_str

def get_check_point_path(config):
    param_str = _get_param_str(config)
    directory = "%s/save/%s/" % (config["data_path"], param_str)
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_check_point_file_full_path(config):
    path = get_check_point_path(config)
    param_str = _get_param_str(config)
    file_full_path = "%s%scheckpoint.tar" % (path, param_str)
    return file_full_path

def _write_output(config, con):
    save_path = get_check_point_path(config)
    file_full_path = "%s/output" % save_path
    f = open(file_full_path, "a")
    f.write("%s\n" %  con)
    f.close()

def record_dict(config, dic):
    save_status(config, "config:")
    for key in dic:
        dic_str = "%s : %s" % (key, dic[key])
        save_status(config, dic_str)

def save_status(config, con):
    print(con)
    _write_output(config, con)

def print_status(step_time, take_time, epoch, i, errD, errG, D_x, D_G_z1, D_G_z2, config, dataloader):
    num_epochs = config["num_epochs"]
    # pylint: disable=bad-continuation
    print_str = '[%d/%d]\t[%d/%d]\t Loss_D: %.4f\t Loss_G: %.4f\t D(x): %.4f\t D(G(z)): %.4f / %.4f take_time: %.fs' % (
                                                        epoch,
                                                        num_epochs,
                                                        i,
                                                        len(dataloader),
                                                        errD.item(),
                                                        errG.item(),
                                                        D_x,
                                                        D_G_z1,
                                                        D_G_z2,
#                                                        step_time,
                                                        take_time,
                                                        )
    # pylint: enable=bad-continuation
    save_status(config, print_str)
