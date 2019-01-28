#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : show.py
# Create date : 2019-01-24 17:19
# Modified date : 2019-01-28 17:31
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from matplotlib import  rcParams

from matplotlib.animation import ImageMagickWriter

import record

rcParams["animation.embed_limit"] = 500

def show_some_batch(real_batch,device):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()

def _plot_real_and_fake_images(real_batch, device, img_list, save_path):
    # Plot the real images
    plt.figure(figsize=(30, 30))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    name = "real_and_fake.jpg"
    full_path_name = "%s/%s" % (save_path, name)
    plt.savefig(full_path_name)
    #plt.show()

def _show_generator_images(G_losses, D_losses, save_path):
    plt.figure(figsize=(40, 20))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    name = "G_D_losses.jpg"
    full_path_name = "%s/%s" % (save_path, name)
    plt.savefig(full_path_name)
    #plt.show()

def _show_img_list(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())
    plt.show()

def _save_img_list(img_list, save_path, config):
    #_show_img_list(img_list)
    metadata = dict(title='generator images', artist='Matplotlib', comment='Movie support!')
    writer = ImageMagickWriter(fps=1,metadata=metadata)
    ims = [np.transpose(i, (1, 2, 0)) for i in img_list]
    fig, ax = plt.subplots()
    with writer.saving(fig, "%s/img_list.gif" % save_path,500):
        for i in range(len(ims)):
            ax.imshow(ims[i])
            ax.set_title("step {}".format(i * config["save_every"]))
            writer.grab_frame()

def show_images(train_model, config, dataloader):
    G_losses = train_model["G_losses"]
    D_losses = train_model["D_losses"]
    img_list = train_model["img_list"]
    save_path = record.get_check_point_path(config)

    _show_generator_images(G_losses, D_losses, save_path)
    _save_img_list(img_list,save_path,config)
    real_batch = next(iter(dataloader))
    _plot_real_and_fake_images(real_batch, config["device"], img_list, save_path)

