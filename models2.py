import os
import numpy as np
import math
import sys
from time import time
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt

class Opt(object):
    dim = 8
    n_epochs = 10000 
    batch_size = dim*4
    lr = 0.00005
    n_cpu = 1
    latent_dim = 100
    img_size = 128
    channels = 3
    n_critic = 5
    clip_value = 0.01
    sample_interval = 400

opt = Opt()
img_shape = (opt.channels, opt.img_size, opt.img_size)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

    def save_model(self,path):
        torch.save(self.state_dict(),path)
        print("Model saved")

    def load_model(self,path):
        self.load_state_dict(torch.load(path,map_location=device))
        if torch.cuda.is_available(): self.cuda()
        print("Model loaded")

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3*128*128, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )
    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

    def save_model(self,path):
        torch.save(self.state_dict(),path)
        print("Model saved")

    def load_model(self,path):
        self.load_state_dict(torch.load(path,map_location=device))
        if torch.cuda.is_available(): self.cuda()
        print("Model loaded")
