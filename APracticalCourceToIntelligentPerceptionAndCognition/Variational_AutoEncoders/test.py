import os
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from dataloader.dataloader import MNIST_Dataloader

device = torch.device("cuda")
net = VAE_Net(hidden_shape=[512, 7, 7], latent_dim=1).to(device)
input = torch.rand(16, 1, 28, 28).to(device)
output, mu, var = net(input)
print(output.shape)
print(net.criterion(output, input, mu, var))