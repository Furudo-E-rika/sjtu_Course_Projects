import os

import torchvision.utils
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from dataloader.dataloader import MNIST_Dataloader

CONFIG_PATH = "config/config.yaml"

with open(CONFIG_PATH, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

## Load MNIST Datasets
train_loader, test_loader = MNIST_Dataloader(batch_size=config["dataloader_batch"])

## Build Model
torch.manual_seed(config['seed'])

if config['train']['model'] == "conv_vae":
    model = Conv_VAE(z_dim=config['train']['z_dim'], input_dim=config['train']['input_dim'], output_dim=config['train']['output_dim'])
elif config['train']['model'] == "res_vae":
    model = Res_VAE(z_dim=config['train']['z_dim'], input_dim=config['train']['input_dim'], output_dim=config['train']['output_dim'])
elif config['train']['model'] == "mlp_vae":
    model = MLP_VAE(input_dim=config['train']['input_dim'], output_dim=config['train']['output_dim'],
                        z_dim=config['train']['z_dim'], encoder_list=config['train']['encoder_list'],
                        decoder_list=config['train']['decoder_list'])
elif config['train']['model'] == "simple_vae":
    model = VAE_SIMPLE(z_dim=config['train']['z_dim'], input_dim=config['train']['input_dim'], output_dim=config['train']['output_dim'])
else:
    raise TypeError("unrecognized model type")

## Training Process

writer = SummaryWriter(config['log_dir'])
CHECKPOINT_DIR = config['check_dir']
if not os.path.isdir(os.path.join(CHECKPOINT_DIR)):
    os.makedirs(CHECKPOINT_DIR+'_dim{}'.format(config['train']['z_dim']))

## Define Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
model = model.to(device)
print("Let's use {} to train VAE model".format(device))

## Define Optimizer
if config['train']['optimizer'] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['train']['optimAdam']['learningRate'],
                                 weight_decay=config['train']['optimAdam']['weightDecay'])

elif config['train']['optimAdam'] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config['train']['optimSgd']['learningRate']),
                                momentum=float(config['train']['optimSgd']['momentum']),
                                weight_decay=float(config['train']['optimSgd']['weight_decay']))
else:
    raise ValueError(
        'Invalid optimizer from config file: "{}". optimizers are ["Adam", "SGD"]'.format(
            config['train']['optimizer']))

total_iter_num = 0

## Training Cycles
for epoch in range(config['train']['epoch']):

    writer.add_scalar('data/epoch_number', epoch, total_iter_num)
    print('\n\nEpoch {}/{}'.format(epoch, config['train']['epoch']-1))
    print('Train:')
    print('=' * 20)

    total_loss = 0.
    total_recons_loss = 0.
    total_kld_loss = 0.

    ## training
    model.train()
    for iter_num, batch in enumerate(tqdm(train_loader)):

        total_iter_num += 1
        img, label = batch
        img = img.to(device)  # (B, 1, 28, 28)

        optimizer.zero_grad()
        recons_img, mu, var = model(img)

        # Calculate criterion
        loss, recons_loss, kld_loss = model.criterion(recons_img, img, mu, var)
        total_loss += loss.item()
        total_recons_loss += recons_loss.item()
        total_kld_loss += kld_loss.item()

        # Update optimizer
        loss.backward()
        optimizer.step()

    num_samples = (len(train_loader))
    epoch_loss = total_loss / num_samples
    epoch_recons_loss = total_recons_loss / num_samples
    epoch_kld_loss = total_kld_loss / num_samples
    print("Training Epoch {}".format(epoch))
    print("Epoch Loss: {}, Epoch Kld Loss:{}, Epoch Reconstruction Loss:{}".format(epoch_loss, epoch_kld_loss,
                                                                                   epoch_recons_loss))
    writer.add_scalar('data/Train Epoch Loss', epoch_loss, total_iter_num)
    writer.add_scalar('data/Train Epoch Reconstruction Loss', epoch_recons_loss, total_iter_num)
    writer.add_scalar('data/Train Epoch Kld Loss', epoch_loss, total_iter_num)

    # testing
    print('Test:')
    print('=' * 20)
    model.eval()

    total_loss = 0.
    total_recons_loss = 0.
    total_kld_loss = 0.
    img_tensor_list = []
    recons_img_tensor_list = []

    for iter_num, batch in enumerate(tqdm(test_loader)):

        img, label = batch
        img = img.to(device)

        with torch.no_grad():
            recons_img, mu, var = model(img)

            # Calculate criterion
            loss, recons_loss, kld_loss = model.criterion(recons_img, img, mu, var)
            total_loss += loss.item()
            total_recons_loss += recons_loss.item()
            total_kld_loss += kld_loss.item()

    num_samples = (len(test_loader))
    epoch_loss = total_loss / num_samples
    epoch_recons_loss = total_recons_loss / num_samples
    epoch_kld_loss = total_kld_loss / num_samples
    print("Test Epoch {}".format(epoch))
    print("Epoch Loss: {}, Epoch Kld Loss:{}, Epoch Reconstruction Loss:{}".format(epoch_loss, epoch_kld_loss,
                                                                                   epoch_recons_loss))
    writer.add_scalar('data/Test Epoch Loss', epoch_loss, total_iter_num)
    writer.add_scalar('data/Test Reconstruction Epoch Loss', epoch_recons_loss, total_iter_num)
    writer.add_scalar('data/Test Epoch Kld Loss', epoch_kld_loss, total_iter_num)

    if (epoch % config['train']['SaveEpoch']) == 0:
        filename = os.path.join(CHECKPOINT_DIR, 'checkpoint-epoch-{:04d}.pth'.format(epoch))
        model_params = model.state_dict()
        torch.save(
            {
                'model_state_dict': model_params,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'total_iter_num': total_iter_num,
                'epoch_loss': epoch_loss,
                'epoch_recons_loss': epoch_recons_loss,
                'epoch_kld_loss': epoch_kld_loss,
                'config': config
            }, filename)

        grid_image = torch.cat([recons_img.view(-1, 1, 28, 28)[:4].detach().cpu(), img.view(-1, 1, 28, 28)[:4].detach().cpu()], dim=3)
        grid_image = torchvision.utils.make_grid(grid_image, nrow=4)
        writer.add_image('Example', grid_image, total_iter_num)


writer.close()














