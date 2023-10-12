import os
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from torchvision.utils import save_image, make_grid


CONFIG_PATH = "./config/config.yaml"
RESULT_PATH = "./plot"

with open(CONFIG_PATH, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

if config['visual']['model'] == "conv_vae":
    model = Conv_VAE(z_dim=config['visual']['z_dim'], input_dim=config['visual']['input_dim'], output_dim=config['visual']['output_dim'])
elif config['visual']['model'] == "res_vae":
    model = Res_VAE(z_dim=config['visual']['z_dim'], input_dim=config['visual']['input_dim'], output_dim=config['visual']['output_dim'])
elif config['visual']['model'] == "mlp_vae":
    model = MLP_VAE(input_dim=config['visual']['input_dim'], output_dim=config['visual']['output_dim'],
                        z_dim=config['visual']['z_dim'], encoder_list=config['visual']['encoder_list'],
                        decoder_list=config['visual']['decoder_list'])
elif config['visual']['model'] == "simple_vae":
    model = VAE_SIMPLE(z_dim=config['visual']['z_dim'], input_dim=config['visual']['input_dim'], output_dim=config['visual']['output_dim'])
else:
    raise TypeError("unrecognized model type")

CHECKPOINT = torch.load(config['visual']['checkpoint'], map_location='cpu')
model.load_state_dict(CHECKPOINT['model_state_dict'])
range_low, range_high = config['visual']['range_low'], config['visual']['range_high']
z_dim = config['visual']['z_dim']
model_type = config['visual']['model']
step = config['visual']['step']

# with torch.no_grad():
#    z = torch.randn(64, 2).cuda()
#    sample = model.decoder(z)
#    save_image(sample.view(64, 1, 28, 28), './samples/sample_' + '.png')

with torch.no_grad():

    input = []

    if config['visual']['z_dim'] == 2:
        for i in range(21):
            for j in range(21):
                z_tensor = torch.tensor([0.5 * i - 5, 0.5 * j - 5])
                input.append(z_tensor)
    elif config['visual']['z_dim'] == 1:
        for i in range(210):
            z_tensor = torch.tensor([0.05 * i - 5])
            input.append(z_tensor)

    input = torch.stack(input).type(torch.float32)
    # print("input shape:",input.shape)

    recons_imgs = model.decoder(input).view(-1, 1, 28, 28)
    # print("reconstruction image shape", recons_imgs.shape)
    grid_img = make_grid(recons_imgs, nrow=21, padding=0)

    if not os.path.isdir(os.path.join(RESULT_PATH)):
        os.makedirs(RESULT_PATH)

    save_image(grid_img, os.path.join(RESULT_PATH, "{}-{}-dim-gaussian.png".format(model_type, z_dim)))


