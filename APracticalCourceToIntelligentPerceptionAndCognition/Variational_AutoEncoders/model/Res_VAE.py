import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):

    def __init__(self, input_dim, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 32, 3, stride=2, padding=1)
        self.shortcut = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, z_dim)
        self.fc2 = nn.Linear(64 * 7 * 7, z_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        residual = F.relu(self.shortcut(x))
        x = self.pool(self.conv2(x))
        x = F.relu(x + residual)
        x = x.view(-1, 64 * 7 * 7)
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, output_dim, z_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, 64 * 7 * 7)
        self.upsample = nn.Upsample(scale_factor=2)
        self.trans_conv1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1)
        self.trans_conv2 = nn.ConvTranspose2d(32, output_dim, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 64, 7, 7)
        residual = self.upsample(x)
        x = F.relu(self.trans_conv1(x))
        x = F.relu(self.conv1(x + residual))
        x = torch.sigmoid(self.trans_conv2(x))
        return x


class Res_VAE(nn.Module):
    def __init__(self, z_dim, input_dim, output_dim):
        super(Res_VAE, self).__init__()
        self.encoder = Encoder(input_dim, z_dim)
        self.decoder = Decoder(output_dim, z_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def criterion(self, recon_img, img, mu, log_var, img_size=784):
        recons_loss = F.binary_cross_entropy(recon_img.view(-1, img_size), img.view(-1, img_size), reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recons_loss + kld_loss, recons_loss, kld_loss

if __name__ == '__main__':
    input = torch.rand((4, 1, 28, 28))
    Net = Res_VAE(z_dim=2, input_dim=1, output_dim=1)
    output, mu, var = Net(input)
    print(output.shape)

