import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_SIMPLE(nn.Module):
    def __init__(self, z_dim=2, input_dim=784, output_dim=784):
        super(VAE_SIMPLE, self).__init__()
        self.z_dim = z_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_var = nn.Linear(512, z_dim)

        # Decoder
        self.fc3 = nn.Linear(z_dim, 512)
        self.fc4 = nn.Linear(512, output_dim)

    def encoder(self, x):
        hidden = F.relu(self.fc1(x))
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def criterion(self, recon_img, img, mu, log_var, img_size=784):
        recons_loss = F.binary_cross_entropy(recon_img.view(-1, img_size), img.view(-1, img_size), reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recons_loss + kld_loss, recons_loss, kld_loss

