import torch
from torch import nn
from torch.nn import functional as F


class MLP_Encoder(nn.Module):

    def __init__(self, input_dim, z_dim, dim_list):
        super(MLP_Encoder, self).__init__()
        assert type(dim_list) == list
        self.input_linear = nn.Linear(28*28, dim_list[0])
        self.encoder = self.build_encoder(dim_list)
        self.mu_linear = nn.Linear(dim_list[-1], z_dim)
        self.var_linear = nn.Linear(dim_list[-1], z_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.relu(self.input_linear(x))
        latent = self.encoder(x)
        mu = self.relu(self.mu_linear(latent))
        var = self.relu(self.var_linear(latent))
        return mu, var

    def build_encoder(self, dim_list):
        modules = []
        for i in range(len(dim_list) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(dim_list[i], dim_list[i + 1]),
                    nn.LeakyReLU()
                )
            )
        return nn.Sequential(*modules)


class MLP_Decoder(nn.Module):
    def __init__(self, output_dim, z_dim, dim_list):
        super(MLP_Decoder, self).__init__()
        self.linear1 = nn.Linear(z_dim, dim_list[0])
        self.output_linear = nn.Linear(dim_list[-1], 28*28)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.decoder = self.build_decoder(dim_list)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        out = self.decoder(x)
        out = self.sigmoid(self.output_linear(out))
        out = out.view(-1, 1, 28, 28)
        return out

    def build_decoder(self, dim_list):
        modules = []
        for i in range(len(dim_list) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(dim_list[i], dim_list[i + 1]),
                    nn.ReLU()
                )
            )
        return nn.Sequential(*modules)

    
class MLP_VAE(nn.Module):

    def __init__(self, input_dim, output_dim, z_dim, encoder_list, decoder_list):
        super(MLP_VAE, self).__init__()
        assert type(encoder_list) == list
        assert type(decoder_list) == list
        self.encoder = MLP_Encoder(input_dim, z_dim, encoder_list)
        self.decoder = MLP_Decoder(output_dim, z_dim, decoder_list)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var

    def criterion(self, recon_img, img, mu, log_var, img_size=784):
        recons_loss = F.binary_cross_entropy(recon_img.view(-1, img_size), img.view(-1, img_size), reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recons_loss + kld_loss, recons_loss, kld_loss



