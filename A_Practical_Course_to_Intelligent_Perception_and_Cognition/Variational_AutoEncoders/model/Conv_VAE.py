import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_VAE(nn.Module):
    def __init__(self, z_dim, input_dim, output_dim):
        super().__init__()
        self.z_dim = z_dim
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(5*5*128, 512)
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_var = nn.Linear(512, z_dim)

        # Decoder
        self.fc2 = nn.Linear(z_dim, 5*5*128)
        self.convtrans1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=1)
        self.convtrans2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2)
        self.convtrans3 = nn.ConvTranspose2d(in_channels=32, out_channels=output_dim, kernel_size=3, stride=2, output_padding=1)


    def encoder(self, x):
        hidden = F.relu(self.conv1(x))
        hidden = F.relu(self.conv2(hidden))
        hidden = F.relu(self.conv3(hidden)).view(-1, 5*5*128)

        hidden = F.relu(self.fc1(hidden))
        mu = self.fc_mu(hidden)
        log_var = self.fc_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decoder(self, z):
        hidden = F.relu(self.fc2(z)).view(-1, 128, 5, 5)
        hidden = F.relu(self.convtrans1(hidden))
        hidden = F.relu(self.convtrans2(hidden))
        output = F.sigmoid(self.convtrans3(hidden))
        return output

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def criterion(self, recon_img, img, mu, log_var, img_size=784):
        recons_loss = F.binary_cross_entropy(recon_img.view(-1, img_size), img.view(-1, img_size), reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recons_loss + kld_loss, recons_loss, kld_loss

if __name__ == "__main__":
    model = Conv_VAE(z_dim=1)
    input = torch.randn(16, 1, 28, 28)
    output, mu, var = model(input)
    print(output.shape)
