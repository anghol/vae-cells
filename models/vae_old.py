import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.size = image_size

    def forward(self, x):
        return x[:, :, :self.size, :self.size]


class VAE(nn.Module):
    def __init__(self, device: torch.device, image_size: int, latent_size=2):
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.device = device

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.image_size**2, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.01),
        )

        self.z_mean = torch.nn.Linear(256, self.latent_size)
        self.z_log_var = torch.nn.Linear(256, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 1024),
            nn.LeakyReLU(0.01),
            nn.Linear(1024, self.image_size**2),
            Reshape(-1, 1, self.image_size, self.image_size),
            nn.Sigmoid(),
        )

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.0)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
    

class ConvolutionalVAE(nn.Module):
    def __init__(self, device: torch.device, image_size: int, latent_size=2):
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            # if image_size = (56, 56): (N, 64, 7, 7) -> (N, 3136)
            # if image_size = (64, 64): (N, 64, 8, 8) -> (N, 4096)
            nn.Flatten()
        )

        self.z_mean = torch.nn.Linear(4096, self.latent_size)
        self.z_log_var = torch.nn.Linear(4096, self.latent_size)

        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, 4096),
            Reshape(-1, 64, 8, 8),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
            Trim(self.image_size),  # 1x57x57 -> 1x56x56
                                    # 1x65x65 -> 1x64x64
            nn.Sigmoid()
        )

    def encoding_fn(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(self.device)
        z = z_mu + eps * torch.exp(z_log_var / 2.0)
        return z

    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded