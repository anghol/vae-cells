import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_size, latent_size, device):
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size
        self.device = device

        self.convolution_seria = nn.Sequential(
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
    
    def forward(self, x):
        x = self.convolution_seria(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        eps = torch.randn((z_mean.size(0), self.latent_size)).to(self.device)
        z = z_mean + eps * torch.exp(z_log_var / 2.0)
        return z, z_mean, z_log_var