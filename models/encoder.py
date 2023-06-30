import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, image_size, latent_size, in_out_channels, kernels, strides, pads):
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size

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
            nn.Flatten() # (N, 64, 7, 7) -> (N, 3136)
        )
        
        self.z_mean = torch.nn.Linear(3136, self.latent_size)
        self.z_log_var = torch.nn.Linear(3136, self.latent_size)
    
    def forward(self, x):
        x = self.convolution_seria(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        eps = torch.randn((z_mean.size(0), self.latent_size)).to('cuda' if torch.cuda.is_available() else 'cpu')
        z = z_mean + eps * torch.exp(z_log_var / 2.0)
        return z, z_mean, z_log_var