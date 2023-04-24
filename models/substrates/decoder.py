import torch
import torch.nn as nn


class Trim(nn.Module):
    def __init__(self, image_size: tuple):
        super().__init__()
        self.height, self.width = image_size
    
    def forward(self, x):
        return x[:, :, :self.height, :self.width]
    

class Decoder(nn.Module):
    def __init__(self, image_size: tuple, latent_size):
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size

        self.convolution_transpose_seria = nn.Sequential(
            torch.nn.Linear(self.latent_size, 1280),
            nn.Unflatten(1, (32, 5, 8)),
            nn.ConvTranspose2d(32, 32, stride=(1, 1), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, stride=(3, 3), kernel_size=(4, 4), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 32, stride=(3, 3), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 16, stride=(3, 3), kernel_size=(4, 4), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(16, 1, stride=(1, 1), kernel_size=(4, 4), padding=0),
            Trim(self.image_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.convolution_transpose_seria(x)