import torch
import torch.nn as nn


class Trim(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.size = image_size

    def forward(self, x):
        return x[:, :, :self.size, :self.size]
    

class Decoder(nn.Module):
    def __init__(self, image_size, latent_size):
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size

        self.convolution_transpose_seria = nn.Sequential(
            # if image_size = (56, 56): (N, 64, 7, 7) and (N, 3136)
            # if image_size = (64, 64): (N, 64, 8, 8) and (N, 4096)
            torch.nn.Linear(self.latent_size, 4096),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 64, stride=(1, 1), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 64, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(64, 32, stride=(2, 2), kernel_size=(3, 3), padding=0),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(32, 1, stride=(1, 1), kernel_size=(3, 3), padding=0),
            Trim(self.image_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.convolution_transpose_seria(x)