import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import lightning.pytorch as pl


class Trim(nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.size = image_size

    def forward(self, x):
        return x[:, :, :self.size, :self.size]
    

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


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, learning_rate=0.001):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = encoder.latent_size
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch

        z, z_mean, z_log_var = self.encoder(x)
        x_hat = self.decoder(z)

        pixelwise = torch.nn.functional.mse_loss(x_hat, x, reduction="none")
        pixelwise = pixelwise.view(batch.size(0), -1).sum(dim=1)
        pixelwise = pixelwise.mean()

        kl_div = -0.5 * torch.sum(
            1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1
        )  # sum over latent dimension  # type: ignore
        kl_div = kl_div.mean()  # average over batch dimension

        loss = pixelwise + kl_div

        # Logging to TensorBoard by default
        self.log("train_combined_loss", loss)
        self.log("pixelwise_loss", pixelwise)
        self.log("kl_div", kl_div)

        return loss

    def on_train_epoch_end(self):
        with torch.no_grad():
            rand_features = torch.randn(16, self.latent_size).to(self.device)
            generated_images = self.decoder(rand_features)
        grid = make_grid(generated_images, nrow=4, pad_value=1)
        
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)

        return super().on_train_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer