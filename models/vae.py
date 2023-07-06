import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import lightning.pytorch as pl


class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder, learning_rate=0.001):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_size = encoder.latent_size
        self.lr = learning_rate

    def training_step(self, batch, batch_idx, n_samples=16):
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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer