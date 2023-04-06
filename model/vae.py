import torch
import torch.nn as nn
import lightning.pytorch as pl

class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        # x = x.view(x.size(0), -1)

        z, z_mean, z_log_var = self.encoder(x)
        x_hat = self.decoder(z)

        pixelwise = nn.functional.mse_loss(x_hat, x)
        kl_div = -0.5 * torch.sum(
            1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1
        )  # sum over latent dimension  # type: ignore
        loss = pixelwise + kl_div

        # Logging to TensorBoard by default
        print(pixelwise.shape)
        print(kl_div.shape)
        print(loss.shape)
        self.log("train_combined_loss", loss, prog_bar=True)
        self.log("pixelwise_loss", pixelwise)
        self.log("kl_div", kl_div)

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


class Encoder(nn.Module):
    def __init__(self, image_size, latent_size):
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
            nn.Flatten()
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


class Decoder(nn.Module):
    def __init__(self, image_size, latent_size):
        super().__init__()
        self.image_size = image_size
        self.latent_size = latent_size

        self.convolution_transpose_seria = nn.Sequential(
            torch.nn.Linear(self.latent_size, 3136),
            nn.Unflatten(1, (64, 7, 7)),
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
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.convolution_transpose_seria(x)


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
            nn.Flatten()
        )

        self.z_mean = torch.nn.Linear(3136, self.latent_size)
        self.z_log_var = torch.nn.Linear(3136, self.latent_size)

        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, 3136),
            Reshape(-1, 64, 7, 7),
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