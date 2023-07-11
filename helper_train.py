import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance


def train(
        num_epochs: int,
        model: nn.Module,
        optimizer,
        device,
        train_loader,
        logging_interval: int,
        reconstruction_term_weight=1
    ):
    
    loss_fn = nn.functional.mse_loss

    log_dict = {'train_combined_loss_per_batch': [],
                'train_combined_loss_per_epoch': [],
                'train_reconstruction_loss_per_batch': [],
                'train_kl_loss_per_batch': []}

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()

        for batch_idx, features in enumerate(train_loader):
            all_losses = training_step(features, model, device, optimizer,
                                       loss_fn, reconstruction_term_weight)

            logging(log_dict, all_losses, logging_interval,
                    epoch, num_epochs, batch_idx, train_loader)
    
        print('Time elapsed: {:.2f} min'.format((time.time() - start_time) / 60))
    
    print('Total training time: {:.2f} min'.format((time.time() - start_time) / 60))

    return log_dict


def training_step(features, model, device, optimizer, loss_fn, reconstruction_term_weight):
    features = features.to(device)
    
    # forward and backpropogation
    encoded, z_mean, z_log_var, decoded = model(features)

    # total loss = reconstruction loss + KL divergence
    # kl_divergence = (0.5 * (z_mean**2 + torch.exp(z_log_var) - z_log_var - 1)).sum()

    kl_div = -0.5 * torch.sum(
        1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1
    )  # sum over latent dimension  # type: ignore

    batch_size = kl_div.size(0)
    kl_div = kl_div.mean()  # average over batch dimension

    pixelwise = loss_fn(decoded, features, reduction='none')
    pixelwise = pixelwise.view(batch_size, -1).sum(axis=1)  # sum over pixels  # type: ignore
    pixelwise = pixelwise.mean()  # average over batch dimension

    loss = reconstruction_term_weight * pixelwise + kl_div

    losses_dict = {'loss': loss.item(),
                  'pixelwise': pixelwise.item(),
                  'kl_div': kl_div.item()}

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return losses_dict


def logging(log_dict, losses_dict, logging_interval,
            epoch, num_epochs, batch_idx, train_loader):
    log_dict['train_combined_loss_per_batch'].append(losses_dict['loss'])
    log_dict['train_reconstruction_loss_per_batch'].append(losses_dict['pixelwise'])
    log_dict['train_kl_loss_per_batch'].append(losses_dict['kl_div'])

    if not batch_idx % logging_interval:
        print("Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f" 
              % (epoch + 1, num_epochs, batch_idx, len(train_loader), losses_dict['loss']))
        

def compute_epoch_loss_autoencoder(model, data_loader, loss_fn, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features in data_loader:
            features = features.to(device)
            decoded = model(features)[-1]
            loss = loss_fn(decoded, features, reduction='sum')
            num_examples += features.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def count_fid_metric(
    dataloader: DataLoader,
    model,
    image_size: int,
    latent_size: int,
    channels: int,
    min_n_samples=1000,
    n_feature=2048,
    normalize=False,
):
    """Compute the FID metric for evaluation of the diffusion models"""

    batch_size = next(iter(dataloader)).shape[0]
    count = int(np.ceil(min_n_samples / batch_size))

    real_imgs, fake_imgs = [], []

    start = time.time()
    for i in range(count):
        images = next(iter(dataloader))
        real_imgs.append(images)

        with torch.no_grad():
            rand_features = torch.randn(batch_size, latent_size)
            samples = model.decoder(rand_features)
            fake_imgs.append(samples)
    stop = time.time()

    real_imgs = torch.cat(real_imgs)
    real_imgs = real_imgs if normalize else (real_imgs * 255).to(torch.uint8)

    fake_imgs = np.concatenate(fake_imgs)
    fake_imgs = torch.tensor(fake_imgs)
    fake_imgs = fake_imgs if normalize else (fake_imgs * 255).to(torch.uint8)
    print(f"{fake_imgs.shape[0]} samples are generated, time = {stop - start :.3f} sec")

    if channels != 3:
        real_imgs = real_imgs.repeat(1, 3, 1, 1)
        fake_imgs = fake_imgs.repeat(1, 3, 1, 1)

    fid = FrechetInceptionDistance(feature=n_feature, normalize=normalize)
    start = time.time()
    fid.update(real_imgs, real=True)
    fid.update(fake_imgs, real=False)
    metric = fid.compute()
    stop = time.time()
    print(f"FID is computed, time = {stop - start :.3f} sec")

    return metric