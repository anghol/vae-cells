import torch
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt

def plot_all_losses(log_dict: dict):
    # defined losses for plotting
    loss_names = list(log_dict.keys())
    for name in loss_names:
        if len(log_dict[name]) == 0:
            loss_names.remove(name)

    n = len(loss_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    fig.suptitle('Training losses')
    for ax, name in zip(axes, loss_names):
        ax.plot(log_dict[name])
        ax.set(title=name, xlabel="Iterations", ylabel="Loss")


def plot_original_and_decoded(model, data_loader, n_images, device):
    features = next(iter(data_loader))
    features = features.to(device)
    _, color_channels, image_height, image_width = features.shape

    orig_images = features[:n_images]
    with torch.no_grad():
        decoded_images = model(orig_images)[-1]

    fig, axes = plt.subplots(
        2, n_images, sharex=True, sharey=True, figsize=(2 * n_images, 4.5)
    )
    fig.suptitle("Original and decoded images")
    titles = ["original", "decoded"]

    for j in range(n_images):
        for ax, images, title in zip(axes, [orig_images, decoded_images], titles):
            curr_image = images[j].detach().to("cpu")
            if color_channels > 1:
                ax[j].imshow(np.transpose(curr_image, (1, 2, 0)))
            else:
                curr_image = curr_image.reshape((image_height, image_width))
                ax[j].imshow(curr_image, cmap="gray")
            ax[j].set(title=" ".join([title, str(j + 1)]))


def generate_and_plot(model, n_images, device):
    latent_size = model.latent_size
    with torch.no_grad():
        rand_features = torch.randn(n_images, latent_size).to(device)
        generated_images = model.decoder(rand_features)

    grid = make_grid(generated_images, nrow=int(np.sqrt(n_images)))
    grid = grid.detach().to(torch.device("cpu"))
    grid = np.transpose(grid, (1, 2, 0))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])