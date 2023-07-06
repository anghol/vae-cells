import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

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


def show_image(image, title=None, save_fig=False):
    """ Show image from FloatTensor 
    
    Params:
    image - the source tensor
    title - string for title of the figure
    figsize - size in dpi
    save_fig - is it need to save picture
    """
    
    _, height, width = image.shape
    # plt.figure(figsize=figsize)
    plt.imshow(image.reshape(height, width), cmap='gray')
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(f'fig_{title}.png')
    plt.show()
    

def show_grid_samples(model, n_images: int, file_name="", save_dir=os.getcwd()):
    latent_size = model.latent_size
    with torch.no_grad():
        rand_features = torch.randn(n_images, latent_size)
        generated_images = model.decoder(rand_features)

    grid = make_grid(generated_images, nrow=int(np.sqrt(n_images)), pad_value=1)
    grid = transforms.ToPILImage()(grid)
    grid.show()

    if file_name:
        path = f"{save_dir}/{file_name}.png"
        grid.save(path)

    return generated_images


def show_images_bar(images: list[np.ndarray], cmap="gray", titles=[]):
    n_images = len(images)
    _, axes = plt.subplots(1, n_images)
    axes = np.array(axes)

    for ax, img in zip(axes.flat, images):
        if len(img.shape) > 2:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap)
        ax.set(xticks=[], yticks=[])

    if titles:
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title)


def generate_and_save_samples(samples_dir, n_samples, batch_size, model, image_size):
    latent_size = model.latent_size

    last_batch = n_samples % batch_size
    count = n_samples // batch_size if last_batch == 0 else n_samples // batch_size + 1
    idx = 0

    for step in range(count):
        with torch.no_grad():
            if step == count - 1 and last_batch != 0:
                rand_features = torch.randn(last_batch, latent_size)
            else:
                rand_features = torch.randn(batch_size, latent_size)
            samples = model.decoder(rand_features)

        for sample in samples:
            save_image(sample, f"{samples_dir}/sample_{idx}.png")
            idx += 1