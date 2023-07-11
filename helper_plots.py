import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


def show_image(image: torch.Tensor | np.ndarray, title='', figsize=(2.5, 2.5), save_fig=False):
    """ Show image from tensor 
    
    Params:\n
    image - The the source tensor
    title - The string for title of the figure
    figsize - The size in dpi
    save_fig - Is it need to save picture
    """
    assert len(image.shape) == 3
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0)
    height, width, channels = image.shape

    if image.max() < 1:
        image *= 255

    plt.figure(figsize=figsize)
    if channels == 1:
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(f'fig_{title}.png')
    plt.show()
    

def show_grid_samples(model, n_images: int, file_name="", save_dir=os.getcwd()):
    latent_size = model.latent_size
    with torch.no_grad():
        rand_features = torch.randn(n_images, latent_size).to(model.device)
        generated_images = model.decoder(rand_features)

    grid = make_grid(generated_images, nrow=int(np.sqrt(n_images)), pad_value=1)
    grid = transforms.ToPILImage()(grid)
    grid.show()

    if file_name:
        path = f"{save_dir}/{file_name}.png"
        grid.save(path)

    return generated_images


def show_images_bar(images: list | torch.Tensor | np.ndarray, cmap="gray", titles=[]):
    n_images = len(images)
    _, axes = plt.subplots(1, n_images)
    axes = np.array(axes)

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()

        assert images.dim() == 4
        images = images.permute(0, 2, 3, 1)

    if isinstance(images, list):
        images = np.array(images)

    for ax, img in zip(axes.flat, images):
        if img.shape[-1] == 1:
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
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


def create_training_animation(animation_name: str, images_dir: str):
    frames = []

    filenames = list(filter(lambda name: name.endswith("png"), os.listdir(images_dir)))
    filenames.sort(key=lambda x: int(x[5:-4]))
    for file in filenames:
        frame = Image.open(f"{images_dir}/{file}")
        frames.append(frame)

    frames[0].save(
        f"{animation_name}.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=100,
        loop=0,
    )


def interpolation(
    vae, x1: torch.Tensor, x2: torch.Tensor, n_steps: int
) -> torch.Tensor:
    assert len(x1.shape) == len(x2.shape) == 3
    x1 = x1.unsqueeze(0)
    x2 = x2.unsqueeze(0)
    x = torch.cat((x1, x2)).to(vae.device)

    encoded = vae.encoding_fn(x)
    z1 = encoded[0]
    z2 = encoded[1]

    z = torch.stack([z1 + (z2 - z1) * t for t in np.linspace(0, 1, n_steps)])
    interpolate_list = vae.decoder(z)
    interpolate_list = interpolate_list.to("cpu")

    return interpolate_list


def show_interpolation(interp_list: torch.Tensor, n_steps: int, animation_name=""):
    grid = make_grid(interp_list, nrow=n_steps)
    grid = transforms.ToPILImage()(grid)
    grid.show()

    if animation_name:
        frames = []
        for stage in interp_list:
            stage = transforms.ToPILImage()(stage)
            frames.append(stage)

        frames[0].save(
            f"{animation_name}.gif",
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0,
        )