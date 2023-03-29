import os, random, torch
import numpy as np
from matplotlib import pyplot as plt

def show_image(image, title=None, figsize=(6, 3), save_fig=False):
    """ Show image from FloatTensor 
    
    Params:
    image - the source tensor
    title - string for title of the figure
    figsize - size in dpi
    save_fig - is it need to save picture
    """
    
    _, height, width = image.shape
    plt.figure(figsize=figsize)
    plt.imshow(image.reshape(height, width), cmap='gray')
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(f'fig_{title}.png')
    plt.show()


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def vae_logging(batch_idx: int, logging_interval: int, log_dict: dict, losses: list):
    loss, pixelwise, kl_div = losses

    log_dict["train_combined_loss_per_batch"].append(loss)
    log_dict["train_reconstruction_loss_per_batch"].append(pixelwise)
    log_dict["train_kl_loss_per_batch"].append(kl_div)

    if not batch_idx % logging_interval:
            print(
                "Epoch: %03d/%03d | Batch %04d/%04d | Loss: %.4f"
                % (epoch + 1, NUM_EPOCHS, batch_idx, len(train_loader), loss)
            )

