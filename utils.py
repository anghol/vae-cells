from matplotlib import pyplot as plt

def show_image(image, title=None, figsize=(6, 3), save_fig=False):
    _, height, width = image.shape
    plt.figure(figsize=figsize)
    plt.imshow(image.reshape(height, width), cmap='gray', vmin=0, vmax=255)
    if title:
        plt.title(title)
    if save_fig:
        plt.savefig(f'fig_{title}.png')
    plt.show()

