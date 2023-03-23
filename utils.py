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