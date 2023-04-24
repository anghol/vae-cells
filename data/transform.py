import torchvision.transforms as transforms


class CellPadResize(object):
    """ Transform images with cell:
    1) padding to square form
    2) resize to necessary size

    Params:
    output_size - size for transform to (output_size, output_size)

    Return:
    image - tourch.Tensor   
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        width, height = image.size
        if height > width:
            pad_value = (height - width) // 2
            padding = transforms.Pad(padding=(pad_value, 0))
            image = padding(image)
        elif width > height:
            pad_value = (width - height) // 2
            padding = transforms.Pad(padding=(0, pad_value))
            image = padding(image)

        resize = transforms.Resize((self.output_size, self.output_size))
        image = resize(image)

        return image