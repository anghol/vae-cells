import os
from torch import Tensor
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset


def rename_images(suffix: str, path: str):
    """ Rename all images in necessary directory.\n

    Params:\n
    suffix - The suffix of filename e.g. cell_0.png where 'cell' is suffix.\n
    path - The path to the directory with images.
    """

    filename = os.listdir(path)[0]
    img_format = filename.split('.')[-1]

    for i, filename in enumerate(sorted(os.listdir(path))):
        old_name = os.path.join(path, filename)
        new_name = os.path.join(path, suffix + f'_{i}.{img_format}')
        os.rename(old_name, new_name)


class ImageDataset(Dataset):
    """ Custom dataset for images with necessary data.\n

    Params:\n
    img_dir - A directory with source images.\n
    suffix - The suffix of filename e.g. cell_0.png where 'cell' is suffix.\n
    transform - Transformations which apply to images.\n
    
    Return:
    image - torch.Tensor    
    """

    def __init__(self, img_dir: str, suffix: str, transform=None):
        self.img_dir = img_dir
        self.suffix = suffix
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx: int) -> Tensor:
        img_path = os.path.join(self.img_dir, f'{self.suffix}_{idx}.png')
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image)
        return image
        

class CellPadResize(object):
    """ Transform images with cell:
    1) padding to square form
    2) resize to necessary size

    Params:\n
    output_size - The size for transform to (output_size, output_size)

    Return:
    image - torch.Tensor   
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