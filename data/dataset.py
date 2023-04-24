import os
from torch import Tensor
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset


class CellsImageDataset(Dataset):
    """ Custom dataset for images with cells

    Params:
    img_dir - a directory with source images
    transform - transformations which apply to images

    Return:
    image - tourch.Tensor    
    """

    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx: int) -> Tensor:
        img_path = os.path.join(self.img_dir, f'cell_{idx}.png')
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image)
        return image
        

class SubstratesImageDataset(Dataset):
    """ Custom dataset for images with substrates

    Params:
    img_dir - a directory with source images
    transform - transformations which apply to images
    
    Return:
    image - tourch.Tensor    
    """

    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx: int) -> Tensor:
        img_path = os.path.join(self.img_dir, f'part_substrate_{idx}.png')
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        if self.transform:
            image = self.transform(image)
        return image