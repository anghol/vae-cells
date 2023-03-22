import os
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torch.utils.data import Dataset

class CellsImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.img_dir))
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f'cell_{idx}.png')
        image = read_image(img_path, mode=ImageReadMode.GRAY)
        if self.transform:
            print(type(image))
            image = self.transform(image)
        return image
        

class CellPadResize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        _, height, width = image.shape
        if height > width:
            pad_value = (height - width) // 2
            padding = transforms.Pad(padding=(pad_value, 0))
            image = padding(image)
        elif width > height:
            pad_value = (width - height) // 2
            padding = transforms.Pad(padding=(0, pad_value))
            image = padding(image)

        resize = transforms.Resize(self.output_size)
        image = resize(image)

        return image
    

def rename_cells(base_name, cells_path='data\separated_cells'):
    for i, file_name in enumerate(os.listdir(cells_path)):
        old_name = os.path.join(cells_path, file_name)
        new_name = os.path.join(cells_path, base_name + f'_{i}.png')
        os.rename(old_name, new_name)


