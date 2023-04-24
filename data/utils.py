import os
import cv2 as cv
import numpy as np
import utils.opencv as utilsCV
from utils.consts import *


def rename_cells(base_name: str, cells_path='data/separated_cells'):
    """ Rename all images with cells in necessary directory """
    
    for i, file_name in enumerate(os.listdir(cells_path)):
        old_name = os.path.join(cells_path, file_name)
        new_name = os.path.join(cells_path, base_name + f'_{i}.png')
        os.rename(old_name, new_name)


def cut_substrates(segmentation_dir, image_dir, substrate_dir, part_substrate_dir):
    """ Cut large substrates with black bacteria's areas """
    
    if not os.path.isdir(substrate_dir):
        os.mkdir(substrate_dir)
    if not os.path.isdir(part_substrate_dir):
        os.mkdir(part_substrate_dir)

    files = sorted(os.listdir(segmentation_dir))

    for file in files:
        color_mask = cv.imread(f"{segmentation_dir}/{file}")
        color_mask = cv.cvtColor(color_mask, cv.COLOR_BGR2RGB)
        gray_mask = cv.cvtColor(color_mask, cv.COLOR_BGR2GRAY)

        _, mask = cv.threshold(gray_mask, 0, 255, cv.THRESH_BINARY_INV)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        source_image = cv.imread(f"{image_dir}/{file}")
        source_image = cv.cvtColor(source_image, cv.COLOR_BGR2RGB)

        substrate = cv.bitwise_and(source_image, mask)
        cv.imwrite(f"{substrate_dir}/{file}", substrate)

    count = 0

    for idx, file in enumerate(files):
        substrate = cv.imread(f"{substrate_dir}/{file}", 0)

        for i in range(5):
            for j in range(5):
                part = substrate[i*178 : (i+1)*178, j*256 : (j+1)*256]
                count_black = (part == 0).sum()
                mean = part.sum() / (178 * 256 - count_black)
                replace = (np.random.randint(-7, 7, 100) + mean).astype('uint8')

                new_part = np.copy(part)
                np.place(new_part, new_part==0, replace)

                new_part = utilsCV.morph_transform(new_part, Morph.OPENING, ksize=3)
                new_part = utilsCV.filter_image(new_part, Filter.GAUSSIAN, gauss_ksize=10)
                cv.imwrite(f"{part_substrate_dir}/part_substrate_{count}.png", new_part)
                count += 1

    print(f"{count} partial substrates cut out successfully!")