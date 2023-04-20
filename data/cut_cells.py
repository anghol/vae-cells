import os
import time

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def main():
    segmentation_dir = '/usr/src/data/segmentation_object'
    images_dir = '/usr/src/data/valid_bf_imgs_dir_png'
    separated_cells_dir = '/usr/src/data/test_separated_cells'

    if not os.path.isdir(separated_cells_dir):
        os.mkdir(separated_cells_dir)

    segmentation_images = os.listdir(segmentation_dir)

    start_time = time.time()
    num = 1
    for idx, img_name in enumerate(segmentation_images):

        os.chdir(segmentation_dir)
        mask = cv.imread(img_name)
        os.chdir('..')

        os.chdir(images_dir)
        image = cv.imread(img_name, 0)
        os.chdir('..')

        # вытаскиваем все цвета сегментированных бактерий, color = [r, g, b]
        colors = np.unique(mask.reshape(-1, 3), axis=0)
        colors = np.delete(colors, 0, axis=0) # удалить черный

        zeros = np.zeros_like(mask)
        ones = np.ones_like(mask) * 255 # type: ignore

        # separate_cells = []
        os.chdir(separated_cells_dir)
        for color in colors:
            # область с клеткой нужного цвета
            true_area = np.all(mask == color, axis=2)
            true_area = np.repeat(true_area, 3).reshape(mask.shape)

            # маска одной бактерии
            cell_mask = np.where(true_area, ones, zeros)
            cell_mask = cv.cvtColor(cell_mask, cv.COLOR_BGR2GRAY)

            # контуры одной бактерии и обрамляющий прямоугольник 
            contours, hierarcy = cv.findContours(cell_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            x, y, w, h = cv.boundingRect(contours[0])
            # separate_cell_masks.append(cell_mask[y:y+h, x:x+w])

            # накладываем на исходное изображение бинарную маску одной бактерии
            cell = image.copy()
            cell[~(cell_mask > 0)] = 0 # type: ignore

            # вырезаем прямоугольник с бактерией
            cell = cell[y:y+h, x:x+w]
            plt.imsave(f'cell_test_{num}.png', cell, cmap='gray', vmin=0, vmax=255)
            num += 1        
            # separate_cells.append(cell)
        os.chdir('..')

        if idx % 10 == 0:
            print(f'Current image: {img_name}\nCurrent count of cells: {num-1}\n')
    
    stop_time = time.time()
    print('FINISH!\nTotal time:', stop_time - start_time)

if __name__ == '__main__':
    # print(os.listdir())
    main()