import os, argparse, time
import cv2 as cv
import numpy as np
import utils_opencv as utilsCV


def get_substrates(segmentation_dir: str, image_dir: str, substrate_dir: str, part_substrate_dir: str):
    """ Get partial substrates """


    # Cut large substrates with black bacteria's areas
    
    if not os.path.isdir(substrate_dir):
        os.mkdir(substrate_dir)
    if not os.path.isdir(part_substrate_dir):
        os.mkdir(part_substrate_dir)

    files = sorted(os.listdir(segmentation_dir))

    start = time.time()
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


    # Cutting partial substrates from prepared big ones 

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

                new_part = utilsCV.morph_transform(new_part, utilsCV.Morph.OPENING, ksize=3)
                new_part = utilsCV.filter_image(new_part, utilsCV.Filter.GAUSSIAN, gauss_ksize=10)
                cv.imwrite(f"{part_substrate_dir}/part_substrate_{count}.png", new_part)
                count += 1

    stop = time.time()
    print(f"{count} partial substrates cut out successfully!\nTotal time = {stop - start :.3f} sec")

def main():
    parser = argparse.ArgumentParser(description="Get partial substrates from big ones using segmentation images")
    parser.add_argument('source_segmentaion_dir', type=str, help='Input dir with annotated images')
    parser.add_argument('source_image_dir', type=str, help='Input dir with files from CVAT')
    parser.add_argument('dest_substrate_dir', type=str, help='Output dir for images of substrates')
    parser.add_argument('dest_partsubstrate_dir', type=str, help='Output dir for images of partial substrates')
    args = parser.parse_args()

    segmentation_dir = args.source_segmentation_dir
    image_dir = args.source_image_dir
    substrate_dir = args.dest_substrate_dir
    part_substrate_dir = args.dest_partsubstrate_dir

    get_substrates(segmentation_dir, image_dir, substrate_dir, part_substrate_dir)

if __name__ == "__main__":
    main()