import os
import sys
from pathlib import Path
from typing import List

import cv2 as cv
import numpy as np

from pipeline_grid_removal import straighten_page, remove_grid_lines
from pipeline_word_detection import get_clusters, get_segmentation_mask, get_row_descriptions, create_mask


def load_images(directory: str, image_count: int) -> List[np.ndarray]:
    images: List[np.ndarray] = []

    for filename_number in range(image_count):
        if filename_number - 1 == image_count:
            break

        filename = f"{filename_number}.png"
        file_path = os.path.join(directory, filename)

        print(file_path)
        file_exists = os.path.exists(file_path)
        print(f"File exists : {file_exists}")

        image: np.ndarray = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
        images.append(image)

    return images


def main():
    if len(sys.argv) < 2:
        print("Missing input path, image count and output path arguments")
        return
    elif len(sys.argv) < 3:
        print("Missing image count and output path arguments")
        return
    elif len(sys.argv) < 4:
        print("Missing output path argument")
        return

    input_path: str = sys.argv[1]
    image_count = int(sys.argv[2])
    output_path: str = sys.argv[3]

    input_images = load_images(input_path, image_count)

    images = []
    images_moments = []
    for image in input_images:
        image_straightened, moments = straighten_page(image)
        image_no_grid = remove_grid_lines(image_straightened)

        images.append(image_no_grid)
        images_moments.append(moments)

    cluster_list = get_clusters(images)
    segmentation_result_list = get_segmentation_mask(images, cluster_list)
    images_row_descriptions = get_row_descriptions(images, segmentation_result_list)

    Path(output_path).mkdir(exist_ok=True)

    loop_collection = enumerate(zip(input_images, images_moments, images_row_descriptions))
    for i, (input_image, moments, row_description_list) in loop_collection:
        mask_shape = segmentation_result_list[0].image.shape
        output_shape = (input_image.shape[1], input_image.shape[0])
        warped_mask = create_mask(mask_shape, output_shape, row_description_list, moments)

        mask_file_path = os.path.join(output_path, f"{i}-wyrazy.png")
        cv.imwrite(mask_file_path, warped_mask)


if __name__ == '__main__':
    main()
