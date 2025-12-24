from pathlib import Path
from typing import Union

import cv2
import numpy as np

from cv_utils import crop, suppress_background, segment_plants, postprocess_mask, extract_bbox_centroids, save_outputs


def run(image_path: Union[str, Path], save: bool = True) -> np.ndarray:
    """
    Run the classical CV pipeline on a image.

    Args:
        image_path (str or Path): Path to the input image file.
        save (bool, optional): Save prediction outputs (JSON and visualization)

    Returns:
        np.ndarray: Centroid coordinates of detected plants as an (N, 2) array in original image coordinates.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Pipeline steps: crop, suppress background, segment, postprocess, extract centroids
    img_cropped, gridline_x = crop(img)
    img_no_bg = suppress_background(img_cropped)
    mask_initial = segment_plants(img_no_bg)
    mask_final = postprocess_mask(mask_initial)
    centroids_cropped = extract_bbox_centroids(mask_final)

    # Map centroids back to original image coordinates
    centroids = centroids_cropped.copy()
    if centroids.size > 0:
        centroids[:, 0] += gridline_x

    if save:
        save_outputs(image_path, img, centroids)

    return centroids