import cv2
import numpy as np

from cv_utils import (
    crop,
    suppress_background,
    segment_plants,
    postprocess_mask,
    extract_bbox_centroids,
    save_outputs,
)


def run(image_path, save=True) -> np.ndarray:
    """
    Run the full classical CV pipeline on the provided image.

    Returns the centroid coordinates of detected plants as an (N, 2) array.
    """

    # Load image from disk
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # 1) Crop away table region to the left of the gridline
    img_cropped, gridline_x = crop(img)

    # 2) Suppress white glare and gray table background
    img_no_bg = suppress_background(img_cropped)

    # 3) Segment plants (binary mask)
    mask_initial = segment_plants(img_no_bg)

    # 4) Postprocess mask
    mask_final = postprocess_mask(mask_initial)

    # 5) Extract centroids (cropped coords)
    centroids_cropped = extract_bbox_centroids(mask_final)

    # Map back to original image coordinates
    centroids = centroids_cropped.copy()
    if centroids.size > 0:
        centroids[:, 0] += gridline_x

    # 6) Save outputs only when not in eval mode
    if save:
        save_outputs(image_path, img, centroids)

    return centroids