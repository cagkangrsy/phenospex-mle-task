import json
import os.path
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


# Central configuration for classical CV parameters (tunable in one place).
CONFIG = {
    "background": {
        "v_white": 220,  # bright whites / glare
        "s_low": 25,
        "v_gray_min": 90,  # gray table (not bright)
        "v_gray_max": 200,
        "s_gray": 25,
    },
    "postprocess": {
        "min_area": 600,
        "kernel_size": 5,
        "merge_iterations": 2,
    },
    "crop": {
        "left_ignore_ratio": 0.05,
        "edge_thresh_ratio": 0.25,
    },
}

# Module-level constants derived from CONFIG for convenience.
V_WHITE = CONFIG["background"]["v_white"]
S_LOW = CONFIG["background"]["s_low"]
V_GRAY_MIN = CONFIG["background"]["v_gray_min"]
V_GRAY_MAX = CONFIG["background"]["v_gray_max"]
S_GRAY = CONFIG["background"]["s_gray"]
MIN_AREA = CONFIG["postprocess"]["min_area"]


def crop(img: np.ndarray, left_ignore_ratio: float = CONFIG["crop"]["left_ignore_ratio"], edge_thresh_ratio: float = CONFIG["crop"]["edge_thresh_ratio"]) -> Tuple[np.ndarray, int]:
    """
    Uses Sobel edge detection to identify the first vertical gridline, then crops to restrict processing to the plant region.

    Args:
        img (np.ndarray): Input image in BGR format.
        left_ignore_ratio (float, optional): Fraction of left border to ignore when searching for gridline. Defaults to CONFIG["crop"]["left_ignore_ratio"].
        edge_thresh_ratio (float, optional): Threshold ratio for edge detection (relative to max edge strength). Defaults to CONFIG["crop"]["edge_thresh_ratio"].

    Returns:
        tuple: (cropped_img, gridline_x) where cropped_img is the image cropped to the right of the gridline, and gridline_x is the X-coordinate of the detected gridline in the original image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Ignore left border where no gridline is expected
    left_ignore_px = int(left_ignore_ratio * W)

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = cv2.normalize(sobel_x, None, 0, 1, cv2.NORM_MINMAX)

    col_strength = sobel_x.mean(axis=0)
    col_strength[:left_ignore_px] = 0.0

    thresh = edge_thresh_ratio * col_strength.max()
    candidates = np.where(col_strength > thresh)[0]
    if len(candidates) == 0:
        raise RuntimeError("No gridline detected")

    gridline_x = int(candidates[0])
    cropped_img = img[:, gridline_x:].copy()

    return cropped_img, gridline_x


def suppress_background(img: np.ndarray) -> np.ndarray:
    """
    Suppress bright white shadows and gray table background using HSV thresholds.

    Args:
        img (np.ndarray): Input image in BGR format.

    Returns:
        np.ndarray: Image with background pixels set to black (0, 0, 0), preserving plant regions in their original colors.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    white_mask = (v >= V_WHITE) & (s <= S_LOW)
    gray_table_mask = (v >= V_GRAY_MIN) & (v <= V_GRAY_MAX) & (s <= S_GRAY)
    background_mask = white_mask | gray_table_mask

    img_no_bg = img.copy()
    img_no_bg[background_mask] = (0, 0, 0)
    return img_no_bg


def segment_plants(img_no_bg: np.ndarray) -> np.ndarray:
    """
    Segment plants using grayscale conversion, median denoising, and Otsu thresholding.

    Args:
        img_no_bg (np.ndarray): Background-suppressed image in BGR format.

    Returns:
        np.ndarray: Binary mask where plant regions are foreground (255) and background is 0.
    """
    gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.medianBlur(gray, 3)

    _, mask = cv2.threshold(gray_denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opening


def filter_small_blobs(binary: np.ndarray, min_area: int) -> np.ndarray:
    """
    Remove connected components smaller than the given area threshold.

    Args:
        binary (np.ndarray): Binary mask image where foreground pixels are non-zero.
        min_area (int): Minimum area threshold in pixels. Components smaller than this will be removed.

    Returns:
        np.ndarray: Filtered binary mask with small blobs removed. Same shape as input.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area

    cleaned = np.zeros_like(binary)
    cleaned[keep[labels]] = 255
    return cleaned


def postprocess_mask(mask: np.ndarray) -> np.ndarray:
    """
    Clean up binary mask: remove small blobs and merge fragmented plants.

    Args:
        mask (np.ndarray): Binary mask from initial segmentation.

    Returns:
        np.ndarray: Cleaned binary mask with small noise removed and fragmented plants merged. Same shape as input.
    """
    kernel_size = CONFIG["postprocess"]["kernel_size"]
    min_area = CONFIG["postprocess"]["min_area"]
    merge_iterations = CONFIG["postprocess"]["merge_iterations"]

    mask_filtered = filter_small_blobs(mask, min_area)

    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_merged = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel_merge, iterations=merge_iterations)
    return mask_merged


def extract_bbox_centroids(binary: np.ndarray) -> np.ndarray:
    """
    Extract centroid coordinates of connected components' bounding boxes.

    Args:
        binary (np.ndarray): Binary mask image where foreground pixels are non-zero.

    Returns:
        np.ndarray: Array of shape (N, 2) containing centroid coordinates in (x, y) order, where N is the number of connected components.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    stats = stats[1:]  # drop background

    cx = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] / 2.0
    cy = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] / 2.0

    return np.stack([cx, cy], axis=1)


def save_outputs(image_path: Union[str, Path], image: np.ndarray, centroids_xy: np.ndarray) -> None:
    """
    Save predictions and visualizations to disk.

    Args:
        image_path (str or Path): Path to the original input image file.
        image (np.ndarray): Original image array for visualization overlay.
        centroids_xy (np.ndarray): Array of shape (N, 2) containing centroid coordinates in (x, y) format.
    """
    base = os.path.basename(image_path)
    out_dir = Path("inference") / base
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{base}_cv_predictions.json"
    payload = {
        "count": int(len(centroids_xy)),
        "centroids": np.round(centroids_xy, 2).tolist(),
    }

    with json_path.open("w") as f:
        json.dump(payload, f, indent=4)

    overlay = image.copy()
    for x, y in centroids_xy:
        cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 255), -1)

    img_path = out_dir / f"{base}_cv_predictions.png"
    cv2.imwrite(str(img_path), overlay)

    print(f"[INFO] Saved JSON → {json_path}")
    print(f"[INFO] Saved image → {img_path}")