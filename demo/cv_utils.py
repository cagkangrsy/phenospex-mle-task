import json
import os.path
from pathlib import Path

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


def crop(img, left_ignore_ratio=CONFIG["crop"]["left_ignore_ratio"], edge_thresh_ratio=CONFIG["crop"]["edge_thresh_ratio"]):
    """
    Detect the vertical gridline and crop away the left part of the image.

    This is the very first step in the alternative pipeline: it restricts
    processing to the plant region by removing the unused area on the left.
    """
    # Grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # Ignore a fraction of the left border where no gridline is expected
    left_ignore_px = int(left_ignore_ratio * W)

    # Vertical edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = cv2.normalize(sobel_x, None, 0, 1, cv2.NORM_MINMAX)

    # Column-wise edge strength
    col_strength = sobel_x.mean(axis=0)
    col_strength[:left_ignore_px] = 0.0

    # Threshold and take the first strong peak as gridline
    thresh = edge_thresh_ratio * col_strength.max()
    candidates = np.where(col_strength > thresh)[0]
    if len(candidates) == 0:
        raise RuntimeError("No gridline detected")

    gridline_x = int(candidates[0])

    # Crop image to the right of the gridline
    cropped_img = img[:, gridline_x:].copy()

    return cropped_img, gridline_x


def suppress_background(img):
    """
    Suppress bright white glare and gray table background using HSV thresholds.

    Pixels matching the configured white/gray table ranges are set to black,
    leaving mainly the plant region for subsequent processing.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    white_mask = (v >= V_WHITE) & (s <= S_LOW)
    gray_table_mask = (v >= V_GRAY_MIN) & (v <= V_GRAY_MAX) & (s <= S_GRAY)
    background_mask = white_mask | gray_table_mask

    img_no_bg = img.copy()
    img_no_bg[background_mask] = (0, 0, 0)
    return img_no_bg


def segment_plants(img_no_bg):
    """
    Segment plants using grayscale conversion, median denoising,
    and Otsu thresholding.

    Returns a binary mask where plant regions are foreground.
    """
    gray = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2GRAY)
    gray_denoised = cv2.medianBlur(gray, 3)

    _, mask = cv2.threshold(gray_denoised,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Simple opening to remove small speckles before blob filtering
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return opening


def filter_small_blobs(binary, min_area):
    """
    Remove connected components smaller than the given area threshold.

    This is used as part of mask postprocessing to drop tiny noise blobs.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area

    cleaned = np.zeros_like(binary)
    cleaned[keep[labels]] = 255
    return cleaned


def postprocess_mask(mask):
    """
    Clean up the binary mask: remove small blobs and merge fragmented plants.

    Uses area-based blob filtering followed by a morphological close with
    a configurable kernel size and number of iterations.
    """
    kernel_size = CONFIG["postprocess"]["kernel_size"]
    min_area = CONFIG["postprocess"]["min_area"]
    merge_iterations = CONFIG["postprocess"]["merge_iterations"]

    # Drop small noise components
    mask_filtered = filter_small_blobs(mask, min_area)

    # Merge fragmented plant parts
    kernel_merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size, kernel_size))
    mask_merged = cv2.morphologyEx(mask_filtered, cv2.MORPH_CLOSE, kernel_merge, iterations=merge_iterations)
    return mask_merged


def extract_bbox_centroids(binary):
    """
    Extract centroid coordinates of connected components' bounding boxes.

    Returns an array of shape (N, 2) in (x, y) order, used as plant positions.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    stats = stats[1:]  # drop background

    cx = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] / 2.0
    cy = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] / 2.0

    return np.stack([cx, cy], axis=1)


def save_outputs(image_path, image, centroids_xy):
    """
    Persist predictions and an overlay visualization to disk.

    Saves a JSON with numeric centroids and an image with red dots marking
    the detected plant positions.
    """
    base = os.path.basename(image_path)
    out_dir = Path("inference") / base
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Save JSON ----------
    json_path = out_dir / f"{base}_cv_predictions.json"

    payload = {
        "count": int(len(centroids_xy)),
        "centroids": np.round(centroids_xy, 2).tolist(),
    }

    with json_path.open("w") as f:
        json.dump(payload, f, indent=4)

    # ---------- Save image ----------
    overlay = image.copy()
    for x, y in centroids_xy:
        cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 255), -1)

    img_path = out_dir / f"{base}_cv_predictions.png"
    cv2.imwrite(str(img_path), overlay)

    print(f"[INFO] Saved JSON → {json_path}")
    print(f"[INFO] Saved image → {img_path}")