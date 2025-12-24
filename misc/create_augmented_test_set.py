import json
import os
from typing import Dict

import cv2
import numpy as np
from tqdm import tqdm


ROTATIONS = [90, 180, 270]
NOISE_SIGMA = 10.0
ERASE_FRACTION = 0.3
SEED = 42
test_images_dir = os.path.join("../ml/data/images/test")
out_images_dir = os.path.join("../ml/data/images/test_aug")
test_annotations_dir = os.path.join("../ml/data/annotations/test.json")
test_aug_annotations_dir = os.path.join("../ml/data/annotations/test_aug.json")



def rotate_image_and_points(img, centroids, angle):
    """
    Rotate image and centroid coordinates by 90/180/270 degrees.

    Args:
        img (np.ndarray): Input image in BGR format.
        centroids (np.ndarray): Array of shape (N, 2) containing centroid coordinates.
        angle (int): Rotation angle in degrees (90, 180, or 270).

    Returns:
        tuple: (rotated_img, rotated_centroids, new_w, new_h) where rotated_img
            is the rotated image, rotated_centroids are transformed coordinates,
            and new_w, new_h are the new image dimensions.
    """
    h, w = img.shape[:2]

    if angle == 90:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # (x, y) -> (h-1-y, x)
        xs = centroids[:, 0]
        ys = centroids[:, 1]
        new_xs = h - 1 - ys
        new_ys = xs
        new_w, new_h = h, w
    elif angle == 180:
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        # (x, y) -> (w-1-x, h-1-y)
        xs = centroids[:, 0]
        ys = centroids[:, 1]
        new_xs = w - 1 - xs
        new_ys = h - 1 - ys
        new_w, new_h = w, h
    elif angle == 270:
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # (x, y) -> (y, w-1-x)
        xs = centroids[:, 0]
        ys = centroids[:, 1]
        new_xs = ys
        new_ys = w - 1 - xs
        new_w, new_h = h, w
    else:
        raise ValueError("Only 90, 180, 270 degree rotations are supported.")

    rotated_centroids = np.stack([new_xs, new_ys], axis=1).astype(np.float32)
    return rotated_img, rotated_centroids, new_w, new_h


def add_gaussian_noise(img, sigma):
    """
    Add Gaussian noise to image.

    Args:
        img (np.ndarray): Input image in BGR format.
        sigma (float): Standard deviation of Gaussian noise. If <= 0, returns original image.

    Returns:
        np.ndarray: Noisy image with same shape and dtype as input.
    """
    if sigma <= 0:
        return img
    noise = np.random.normal(loc=0.0, scale=sigma, size=img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def erase_random_plants(img, centroids, erase_fraction):
    """
    Erase plants inside random rectangles and drop their annotations.

    Samples 2-3 rectangular regions, blanks them out with background color,
    and keeps only centroids outside those regions. erase_fraction controls
    the approximate total fraction of image area covered by rectangles.

    Args:
        img (np.ndarray): Input image in BGR format.
        centroids (np.ndarray): Array of shape (N, 2) containing centroid coordinates.
        erase_fraction (float): Approximate fraction of image area to erase (clamped to [0.01, 0.9]).

    Returns:
        tuple: (erased_img, kept_centroids) where erased_img is the image with
            rectangles blanked out, and kept_centroids are centroids outside erased regions.
    """
    h, w = img.shape[:2]
    n = centroids.shape[0]
    if n == 0 or erase_fraction <= 0:
        return img, centroids

    erase_fraction = float(max(0.01, min(0.9, erase_fraction)))

    img_out = img.copy()

    bg_color = np.median(img_out.reshape(-1, 3), axis=0).astype(np.uint8)

    num_rects = np.random.randint(2, 4)

    total_target_area = erase_fraction * h * w
    area_per_rect = total_target_area / float(num_rects)
    base_side = np.sqrt(area_per_rect)

    xs = centroids[:, 0]
    ys = centroids[:, 1]

    inside = np.zeros(n, dtype=bool)

    rects = []

    for _ in range(num_rects):
        for attempt in range(10):
            aspect = np.random.uniform(0.7, 1.5)
            rect_h = int(np.clip(base_side * np.sqrt(aspect), h * 0.15, h * 0.7))
            rect_w = int(np.clip(base_side / np.sqrt(aspect), w * 0.15, w * 0.7))

            cx = np.random.randint(int(w * 0.25), int(w * 0.75))
            cy = np.random.randint(int(h * 0.25), int(h * 0.75))

            x0 = max(0, cx - rect_w // 2)
            y0 = max(0, cy - rect_h // 2)
            x1 = min(w, x0 + rect_w)
            y1 = min(h, y0 + rect_h)

            if x1 <= x0 or y1 <= y0:
                continue

            ok = True
            for (ex0, ey0, ex1, ey1) in rects:
                ix0 = max(x0, ex0)
                iy0 = max(y0, ey0)
                ix1 = min(x1, ex1)
                iy1 = min(y1, ey1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                area_new = (x1 - x0) * (y1 - y0)
                if area_new > 0 and inter / area_new > 0.3:
                    ok = False
                    break

            if ok:
                rects.append((x0, y0, x1, y1))
                break
        else:
            continue

        img_out[y0:y1, x0:x1, :] = bg_color

        inside_rect = (xs >= x0) & (xs < x1) & (ys >= y0) & (ys < y1)
        inside |= inside_rect

    kept_centroids = centroids[~inside]

    return img_out, kept_centroids


def main():
    """
    Generate augmented test set with rotations, noise, and erased plants.
    """
    np.random.seed(SEED)

    with open(test_annotations_dir, "r") as f:
        test_annotations: Dict[str, Dict] = json.load(f)

    os.makedirs(out_images_dir, exist_ok=True)
    aug_annotations: Dict[str, Dict] = {}

    for img_id, meta in tqdm(test_annotations.items(), desc="Augmenting test set"):
        img_path = os.path.join(test_images_dir, img_id)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}, skipping.")
            continue

        h, w = img.shape[:2]
        centroids = np.array(meta["centroids"], dtype=np.float32)
        if centroids.ndim == 1 and centroids.size == 2:
            centroids = centroids.reshape(1, 2)
        elif centroids.ndim == 1:
            centroids = centroids.reshape(0, 2)

        stem, ext = os.path.splitext(img_id)

        for angle in ROTATIONS:
            rotated_img, rotated_centroids, new_w, new_h = rotate_image_and_points(img, centroids, angle)
            rotated_noisy = add_gaussian_noise(rotated_img, NOISE_SIGMA)

            out_name = f"{stem}_rot{angle}_noise{int(NOISE_SIGMA)}{ext}"
            out_path = os.path.join(out_images_dir, out_name)
            cv2.imwrite(out_path, rotated_noisy)

            aug_annotations[out_name] = {
                "width": int(new_w),
                "height": int(new_h),
                "centroids": rotated_centroids.tolist()
            }

        erased_img, kept_centroids = erase_random_plants(img, centroids, erase_fraction=ERASE_FRACTION)

        out_name_erase = f"{stem}_erased{ext}"
        out_path_erase = os.path.join(out_images_dir, out_name_erase)
        cv2.imwrite(out_path_erase, erased_img)

        aug_annotations[out_name_erase] = {
            "width": int(w),
            "height": int(h),
            "centroids": kept_centroids.tolist(),
        }

    with open(test_aug_annotations_dir, "w") as f:
        json.dump(aug_annotations, f, indent=2)

    print(f"\nAugmented test images written to: {out_images_dir}")
    print("Augmented annotations written to: ", test_aug_annotations_dir)


if __name__ == "__main__":
    main()
