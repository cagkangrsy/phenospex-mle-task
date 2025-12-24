import json
import os
from typing import List, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_heatmap(h: int, w: int, centroids: Union[np.ndarray, List], sigma: float = 3.0) -> np.ndarray:
    """
    Generate a density heatmap from plant centroid coordinates.

    Creates a 2D Gaussian heatmap where each plant location is represented by
    a Gaussian distribution. Multiple plants are combined using element-wise
    maximum to create overlapping density regions.

    Args:
        h (int): Height of the heatmap in pixels.
        w (int): Width of the heatmap in pixels.
        centroids (np.ndarray or list): Array of (x, y) coordinates for plant
            centers. Shape: (N, 2) where N is the number of plants.
        sigma (float, optional): Standard deviation of the Gaussian kernel.
            Defaults to 3.0.

    Returns:
        np.ndarray: Heatmap array of shape (h, w) with values in [0, 1],
            where higher values indicate higher plant density.
    """
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for (cx, cy) in centroids:
        x = np.arange(0, w)
        y = np.arange(0, h)
        xx, yy = np.meshgrid(x, y)
        
        gaussian = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian)
    
    heatmap = np.clip(heatmap, 0, 1)
    return heatmap



class PlantDataset(Dataset):
    """
    PyTorch Dataset for plant counting task.

    Loads plant images and annotations, generating density heatmaps from
    centroid coordinates. Supports data augmentation for training and
    deterministic loading for validation/testing.
    """
    
    def __init__(self, root: str = "data", split: str = "train", img_size: int = 512, sigma: float = 3.0, augment: bool = False) -> None:

        assert split in ["train", "val", "test", "test_aug"], f"Invalid split: {split}"

        self.img_dir = os.path.join(root, "images", split)
        self.ann_path = os.path.join(root, "annotations", f"{split}.json")

        self.img_size = img_size
        self.augment = augment
        self.sigma = sigma

        with open(self.ann_path, "r") as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations.keys())

        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(p=0.3),
                ],
                keypoint_params=keypoint_params,
            )
        else:
            self.transform = A.Compose([A.NoOp()], keypoint_params=keypoint_params)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict:
        img_id = self.image_ids[idx]
        ann = self.annotations[img_id]

        img_path = os.path.join(self.img_dir, img_id)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"[Dataset] Cannot read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]
        
        centroids = np.array(ann["centroids"], dtype=np.float32)
        
        if centroids.size == 0:
            centroids = np.zeros((0, 2), dtype=np.float32)
        elif centroids.ndim == 1:
            centroids = centroids.reshape(-1, 2)

        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        scale_x = self.img_size / orig_w
        scale_y = self.img_size / orig_h
        
        centroids_resized = centroids.copy()
        if centroids_resized.size > 0:
            centroids_resized[:, 0] *= scale_x
            centroids_resized[:, 1] *= scale_y

        transformed = self.transform(image=img_resized, keypoints=centroids_resized.tolist())
        img_aug = transformed["image"]
        centroids_aug = np.array(transformed["keypoints"], dtype=np.float32)

        heatmap = generate_heatmap(self.img_size, self.img_size, centroids_aug, sigma=self.sigma)

        img_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).float()

        return {"image": img_tensor,
                "heatmap": heatmap_tensor,
                "centroids": centroids_aug,
                "id": img_id}