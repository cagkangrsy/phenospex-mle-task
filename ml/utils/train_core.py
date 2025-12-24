from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from utils.dataset import PlantDataset


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Any, criterion: nn.Module, device: torch.device, epoch: int, total_epochs: int) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): PyTorch model to train (should be UNetTiny).
        loader (DataLoader): DataLoader providing training batches.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function (typically MSELoss).
        device (torch.device): Device to run training on (cuda or cpu).
        epoch (int): Current epoch number (for progress display).
        total_epochs (int): Total number of epochs (for progress display).

    Returns:
        float: Average training loss over all batches in the epoch.
    """
    model.train()
    running = 0.0
    total = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch:03d}/{total_epochs}", ncols=80):
        images = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, heatmaps)
        loss.backward()
        optimizer.step()

        running += loss.item() * images.size(0)
        total += images.size(0)

    return running / total if total > 0 else 0.0


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """
    Validate the model on a validation dataset.

    Args:
        model (nn.Module): PyTorch model to validate (should be UNetTiny).
        loader (DataLoader): DataLoader providing validation batches.
        criterion: Loss function (typically MSELoss).
        device (torch.device): Device to run validation on (cuda or cpu).

    Returns:
        float: Average validation loss over all batches.
    """
    model.eval()
    running = 0.0
    total = 0

    for batch in loader:
        images = batch["image"].to(device)
        heatmaps = batch["heatmap"].to(device)

        preds = model(images)
        loss = criterion(preds, heatmaps)

        running += loss.item() * images.size(0)
        total += images.size(0)

    return running / total if total > 0 else 0.0


def get_dataloaders(img_size: int, sigma: float, augment: bool, os_wt: float, s_multip: float, batch_size: int, n_workers: int, generator: torch.Generator, root: str = "data") -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Create training and validation DataLoaders with weighted sampling.

    Sets up data loaders with weighted random sampling for training (to handle
    class imbalance) and deterministic order for validation. Oversamples dataset
    ID 14 based on configuration.

    Args:
        img_size (int): Image size for resizing.
        sigma (float): Standard deviation for Gaussian heatmap generation.
        augment (bool): Whether to apply data augmentation.
        os_wt (float): Weight multiplier for dataset ID 14 samples.
        s_multip (float): Multiplier for total training samples.
        batch_size (int): Batch size for both loaders.
        n_workers (int): Number of worker processes for data loading.
        generator (torch.Generator): PyTorch random generator for reproducible sampling.
        root (str): Root directory of the dataset (should contain 'images' and
            'annotations' subdirectories).

    Returns:
        tuple: (train_loader, val_loader, train_len, val_len) where train_loader
            and val_loader are DataLoaders, and train_len, val_len are dataset sizes.
    """
    train_dataset = PlantDataset(split="train",
                                 root=root,
                                 img_size=img_size,
                                 sigma=sigma,
                                 augment=bool(augment))

    val_dataset = PlantDataset(split="val",
                               root=root,
                               img_size=img_size,
                               sigma=sigma,
                               augment=False)

    weights = []
    for img_id in train_dataset.image_ids:
        try:
            ds_id = int(img_id.split("_")[0])
        except ValueError:
            ds_id = -1

        weights.append(os_wt if ds_id == 14 else 1.0)

    total_samples = int(len(weights) * s_multip)

    sampler = WeightedRandomSampler(weights=weights,
                                    num_samples=total_samples,
                                    replacement=True,
                                    generator=generator)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=sampler,
                              num_workers=n_workers,
                              pin_memory=True,
                              collate_fn=custom_collate_fn,
                              generator=generator)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=n_workers,
                            pin_memory=True,
                            collate_fn=custom_collate_fn)

    return train_loader, val_loader, len(train_dataset), len(val_dataset)



def custom_collate_fn(batch: list) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader batching.

    Stacks tensor data (images and heatmaps) while preserving non-tensor
    metadata (centroids and IDs) as lists.

    Args:
        batch (list): List of samples from PlantDataset, where each sample is
            a dict containing 'image', 'heatmap', 'centroids', and 'id'.

    Returns:
        dict: Batched dictionary containing:
            - 'image': torch.Tensor of shape (B, C, H, W) - stacked images
            - 'heatmap': torch.Tensor of shape (B, 1, H, W) - stacked heatmaps
            - 'centroids': List of lists, one per sample
            - 'id': List of strings, one per sample
    """
    centroids = [item['centroids'] for item in batch]
    ids = [item['id'] for item in batch]
    images = [item['image'] for item in batch]
    heatmaps = [item['heatmap'] for item in batch]

    collated_batch = {'image': torch.stack(images, 0),
                      'heatmap': torch.stack(heatmaps, 0)}
    
    collated_batch['centroids'] = centroids
    collated_batch['id'] = ids
    return collated_batch