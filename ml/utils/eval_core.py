import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import PlantDataset
from utils.model import UNetTiny
from utils.train_core import custom_collate_fn


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained UNetTiny model from checkpoint file and put in eval mode.

    Args:
        model_path: Path to the model checkpoint file (.pt file).
        device: Device to load the model onto (CPU or CUDA).

    Returns:
        Loaded model in evaluation mode, ready for inference.

    Raises:
        FileNotFoundError: If the model file does not exist at the specified path.
    """
    model = UNetTiny(in_ch=3, out_ch=1).to(device)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_run_params(run_path: str) -> Dict[str, Any]:
    """
    Load hyperparameters from the training run directory.

    Args:
        run_path: Path to the training run directory containing hyperparameters.json.

    Returns:
        Dictionary containing all hyperparameters used during training.
    """
    params_path = os.path.join(run_path, "hyperparameters.json")
    with open(params_path, "r") as f:
        return json.load(f)


def build_dataloader(split: str, params: Dict[str, Any]) -> Tuple[PlantDataset, DataLoader]:
    """
    Build PlantDataset and corresponding DataLoader for evaluation.

    Args:
        split: Dataset split to use (e.g., "test", "val", "test_empty").
        params: Dictionary containing hyperparameters (img_size, gaussian_sigma).

    Returns:
        Tuple of (dataset, dataloader) for the specified split.
    """
    dataset = PlantDataset(
        split=split,
        img_size=params["img_size"],
        sigma=params["gaussian_sigma"],
        augment=False,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    return dataset, dataloader


def heatmap_to_centroids(heatmap: torch.Tensor, threshold: float = 0.20, radius: int = 3) -> List[Tuple[float, float]]:
    """
    Extract plant centroid coordinates from density heatmap using non-maximum suppression.

    This function identifies local maxima in the heatmap that exceed the threshold
    and applies NMS to remove duplicate detections within the specified radius.

    Args:
        heatmap: Density heatmap tensor of shape (H, W), (C, H, W), or (1, C, H, W).
        threshold: Minimum heatmap value to consider as a detection. Defaults to 0.20.
        radius: NMS radius in pixels. Defaults to 3.

    Returns:
        List of (x, y) tuples representing detected plant centroids.
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

    kernel = 2 * radius + 1
    pooled = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=radius)

    maxima = (heatmap == pooled) & (heatmap > threshold)
    coords = maxima.nonzero(as_tuple=False)

    return [(float(x), float(y)) for _, _, y, x in coords]


def match_and_compute_errors(gt: np.ndarray, pred: np.ndarray, max_dist_px: float) -> Tuple[List[float], int, int]:
    """
    Match predicted centroids to ground truth using greedy 1-to-1 matching.

    This function performs a greedy matching algorithm where each ground truth
    centroid is matched to the closest unmatched prediction within max_dist_px.
    Unmatched ground truth centroids are assigned a distance of max_dist_px.

    Args:
        gt: Ground truth centroids as (N, 2) array.
        pred: Predicted centroids as (M, 2) array.
        max_dist_px: Maximum pixel distance for a valid match.

    Returns:
        Tuple containing:
            - List of distances (one per GT, max_dist_px for unmatched)
            - Number of true positives (matched predictions)
            - Number of false positives (unmatched predictions)
    """
    if len(gt) == 0:
        return [], 0, len(pred)

    if len(pred) == 0:
        return [max_dist_px] * len(gt), 0, 0

    distances: List[float] = []
    gt_matched_count = 0
    pred_used = [False] * len(pred)

    for g in gt:
        diff = pred - g
        dd = np.sqrt((diff ** 2).sum(axis=1))

        dmin = max_dist_px
        dmin_index = -1

        for idx, d in enumerate(dd):
            if d < dmin and not pred_used[idx]:
                dmin = d
                dmin_index = idx

        if dmin < max_dist_px:
            distances.append(float(dmin))
            gt_matched_count += 1
            pred_used[dmin_index] = True
        else:
            distances.append(float(max_dist_px))

    TP = gt_matched_count
    FP = len(pred) - TP

    return distances, TP, FP


def run_evaluation(model: torch.nn.Module, dataloader: DataLoader, nms_radius: int, match_dist: float) -> tuple:
    """
    Run evaluation on all images at multiple heatmap thresholds.

    This function evaluates the model across multiple threshold values to find
    the optimal threshold that maximizes F1 score. For each threshold, it computes
    matching statistics and localization errors.

    Args:
        model: Trained UNetTiny model in evaluation mode.
        dataloader: DataLoader providing evaluation images and ground truth.
        nms_radius: Radius for non-maximum suppression in centroid extraction.
        match_dist: Maximum pixel distance for matching predictions to ground truth.

    Returns:
        Tuple of (stats dictionary, list of thresholds tested).
    """
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    stats: Dict[float, Dict] = {
        th: {"all_distances": [], "abs_count_errors": [], "TP": 0, "FP": 0}
        for th in thresholds
    }

    for sample in tqdm(dataloader, desc="Evaluating"):
        img = sample["image"].to(next(model.parameters()).device)
        gt = np.array(sample["centroids"][0], dtype=np.float32)

        t0 = time.time()
        with torch.no_grad():
            heatmap_pred = model(img)[0, 0]

        for th in thresholds:
            pred = np.array(heatmap_to_centroids(heatmap_pred.cpu(), threshold=th, radius=nms_radius),dtype=np.float32)

            stats[th]["abs_count_errors"].append(abs(len(pred) - len(gt)))

            d, tp, fp = match_and_compute_errors(gt, pred, match_dist)

            stats[th]["TP"] += tp
            stats[th]["FP"] += fp

            if len(gt) > 0:
                stats[th]["all_distances"].extend(d)

    return stats, thresholds,


def aggregate_metrics(stats: Dict[float, Dict[str, Any]], thresholds: List[float], dataset: PlantDataset, match_dist: float) -> Dict[str, Any]:
    """
    Aggregate evaluation statistics and compute final metrics.

    This function processes statistics collected across multiple thresholds and
    selects the optimal threshold based on F1 score. It computes precision, recall,
    F1, count MAE, and localization errors.

    Args:
        stats: Dictionary of statistics per threshold from run_evaluation.
        thresholds: List of threshold values tested.
        dataset: PlantDataset instance for accessing ground truth.
        match_dist: Maximum pixel distance used for matching.

    Returns:
        Dictionary containing optimal threshold and corresponding metrics.
    """
    total_gt = sum(len(dataset[i]["centroids"]) for i in range(len(dataset)))
    per_threshold: Dict[str, Dict] = {}

    max_f1 = -1.0
    optimal_th = None

    for th in thresholds:
        distances = stats[th]["all_distances"]
        # Distances are capped at match_dist for unmatched GT; we report two metrics:
        # 1) mean over ALL GT (including capped for missed plants)
        # 2) mean over MATCHED GT only (d < match_dist)
        matched_distances = [d for d in distances if d < match_dist]

        count_mae = float(np.mean(stats[th]["abs_count_errors"]))
        mean_loc_all = float(np.mean(distances)) if len(distances) > 0 else None
        mean_loc_matched = float(np.mean(matched_distances)) if len(matched_distances) > 0 else None

        TP = stats[th]["TP"]
        FP = stats[th]["FP"]
        FN = total_gt - TP
        total_predictions = TP + FP

        recall = TP / total_gt if total_gt > 0 else 0.0
        precision = TP / total_predictions if total_predictions > 0 else 0.0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )


        if f1_score > max_f1:
            max_f1 = f1_score
            optimal_th = th

        per_threshold[f"{th:.2f}"] = {
            "recall": recall,
            "precision": precision,
            "f1_score": f1_score,
            "count_mae": count_mae,
            # Mean localization error over all GT (missed plants contribute match_dist)
            "mean_loc_error_all_px": mean_loc_all,
            # Mean localization error over matched GT only
            "mean_loc_error_matched_px": mean_loc_matched,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "total_gt": total_gt,
        }

    optimal_metrics = per_threshold.get(f"{optimal_th:.2f}", {}) if optimal_th is not None else {}

    final_metrics = {
        "match_dist": match_dist,
        "total_gt": total_gt,
        "optimal_threshold": optimal_th,
        "f1_score": optimal_metrics.get("f1_score"),
        "count_mae": optimal_metrics.get("count_mae"),
        "recall": optimal_metrics.get("recall"),
        "mean_loc_error_all_px": optimal_metrics.get("mean_loc_error_all_px"),
        "mean_loc_error_matched_px": optimal_metrics.get("mean_loc_error_matched_px"),
        "TP": optimal_metrics.get("TP"),
        "FP": optimal_metrics.get("FP"),
        "FN": optimal_metrics.get("FN"),
    }

    return final_metrics


def run_evaluation_test_empty(model: torch.nn.Module, dataloader: DataLoader, dataset: PlantDataset, nms_radius: int) -> tuple:
    """
    Run evaluation on test_empty split to analyze false positives.

    This function evaluates the model on images with no ground truth plants
    to measure false positive rate. It tests multiple thresholds and reports
    the number of predictions (false positives) per image.

    Args:
        model: Trained UNetTiny model in evaluation mode.
        dataloader: DataLoader providing test_empty images.
        dataset: PlantDataset instance (unused but kept for API consistency).
        nms_radius: Radius for non-maximum suppression in centroid extraction.

    Returns:
        Tuple of (predictions_per_image dictionary, list of thresholds tested).
    """
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    
    # Store predictions per image and per threshold
    predictions_per_image: Dict[str, Dict[float, int]] = {}
    
    for sample in tqdm(dataloader, desc="Evaluating"):
        img = sample["image"].to(next(model.parameters()).device)
        img_id = sample["id"][0]
        
        with torch.no_grad():
            heatmap_pred = model(img)[0, 0]
        
        predictions_per_image[img_id] = {}
        
        for th in thresholds:
            pred = np.array(heatmap_to_centroids(heatmap_pred.cpu(), threshold=th, radius=nms_radius), dtype=np.float32)
            num_predictions = len(pred)
            predictions_per_image[img_id][th] = num_predictions
    
    return predictions_per_image, thresholds


def aggregate_metrics_test_empty(predictions_per_image: Dict[str, Dict[float, int]], thresholds: List[float]) -> Dict[str, Any]:
    """
    Aggregate prediction statistics for test_empty split.

    This function computes average predictions across all thresholds for each image
    and generates summary statistics about false positive rates.

    Args:
        predictions_per_image: Dictionary mapping image IDs to threshold-specific
            prediction counts.
        thresholds: List of threshold values tested.

    Returns:
        Dictionary containing aggregated false positive statistics.
    """
    # Compute average predictions per image across all thresholds
    avg_predictions_per_image = {}
    all_predictions_list = []
    images_with_predictions = 0
    images_without_predictions = 0
    
    for img_id, pred_dict in predictions_per_image.items():
        # Average predictions across all thresholds for this image
        predictions_across_thresholds = [pred_dict[th] for th in thresholds]
        avg_pred = float(np.mean(predictions_across_thresholds))
        avg_predictions_per_image[img_id] = avg_pred
        all_predictions_list.append(avg_pred)
        
        # Check if image has any predictions on average
        if avg_pred > 0:
            images_with_predictions += 1
        else:
            images_without_predictions += 1
    
    total_images = len(predictions_per_image)
    mean_predictions_per_image = float(np.mean(all_predictions_list)) if all_predictions_list else 0.0
    max_predictions = float(np.max(all_predictions_list)) if all_predictions_list else 0.0
    min_predictions = float(np.min(all_predictions_list)) if all_predictions_list else 0.0
    total_avg_predictions = float(np.sum(all_predictions_list))
    
    return {
        "total_images": total_images,
        "total_avg_predictions": total_avg_predictions,
        "images_with_predictions": images_with_predictions,
        "images_without_predictions": images_without_predictions,
        "mean_predictions_per_image": mean_predictions_per_image,
        "max_predictions_in_single_image": max_predictions,
        "min_predictions_in_single_image": min_predictions,
        "predictions_per_image": {img_id: float(avg_pred) for img_id, avg_pred in avg_predictions_per_image.items()}
    }


def save_eval_summary(run_path: str, summary: Dict[str, Any], split: str) -> None:
    """
    Save evaluation summary to JSON file in the run directory.

    Args:
        run_path: Path to the training run directory where summary will be saved.
        summary: Dictionary containing evaluation metrics and statistics.
        split: Dataset split name (used in filename).
    """
    filename = f"eval_summary_{split}.json"
    out_path = os.path.join(run_path, filename)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to: {out_path}")
