import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cv_pipeline import run as run_cv_pipeline

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate classical CV pipeline on dataset")
    parser.add_argument("--split", "--s", type=str, default="test", choices=["train", "val", "test", "test_aug", "test_empty"], help="Dataset split to evaluate on (default: test).")
    parser.add_argument("--match_dist", "--m", type=float, default=5.0, help="Maximum pixel distance to match prediction to ground truth.")
    return parser.parse_args()


class AnnotationDataset:
    """Minimal dataset wrapper over JSON annotations."""

    def __init__(self, root="../ml/data", split: str = "test") -> None:
        self.root = root
        self.split = split

        ann_path = os.path.join(root, "annotations", f"{split}.json")
        with open(ann_path, "r") as f:
            self.annotations = json.load(f)

        self.image_ids = sorted(self.annotations.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        img_id = self.image_ids[idx]
        ann = self.annotations[img_id]
        centroids = np.array(ann["centroids"], dtype=np.float32)
        return {"id": img_id, "centroids": centroids}


def match_and_compute_errors(gt: np.ndarray, pred: np.ndarray, max_dist_px: float) -> Tuple[List[float], int, int]:
    """
    Greedy 1-to-1 matching between GT and predicted centroids.

    Args:
        gt (np.ndarray): Ground truth centroids as (N, 2) array.
        pred (np.ndarray): Predicted centroids as (M, 2) array.
        max_dist_px (float): Maximum pixel distance for a valid match.

    Returns:
        tuple: (distances, tp, fp) where distances is a list of per-GT distances (max_dist_px for missed GT), true positives, and false positives.
    """
    if len(gt) == 0:
        return [], 0, int(len(pred))

    if len(pred) == 0:
        return [max_dist_px] * int(len(gt)), 0, 0

    distances: List[float] = []
    gt_matched_count = 0
    pred_used = [False] * len(pred)

    for g in gt:
        diff = pred - g
        dd = np.sqrt((diff**2).sum(axis=1))

        dmin = max_dist_px
        dmin_index = -1
        for idx, d in enumerate(dd):
            if d < dmin and not pred_used[idx]:
                dmin = float(d)
                dmin_index = idx

        if dmin < max_dist_px and dmin_index >= 0:
            distances.append(dmin)
            gt_matched_count += 1
            pred_used[dmin_index] = True
        else:
            distances.append(max_dist_px)

    tp = int(gt_matched_count)
    fp = int(len(pred) - tp)
    return distances, tp, fp


def aggregate_metrics(stats: Dict[str, object], runtimes: List[float], dataset: AnnotationDataset, args: argparse.Namespace) -> Dict[str, object]:
    """
    Compute precision/recall/F1, MAE and localization error.

    Args:
        stats (Dict[str, object]): Statistics dictionary containing all_distances, abs_count_errors, TP, and FP.
        runtimes (List[float]): List of inference runtimes in seconds.
        dataset: AnnotationDataset instance.
        args: Parsed arguments containing match_dist.

    Returns:
        Dict[str, object]: Metrics dictionary containing match_dist, total_gt_centroids,
            avg_runtime_ms, f1_score, mean_count_error_per_image, recall, miss_rate,
            mean_localization_error_pixels, TP, FP, and FN.
    """
    total_gt = sum(len(dataset[i]["centroids"]) for i in range(len(dataset)))

    avg_runtime_sec = float(np.mean(runtimes)) if runtimes else 0.0
    avg_runtime_ms = avg_runtime_sec * 1000.0

    distances = stats["all_distances"]
    finite = [d for d in distances if d < args.match_dist]

    count_mae = float(np.mean(stats["abs_count_errors"])) if stats["abs_count_errors"] else 0.0
    mean_loc = float(np.mean(finite)) if len(finite) > 0 else None

    tp = int(stats["TP"])
    fp = int(stats["FP"])
    fn = int(total_gt - tp)
    total_predictions = tp + fp

    recall = tp / total_gt if total_gt > 0 else 0.0
    precision = tp / total_predictions if total_predictions > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0)

    plant_error_rate = fn / total_gt if total_gt > 0 else 0.0
    plant_accuracy = recall

    return {
        "match_dist": float(args.match_dist),
        "total_gt_centroids": int(total_gt),
        "avg_runtime_ms": float(avg_runtime_ms),
        "f1_score": float(f1_score),
        "mean_count_error_per_image": float(count_mae),
        "recall": float(plant_accuracy),
        "miss_rate": float(plant_error_rate),
        "mean_localization_error_pixels": mean_loc,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def save_eval_summary(summary: Dict[str, object], split: str) -> None:
    """
    Save evaluation summary to JSON file.

    Args:
        summary (Dict[str, object]): Evaluation metrics dictionary.
        split (str): Dataset split name used in filename.
    """
    filename = f"eval_summary_{split}.json"
    out_path = os.path.join(os.getcwd(), filename)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved evaluation summary to: {out_path}")


def run_evaluation(dataset: AnnotationDataset, args: argparse.Namespace) -> Tuple[Dict[str, object], List[float]]:
    """
    Run evaluation for the classical CV pipeline.

    Args:
        dataset (AnnotationDataset): Dataset instance containing ground truth.
        args (argparse.Namespace): Parsed arguments containing split and match_dist.

    Returns:
        tuple: (stats, runtimes) where stats is a dictionary containing
            all_distances, abs_count_errors, TP, and FP, and runtimes is
            a list of inference times in seconds.
    """
    stats: Dict[str, object] = {
        "all_distances": [],
        "abs_count_errors": [],
        "TP": 0,
        "FP": 0,
    }
    runtimes: List[float] = []

    img_dir = os.path.join("../ml/data", "images", args.split)

    for i in range(len(dataset)):
        sample = dataset[i]
        img_id = sample["id"]
        gt = sample["centroids"]

        img_path = Path(os.path.join(img_dir, img_id))

        t0 = time.time()
        pred_full = run_cv_pipeline(img_path, save=False)
        runtimes.append(time.time() - t0)

        if pred_full is None:
            pred_full = np.zeros((0, 2), dtype=np.float32)

        stats["abs_count_errors"].append(abs(len(pred_full) - len(gt)))

        dists, tp, fp = match_and_compute_errors(gt, pred_full, args.match_dist)
        stats["TP"] += tp
        stats["FP"] += fp

        if len(gt) > 0:
            stats["all_distances"].extend(dists)

    return stats, runtimes


def main() -> None:
    """
    Main evaluation function for classical CV pipeline.
    """
    args = parse_args()

    dataset = AnnotationDataset(split=args.split)
    total_images = len(dataset)
    print(f"Evaluating classical CV on split='{args.split}' with {total_images} images.")

    stats, runtimes = run_evaluation(dataset, args)
    metrics = aggregate_metrics(stats, runtimes, dataset, args)

    summary = {
        "method": "classical_cv",
        "split": args.split,
        **metrics,
    }

    focused_summary = {
        "model": "Connected Component Centroids Extraction",
        "split": args.split,
        "match_distance_pixels": summary["match_dist"],
        "f1_score": summary["f1_score"],
        "mean_count_error_per_image": summary["mean_count_error_per_image"],
        "recall": summary["recall"],
        "miss_rate": summary["miss_rate"],
        "true_positives": summary["TP"],
        "false_positives": summary["FP"],
        "false_negatives": summary["FN"],
        "total_ground_truth_plants": summary["total_gt_centroids"],
        "mean_localization_error_pixels": summary["mean_localization_error_pixels"],
    }

    opt_loc = summary["mean_localization_error_pixels"]
    opt_rec = summary["recall"] or 0.0
    opt_miss = summary["miss_rate"] or 0.0
    opt_TP = summary["TP"]
    opt_FN = summary["FN"]
    total_gt = summary["total_gt_centroids"]

    print("\n=== Classical CV Evaluation ===")
    print(f"Match distance (px):           {summary['match_dist']}")
    print(f"F1-Score:                      {summary['f1_score']:.3f}")
    print(f"Count MAE (per image):         {summary['mean_count_error_per_image']:.3f}")
    print("\n--- Plant-Instance Level ---")
    print(f"Recall (plant accuracy):       {opt_rec*100:.2f}%")
    print(f"Miss rate:                     {opt_miss*100:.2f}%")
    print(f"Correctly identified:          {opt_TP}/{total_gt} plants")
    print(f"Missed plants (FN):            {opt_FN}/{total_gt} plants")
    print("\n--- Localization ---")
    if opt_loc is not None:
        print(f"Mean localization error:       {opt_loc:.3f} px")
    else:
        print("Mean localization error:       N/A")

    save_eval_summary(focused_summary, split=args.split)


if __name__ == "__main__":
    main()