import argparse
import os

import torch

from utils.eval_core import (aggregate_metrics, aggregate_metrics_test_empty, build_dataloader, load_model, load_run_params, run_evaluation, run_evaluation_test_empty, save_eval_summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate model")
    parser.add_argument('--device', type=str, default=None, help='Explicit device to use (e.g., "cuda" or "cpu"). Overrides automatic detection.')
    parser.add_argument("--model", '--m', type=str, help="Path to the trained model checkpoint file (.pt)")
    parser.add_argument("--split", '--s', type=str, default="test", help="Dataset split to evaluate on (default: test)",)
    parser.add_argument("--nms_radius", '--nms', type=int, default=3, help="Radius for non-maximum suppression (default: 3)")
    parser.add_argument("--match_dist", '--md', type=float, default=5.0, help="Maximum pixel distance to match prediction to ground truth (default: 5.0)")
    args = parser.parse_args()
    if args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    model_path = args.model
    run_path = os.path.dirname(model_path)

    model = load_model(model_path, device)
    params = load_run_params(run_path)
    dataset, dataloader = build_dataloader(args.split, params)

    print(f"Running on: {device}")
    total_images = len(dataset)
    print(f"Total samples: {total_images}")

    if args.split == "test_empty":
        predictions_per_image, thresholds = run_evaluation_test_empty(model, dataloader, dataset, args.nms_radius)
        metrics = aggregate_metrics_test_empty(predictions_per_image, thresholds)
        
        summary = {
            "model_path": model_path,
            "run_path": run_path,
            "split": args.split,
            **metrics,
        }
        
        print(f"\n=== Test Empty Evaluation (False Positive Analysis) ===")
        print(f"Average predictions across all thresholds (0.10 - 0.90)")
        print(f"\n--- Summary Statistics ---")
        print(f"Total images: {summary['total_images']}")
        print(f"Total average predictions (false positives): {summary['total_avg_predictions']:.2f}")
        print(f"Images with predictions: {summary['images_with_predictions']}")
        print(f"Images without predictions: {summary['images_without_predictions']}")
        print(f"Mean predictions per image: {summary['mean_predictions_per_image']:.2f}")
        print(f"Max predictions in single image: {summary['max_predictions_in_single_image']:.2f}")
        print(f"Min predictions in single image: {summary['min_predictions_in_single_image']:.2f}")
        
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        focused_summary = {
            "model": model_name,
            "split": summary["split"],
            "total_images": summary["total_images"],
            "total_avg_predictions": summary["total_avg_predictions"],
            "images_with_predictions": summary["images_with_predictions"],
            "images_without_predictions": summary["images_without_predictions"],
            "mean_predictions_per_image": summary["mean_predictions_per_image"],
            "max_predictions_in_single_image": summary["max_predictions_in_single_image"],
            "min_predictions_in_single_image": summary["min_predictions_in_single_image"],
            "predictions_per_image": summary["predictions_per_image"],
        }
        
        save_eval_summary(run_path, focused_summary, split=args.split)
    else:
        stats, thresholds = run_evaluation(model, dataloader, args.nms_radius, args.match_dist)
        metrics = aggregate_metrics(stats, thresholds, dataset, args.match_dist)

        summary = {
            "model_path": model_path,
            "run_path": run_path,
            "split": args.split,
            **metrics,
        }

        th = summary["optimal_threshold"]
        loc_all = summary.get("mean_loc_error_all_px")
        loc_matched = summary.get("mean_loc_error_matched_px")
        recall = summary.get("recall", 0.0)
        TP = summary.get("TP", 0)
        FN = summary.get("FN", 0)
        total_gt = summary["total_gt"]

        print(f"\nOptimal Heatmap Threshold:      {th:.2f}")
        print(f"F1-Score:                      {summary['f1_score']:.3f}")
        print(f"\n--- Image-Level Counting ---")
        print(f"Count MAE (per image):         {summary['count_mae']:.3f}")
        print(f"\n--- Plant-Instance Level Counting ---")
        print(f"Recall (sensitivity):           {recall*100:.2f}%")
        print(f"Correctly Identified (TP):     {TP}/{total_gt} plants")
        print(f"Missed Plants (FN):             {FN}/{total_gt} plants")
        print(f"\n--- Localization ---")
        if loc_all is not None:
            print(f"Mean localization error (all GT, capped at {args.match_dist:.1f} px): "
                  f"{loc_all:.3f} px")
        if loc_matched is not None:
            print(f"Mean localization error (matched GT only):             {loc_matched:.3f} px")

        model_name = os.path.splitext(os.path.basename(model_path))[0]

        focused_summary = {
            "model": model_name,
            "split": summary["split"],
            "match_distance_pixels": summary["match_dist"],
            "optimal_heatmap_threshold": summary["optimal_threshold"],
            "f1_score": summary["f1_score"],
            "mean_count_error_per_image": summary["count_mae"],
            "recall": summary["recall"],
            "TP": summary["TP"],
            "FP": summary["FP"],
            "false_negatives": summary["FN"],
            "total_ground_truth_plants": summary["total_gt"],
            "mean_localization_error_all_pixels": summary.get("mean_loc_error_all_px"),
            "mean_localization_error_matched_pixels": summary.get("mean_loc_error_matched_px"),
        }

        save_eval_summary(run_path, focused_summary, split=args.split)


if __name__ == "__main__":
    args = parse_args()
    main(args)
