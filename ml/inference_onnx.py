import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

CONFIG = {
    "optimal_threshold": 0.2,
    "nms_radius": 3,
    "image_size": 512,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference using an ONNX UNet model.")
    parser.add_argument("--image", "--i", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--model", "--m", type=str, default="UNetTiny_aug_sigma3.0_osw12.0_sm2.0_val0.001647_export.onnx", help="Path to the ONNX model file.")
    parser.add_argument("--novisual", "--nv", action="store_true", help="Skip visualization plot generation.")
    return parser.parse_args()


def preprocess_image(img_path: str, target_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Load and preprocess an image for ONNX inference.

    Args:
        img_path (str): Path to the input image file.
        target_size (int): Target spatial size expected by the model.

    Returns:
        tuple: (img_np, (original_w, original_h)) where img_np is preprocessed
            image array (1, 3, H, W) and original_w, original_h are original
            image dimensions.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {img_path}")

    original_h, original_w = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img, (original_w, original_h)


def heatmap_to_centroids(heatmap: torch.Tensor, threshold: float, radius: int) -> List[Tuple[float, float]]:
    """
    Extract centroid coordinates from a density heatmap using non-maximum suppression.

    Args:
        heatmap: Density heatmap tensor of shape (H, W), (C, H, W), or (1, C, H, W).
        threshold: Minimum heatmap value to consider as a detection.
        radius: NMS radius in pixels.

    Returns:
        List of (x, y) tuples representing detected plant centroids.
    """
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)

    kernel = 2 * radius + 1
    pooled = F.max_pool2d(heatmap, kernel_size=kernel, stride=1, padding=radius)
    maxima = (heatmap == pooled) & (heatmap > threshold)

    coords = maxima.nonzero(as_tuple=False)
    return [(float(x), float(y)) for _, _, y, x in coords]


def rescale_centroids(centroids: List[Tuple[float, float]], input_size: int, original_w: int, original_h: int,) -> List[Tuple[float, float]]:
    """
    Rescale centroid coordinates from model input size to original image dimensions.

    Args:
        centroids: List of (x, y) tuples in model input coordinate space.
        input_size: Size of the model input (assumed square).
        original_w: Original image width.
        original_h: Original image height.

    Returns:
        List of (x, y) tuples scaled to original image dimensions.
    """
    scale_x = original_w / input_size
    scale_y = original_h / input_size

    return [(x * scale_x, y * scale_y) for x, y in centroids]


def save_results(results: Dict[str, Any], output_dir: str, image_name: str, image_path: str, visualize: bool = True) -> Dict[str, str]:
    """
    Save inference results to JSON and optionally visualize them on the image.

    Args:
        results (Dict): Inference results dictionary containing 'count', 'centroids',
            and 'inference_time_sec'.
        output_dir (str): Directory path where results will be saved.
        image_name (str): Base name of the image (without extension) for naming output files.
        image_path (str): Path to the original input image (needed for visualization).
        visualize (bool, optional): Whether to generate and save visualization plot.
            Defaults to True.

    Returns:
        Dict[str, str]: Dictionary with paths to saved files ('json_path' and
            optionally 'image_path').
    """
    os.makedirs(output_dir, exist_ok=True)
    
    json_path = os.path.join(output_dir, f"{image_name}_predictions.json")
    
    with open(json_path, "w") as f:
        json.dump(
            {
                "count": results["count"],
                "centroids": np.round(
                    np.array(results["centroids"]), 2
                ).tolist(),
            },
            f,
            indent=4,
        )
    
    saved_paths = {"json_path": json_path}
    
    if visualize:
        image_out_path = os.path.join(output_dir, f"{image_name}_predictions.bmp")
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        for x, y in results["centroids"]:
            center = (int(round(x)), int(round(y)))
            cv2.circle(img, center, radius=3, color=(0, 0, 255), thickness=-1)
        
        cv2.imwrite(image_out_path, img)
        saved_paths["image_path"] = image_out_path
    
    return saved_paths


def run_inference(img_path: str, session: ort.InferenceSession, target_size: int) -> Dict[str, Any]:
    """
    Run full ONNX inference pipeline on a single image.

    Args:
        img_path: Path to the input image file.
        session: ONNX Runtime InferenceSession for model execution.
        target_size: Target image size for model input (assumed square).

    Returns:
        Dict: Inference results.
    """
    img_np, (original_w, original_h) = preprocess_image(img_path, target_size)

    input_name = session.get_inputs()[0].name

    t0 = time.time()
    outputs = session.run(None, {input_name: img_np})
    inference_time = time.time() - t0

    heatmap = torch.from_numpy(outputs[0][0, 0])

    centroids_model = heatmap_to_centroids(
        heatmap,
        threshold=CONFIG["optimal_threshold"],
        radius=CONFIG["nms_radius"],
    )

    centroids_rescaled = rescale_centroids(
        centroids_model,
        input_size=target_size,
        original_w=original_w,
        original_h=original_h,
    )

    return {
        "count": len(centroids_rescaled),
        "centroids": centroids_rescaled,
        "inference_time_sec": inference_time,
    }


def main(args: argparse.Namespace) -> None:
    image_path = args.image
    model_path = args.model

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = os.path.join(os.getcwd(), "inference_onnx", image_name)

    results = run_inference(image_path, session, CONFIG["image_size"])

    visualize = not getattr(args, 'novisual', False)
    
    saved_paths = save_results(
        results=results,
        output_dir=output_dir,
        image_name=image_name,
        image_path=image_path,
        visualize=visualize
    )

    print(f"\nFile:           {os.path.basename(image_path)}")
    print(f"Total Count:    {results['count']}")
    print(f"Inference Time: {results['inference_time_sec']:.4f} seconds")
    print(f"JSON saved to:  {saved_paths['json_path']}")
    if visualize:
        print(f"Image saved to: {saved_paths['image_path']}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
