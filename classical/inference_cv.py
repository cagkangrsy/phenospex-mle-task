import argparse
import time

from cv_pipeline import run as run_cv_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run classical CV pipeline on a single image")
    parser.add_argument("--image", "--i", type=str, required=True, help="Path to the input image file.")
    return parser.parse_args()


def main(args: argparse.Namespace) -> float:
    """
    Run classical CV pipeline inference on a single image.

    Args:
        args (argparse.Namespace): Parsed arguments containing image path.

    Returns:
        float: Inference runtime in seconds.
    """
    image_path = args.image

    start_time = time.perf_counter()
    centroids = run_cv_pipeline(image_path, save=True)
    end_time = time.perf_counter()
    runtime = end_time - start_time

    print(f"Image: {image_path}")
    print(f"Detected plants: {len(centroids)}")
    print(f"Runtime: {runtime:.4f} seconds")
    return runtime


if __name__ == "__main__":
    args = parse_args()
    main(args)
