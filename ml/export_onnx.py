import argparse
import os

import onnx
import torch

from utils.model import UNetTiny
import warnings

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a PyTorch UNet model to ONNX format.")
    parser.add_argument("--model", "--m", type=str, required=True, help="Path to the trained PyTorch model checkpoint (.pt).")
    return parser.parse_args()


def load_model(model_path: str) -> tuple:
    """
    Load trained UNetTiny model from checkpoint file to CPU for ONNX export.

    Args:
        model_path (str): Path to the trained model checkpoint.

    Returns:
        tuple: (model, input_shape) where model is the loaded UNetTiny model
            in evaluation mode, and input_shape is (1, 3, 512, 512).
    """
    device = torch.device("cpu")
    input_shape = (1, 3, 512, 512)

    model = UNetTiny(in_ch=3, out_ch=1).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model, input_shape


def export_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): Trained PyTorch model in evaluation mode.
        dummy_input (torch.Tensor): Example input tensor defining input shape.
        output_path (str): Destination path for the exported ONNX model.
    """
    print("Exporting model to ONNX...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=18,
            input_names=["input"],
            output_names=["mask"],
            dynamic_axes={
                "input": {0: "batch_size", 2: "height", 3: "width"},
                "mask": {0: "batch_size", 2: "height", 3: "width"},
            },
            dynamo=False,
        )

    print("ONNX export completed.")


def validate_onnx(onnx_path: str) -> None:
    """
    Validate the exported ONNX model for correctness.

    Args:
        onnx_path (str): Path to the exported ONNX model file.
    """
    print("Validating ONNX model...")

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print("ONNX model is valid.")


def main() -> None:
    args = parse_args()

    model, input_shape = load_model(args.model)
    dummy_input = torch.randn(*input_shape)

    model_name = os.path.splitext(os.path.basename(args.model))[0]
    output_path = os.path.join(f"{model_name}_export.onnx")

    export_onnx(model, dummy_input, output_path)
    validate_onnx(output_path)


if __name__ == "__main__":
    main()
