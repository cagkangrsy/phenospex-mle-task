import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.model import UNetTiny
from utils.train_core import train_one_epoch, validate, get_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("UNetTiny Training for Plant Counting")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default=None, help='Explicit device to use (e.g., "cuda" or "cpu"). Overrides automatic detection.')
    parser.add_argument('--dataset', type=str, default='data', help='Path to the data folder containing images and annotations subdirectories.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation.')
    parser.add_argument('--img_size', type=int, default=512, help='Input image size (assumed to be square).')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for data loading.')
    parser.add_argument('--gaussian_sigma', type=float, default=3.0, help='Standard deviation for Gaussian kernel used to create density maps.')
    parser.add_argument('--oversampling_weight', type=float, default=25.0, help='Weight for oversampling minority classes (if used in custom dataloader).')
    parser.add_argument('--sampler_multiplier', type=float, default=2.0, help='Multiplier for the size of the training set when using weighted sampling.')
    parser.add_argument('--augment', type=int, choices=[0, 1], default=0, help='Use data augmentation during training: 1=on (default), 0=off.')
    parser.add_argument('--epochs', type=int, default=300, help='Maximum number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate for the Adam optimizer.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 regularization coefficient.')
    parser.add_argument('--patience_es', type=int, default=15, help='Patience for Early Stopping (epochs without validation loss improvement).')
    parser.add_argument('--patience_lr', type=int, default=5, help='Patience for LR scheduler (epochs without validation loss improvement before dropping LR).')
    parser.add_argument('--lr_drop_factor', type=float, default=0.1, help='Factor by which the learning rate will be reduced on plateau.')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate.')

    args = parser.parse_args()

    if args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across PyTorch operations.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_name(args: argparse.Namespace) -> str:
    """
    Build a run name that encodes the most relevant hyperparameters.

    Returns:
        str: Run name string with timestamp and hyperparameter encoding.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aug_tag = "aug_" if args.augment else ""
    return (
        f"{timestamp}_UNetTiny_{aug_tag}"
        f"sigma{args.gaussian_sigma}_"
        f"osw{args.oversampling_weight}_"
        f"sm{args.sampler_multiplier}"
    )


def prepare_run_directories(run_name: str) -> str:
    """
    Create run directory for saving training artifacts.
    """
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def setup_model_and_optimizer(args: argparse.Namespace, device: torch.device) -> tuple:
    """
    Construct the model, loss function, optimizer and LR scheduler.

    Args:
        args (argparse.Namespace): Parsed arguments containing training configuration.
        device (torch.device): Device to place model on.

    Returns:
        tuple: (model, criterion, optimizer, scheduler) tuple.
    """
    model = UNetTiny(in_ch=3, out_ch=1).to(device)

    criterion = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_drop_factor,
        patience=args.patience_lr,
        min_lr=args.min_lr,
    )

    print(
        f"Model: UNetTiny | "
        f"Loss: {criterion.__class__.__name__} | "
        f"Optimizer: {optimizer.__class__.__name__} | "
        f"Scheduler: {scheduler.__class__.__name__}"
    )

    return model, criterion, optimizer, scheduler


def train_model(args: argparse.Namespace, model: nn.Module, optimizer: Adam, scheduler: ReduceLROnPlateau, criterion: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, run_dir: str) -> tuple:
    """
    Full training loop with validation, checkpointing, and early stopping.

    Args:
        args (argparse.Namespace): Parsed arguments containing training configuration.
        model (nn.Module): PyTorch model to train.
        optimizer (Adam): Optimizer for updating model parameters.
        scheduler (ReduceLROnPlateau): Learning rate scheduler.
        criterion (nn.Module): Loss function.
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        device (torch.device): Device to run training on.
        run_dir (str): Directory path for saving checkpoints.

    Returns:
        tuple: (epoch_logs, best_epoch, best_val_loss, best_train_loss, total_time).
    """
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_model_path = None

    epoch_logs = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.epochs)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]

        epoch_logs.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": lr_now,
            })

        print(f"LR {lr_now:.2e} | " f"Train {train_loss:.6f} | " f"Val {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_train_loss = train_loss
            best_epoch = epoch
            patience_counter = 0

            os.makedirs(run_dir, exist_ok=True)

            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)

            aug_tag_ckpt = "aug_" if args.augment else ""
            model_name = (
                f"UNetTiny_{aug_tag_ckpt}"
                f"sigma{args.gaussian_sigma}_"
                f"osw{args.oversampling_weight}_"
                f"sm{args.sampler_multiplier}_"
                f"val{val_loss:.6f}.pt"
            )

            best_model_path = os.path.join(run_dir, model_name)
            torch.save(model.state_dict(), best_model_path)

            print(
                f"  >>> Best model updated at epoch {epoch} "
                f"(Val Loss={best_val_loss:.6f})"
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience_es:
            print("\nEarly stopping triggered.")
            break

    total_time = time.time() - start_time

    return epoch_logs, best_epoch, best_val_loss, best_train_loss, total_time


def save_run_artifacts(args: argparse.Namespace, run_name: str, run_dir: str, epoch_logs: list, best_epoch: int, best_val_loss: float, best_train_loss: float, total_time: float) -> None:
    """
    Save hyperparameters, per-epoch logs, and high-level run summary.

    Args:
        args (argparse.Namespace): Parsed arguments containing hyperparameters.
        run_name (str): Unique run name.
        run_dir (str): Directory path for saving artifacts.
        epoch_logs (list): List of per-epoch metric dictionaries.
        best_epoch (int): Epoch number with best validation loss.
        best_val_loss (float): Best validation loss value.
        best_train_loss (float): Training loss at best epoch.
        total_time (float): Total training time in seconds.
    """
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    if epoch_logs:
        with open(os.path.join(run_dir, "training_log.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
            writer.writeheader()
            writer.writerows(epoch_logs)

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_train_loss": best_train_loss,
        "total_training_time_s": total_time,
    }

    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    summary_csv = os.path.join("runs", "runs_summary.csv")
    write_header = not os.path.exists(summary_csv)

    with open(summary_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(summary)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device

    run_name = create_run_name(args)
    run_dir = prepare_run_directories(run_name)
    run_success = False

    print("=" * 60)
    print(f"Run: {run_name}")
    print(f"Device: {device}")
    print("=" * 60)

    try:
        generator = torch.Generator().manual_seed(args.seed)

        train_loader, val_loader, train_len, val_len = get_dataloaders(img_size=args.img_size,
                                                                       sigma=args.gaussian_sigma,
                                                                       augment=args.augment,
                                                                       os_wt=args.oversampling_weight,
                                                                       s_multip=args.sampler_multiplier,
                                                                       batch_size=args.batch_size,
                                                                       n_workers=args.num_workers,
                                                                       generator=generator,
                                                                       root=args.dataset)

        print(f"Train samples: {train_len} | Val samples: {val_len}\n")

        model, criterion, optimizer, scheduler = setup_model_and_optimizer(args, device)

        epoch_logs, best_epoch, best_val_loss, best_train_loss, total_time = train_model(args=args,
                                                                                         model=model,
                                                                                         optimizer=optimizer,
                                                                                         scheduler=scheduler,
                                                                                         criterion=criterion,
                                                                                         train_loader=train_loader,
                                                                                         val_loader=val_loader,
                                                                                         device=device,
                                                                                         run_dir=run_dir)

        save_run_artifacts(args=args,
                           run_name=run_name,
                           run_dir=run_dir,
                           epoch_logs=epoch_logs,
                           best_epoch=best_epoch,
                           best_val_loss=best_val_loss,
                           best_train_loss=best_train_loss,
                           total_time=total_time)

        run_success = True

        print("\n" + "=" * 60)
        print("Training complete")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Artifacts saved to: {run_dir}")
        print("=" * 60)

    finally:
        if not run_success and os.path.exists(run_dir):
            print("\nTraining failed — removing run directory")
            shutil.rmtree(run_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)