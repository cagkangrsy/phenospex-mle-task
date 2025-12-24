<div align="center">
  <h2 style="font-size: 3.2em; margin-bottom: 0.2em;">PHENOSPEX</h2>
  <h3 style="font-size: 1.9em; margin-top: 0.2em; margin-bottom: 0.3em;">
    Machine Learning Engineer – Take-Home Assignment
  </h3>
  <p style="font-size: 1.35em; margin-top: 0;">
    <strong>Plant Counting and Localization Under Edge Constraints</strong>
  </p>
</div>

<div align="center">
  <p style="font-size: 1.1em;">
    📄 <strong>Technical Report:</strong>
    <a href="./report.pdf">
      Machine Learning Engineer – Take Home Assignment (PDF)
    </a>
  </p>
</div>



## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Dataset Utilities (Optional)](#dataset-utilities)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Inference (ONNX – Recommended)](#ml-inference-onnx)
  - [Training](#ml-training)
  - [Evaluation](#ml-evaluation)
  - [Inference (PyTorch)](#ml-inference-pytorch)
  - [Model Export (PyTorch → ONNX)](#ml-export-onnx)
- [Classical Computer Vision Pipeline](#classical-computer-vision-pipeline)
  - [Inference](#cv-inference)
  - [Evaluation](#cv-evaluation)

    
## Overview

This repository contains the implementation for the Phenospex Machine Learning Engineer take-home assignment.

The objective of the codebase is to **count plants and localize their centroids** in
top-down 2D images under practical deployment constraints (offline execution, CPU-only
inference, low memory usage).

The repository includes:
- A **machine learning pipeline** based on heatmap regression (training, evaluation, inference, ONNX export)
- A **classical computer vision baseline** for comparison
- Dataset utilities for preparation, augmentation, and annotation
- A runnable **demo notebook** for end-to-end inspection

> **Note**  
> This README is **implementation-focused**.  
> A separate report accompanies this for the problem formulation, methodological rationale, experiments, and results.

All scripts are designed to be **reproducible** and runnable on CPU-only systems.

---

## Quick Start

The fastest way to inspect the model results is via the provided demo notebook.

### Demo Notebook
- **Notebook**: `demo/demo.ipynb`
- **Data directory**: `demo/data/`

The demo directory already contains sample images that can be used immediately to run
both the **machine learning** and **classical computer vision** pipelines.

Additional images can be placed in `demo/data/` to observe model behavior on new inputs
without modifying the code.

### Steps
1. Install dependencies (see *Installation*)
2. Open `demo/demo.ipynb`
3. Run all cells to view inference results and visualizations

---

## Project Structure

The repository is organized by functionality to separate the machine learning pipeline,
the classical baseline, and supporting utilities.

```
phenospex-ml-engineer-task/
├── ml/ # Machine learning pipeline
│ ├── data/ # Dataset (images, annotations, point data)
│ ├── utils/ # Dataset, heatmap, training, and evaluation utilities
│ ├── train.py # Model training
│ ├── eval.py # Model evaluation
│ ├── inference.py # PyTorch-based inference
│ ├── inference_onnx.py# ONNX Runtime inference (recommended)
│ └── export_onnx.py # PyTorch → ONNX model export
│
├── classical/ # Classical computer vision baseline
│ ├── inference_cv.py # CV-based plant counting and localization
│ └── eval_cv.py # Evaluation of the CV baseline
│
├── demo/ # End-to-end demo
│ ├── demo.ipynb
│ └── data/ # Demo images (extensible)
│
├── misc/ # Dataset creation and annotation utilities (reference)
└── README.md
```

---

## Installation

### Requirements
- Python >3.10
- PyTorch
- (Optional) CUDA-capable GPU for faster training

### Environment Setup

Create and activate a clean Python environment (recommended):

```bash
conda create -n psx-ml python=3.10 -y
conda activate psx-ml
```

### PyTorch Installation

#### GPU (CUDA) Installation
Use this option if a CUDA-capable GPU is available.

Install the PyTorch build that matches your local CUDA version.
Refer to the official PyTorch installation guide to select the correct command:

https://pytorch.org/get-started/locally/

Example (replace with the appropriate CUDA version for your system):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

#### CPU-only Installation
Use this option for CPU-only execution.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import torch; print(torch.__version__)"
```

(Optional) Verify CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Data Format

The dataset included in the submisison follows a fixed directory and annotation format
expected by all machine learning and evaluation scripts.

> Copy the dataset included with the submission directly under `ml` to create `ml/data/`

### Directory Structure

**Default location:** `ml/data/`

```
ml/data/
├── images/
│   ├── train/          # Training images
│   ├── val/            # Validation images
│   ├── test/           # Test images
│   └── test_aug/       # Augmented test images
│
└── annotations/
    ├── train.json      # Training annotations
    ├── val.json        # Validation annotations
    ├── test.json       # Test annotations
    └── test_aug.json   # Augmented test annotations
```

All machine learning scripts assume this directory structure by default.
If a different dataset location is required, the dataset root can be modified
via the corresponding script arguments.

---

<a id="dataset-utilities"></a>
## Dataset Utilities (Optional)

> **Note**  
> The scripts in this section are provided for **reference and reproducibility**.  
> Annotation requires **manual interaction** and can be time-consuming; therefore,
> the **pre-prepared dataset included in the submission is recommended** for use.

Utility scripts for dataset preparation, augmentation, and annotation are located
in the `misc/` directory.

### Raw Data Requirements

Dataset creation scripts expect raw data folders (`11/`, `12/`, `13/`, `14/`, `20/`)
to be placed at the **repository root level**. Each folder is treated as an
independent acquisition batch.

These raw folders are used by the scripts in `misc/` to generate the structured
dataset under `ml/data/`.

### Usage

Before running any utility scripts, navigate to the directory:

```bash
cd misc
```

---

### Dataset Creation

Creates train/validation/test splits from raw dataset folders.

```bash
python dataset_creation.py
```

**Purpose**
- Generates deterministic train/val/test splits
- Renames images consistently
- Preserves grid distribution across splits

---

### Augmented Test Set Creation

Creates an augmented test set from the original test split.

```bash
python create_augmented_test_set.py
```

**Augmentations**
- Discrete rotations
- Gaussian noise
- Random plant removal to break grid structure

**Outputs**
- Augmented images saved to `ml/data/images/test_aug/`
- Updated annotations saved to `ml/data/annotations/test_aug.json`

---

### Annotation Tool

Semi-automatic annotation tool for creating centroid ground truth.

```bash
python annotate.py
```

**Features**
- Grid-based automatic centroid interpolation
- Manual annotation mode for refinement and fallback.
- Interactive editing and correction

---

## Machine Learning Pipeline

All machine learning–related scripts are located in the `ml/` directory.

Before running any ML scripts, navigate to the directory:

```bash
cd ml
```

---
<a id="ml-inference-onnx"></a>
### Inference (ONNX – Recommended)

Runs plant counting and centroid localization on a single image using an
ONNX-exported model. This is the **recommended inference method** for deployment
and CPU-only execution.

```bash
python inference_onnx.py --image [image.bmp] --model [model.onnx]
```

**Arguments**
- `--image` / `--i` (required): Path to input image (BMP, PNG, JPEG)
- `--model` / `--m` (optional): Path to ONNX model (defaults to model in repo)
- `--novisual` / `--nv` (optional): Disable visualization output

**Outputs**
Saved to `ml/inference_onnx/{image_name}/`:
- `{image_name}_predictions.json`
- `{image_name}_predictions.bmp` (if visualization is enabled)

---
<a id="ml-training"></a>
### Training

Trains a UNetTiny model on the prepared dataset.

```bash
python train.py --batch_size [BATCH_SIZE] --epochs [EPOCHS] --augment [AUGMENT] --learning_rate [LEARNING_RATE]
```

**Key Arguments**
- `--batch_size`: Batch size
- `--epochs`: Maximum number of epochs
- `--learning_rate`: Initial learning rate
- `--img_size`: Input image size
- `--gaussian_sigma`: Gaussian kernel sigma for heatmap generation
- `--augment`: Enable data augmentation (`1` = on, `0` = off)
- `--oversampling_weight`: Weight for oversampling minority grid configurations
- `--sampler_multiplier`: Multiplier for training set size when using weighted sampling
- `--device`: `cpu` or `cuda` (auto-detected if not specified)
- `--patience_es`: Early stopping patience
- `--patience_lr`: Learning-rate scheduler patience

**Outputs**  
Saved to `ml/runs/{run_name}/`:
- Model checkpoints (`UNetTiny_*.pt`)
- Training logs and summary files

---
<a id="ml-evaluation"></a>
### Evaluation

Evaluates a trained model from a training run on a specified dataset split. Writes the results to the folder where model is located

```bash
python eval.py --model [runs/{run_name}/UNetTiny_*.pt] --split [split_name]
```

**Arguments**
- `--model` (required): Path to trained checkpoint
- `--split`: Dataset split (`train`, `val`, `test`, `test_aug`)
- `--nms_radius`: Non-maximum suppression radius
- `--match_dist`: Maximum matching distance in pixels
- `--device`: `cpu` or `cuda`

**Outputs**
- Evaluation summary saved to `{run_name}/eval_summary_{split}.json`

---
<a id="ml-inference-pytorch"></a>
### Inference (PyTorch)

Runs inference using a PyTorch checkpoint. This is provided for completeness;
ONNX inference is recommended for deployment.

```bash
python inference.py --image [image.bmp] --model [model_path]
```

**Arguments**
- `--image` / `--i` (required): Path to input image
- `--model` / `--m` (optional): Path to PyTorch model (Defaults to model in repo)
- `--novisual` / `--nv` (optional): Disable visualization output

**Outputs**
Saved to `ml/inference/{image_name}/`:
- `{image_name}_predictions.json`
- `{image_name}_predictions.bmp` (if visualization is enabled)

---
<a id="ml-export-onnx"></a>
### Model Export (PyTorch → ONNX)

Exports a trained PyTorch checkpoint to ONNX format and validates the exported model.

```bash
python export_onnx.py --model [model_path]
```

**Output**
- Exported ONNX model saved to `ml/{model_name}_export.onnx`

---

## Classical Computer Vision Pipeline

The classical baseline is implemented in the `classical/` directory and provides a
rule-based alternative to the machine learning approach.

Before running any scripts, navigate to the directory:

```bash
cd classical
```

---
<a id="cv-inference"></a>
### Inference

Runs plant counting and centroid localization using classical computer vision
techniques (background suppression, segmentation, and connected components).

```bash
python inference_cv.py --image [image.bmp]
```

**Arguments**
- `--image` (required): Path to input image (BMP, PNG, JPEG)

**Outputs**  
Saved to `classical/inference/{image_name}/`:
- `{image_name}_cv_predictions.json`
- `{image_name}_cv_predictions.png` (visualization with detected centroids)

---
<a id="cv-evaluation"></a>
### Evaluation

Evaluates the classical CV pipeline on a dataset split.

```bash
python eval_cv.py --split [split_name] --match_dist [match_dist]
```

**Arguments**
- `--split`: Dataset split (`train`, `val`, `test`, `test_aug`)
- `--match_dist`: Maximum pixel distance for matching predictions to ground truth

**Outputs**  
- Evaluation summary saved to `classical/eval_summary_{split}.json`