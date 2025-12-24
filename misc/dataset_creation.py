import glob
import os
import shutil
import random

SEED = 42
OUTPUT_DIR = "../ml/data/images"
dataset_ids = ["11", "12", "13", "14", "20"]
source_file = "thumbnail.bmp"
extension = ".bmp"
split_ratios = {
    "train": 0.70,
    "val": 0.20,
    "test": 0.10,
}


def collect_dataset_files(dataset_ids, source_file, extension):
    """Collect all file paths grouped by dataset ID.

    Args:
        dataset_ids (list): List of dataset ID strings.
        source_file (str): Source filename to look for (e.g., "thumbnail.bmp").
        extension (str): File extension to append to new names.

    Returns:
        dict: Dictionary mapping dataset ID to list of (source_path, new_name) tuples.
    """
    dataset_files = {}

    for id in dataset_ids:
        path_pattern = os.path.join("..", id, "*")
        folders = sorted(glob.glob(path_pattern))  # Sort for deterministic order

        files_to_copy = []
        for folder in folders:
            source_file_path = os.path.join(folder, source_file)

            if os.path.isfile(source_file_path):
                new_name = f"{id}_{os.path.basename(folder)}{extension}"
                files_to_copy.append((source_file_path, new_name))
            else:
                print(f"No {source_file} found in: {folder}")

        # Sort by new_name to ensure deterministic order before shuffling
        files_to_copy.sort(key=lambda x: x[1])
        dataset_files[id] = files_to_copy

    return dataset_files


def create_splits(file_list, split_ratios, dataset_id):
    """
    Create train/val/test splits for a dataset.

    Args:
        file_list (list): List of (source_path, new_name) tuples.
        split_ratios (dict): Dictionary with "train", "val", "test" ratios.
        dataset_id (str): Dataset ID string.

    Returns:
        dict: Dictionary mapping split names to (files, count) tuples.
    """
    # Create a copy and sort to ensure deterministic shuffle input
    file_list = sorted(file_list, key=lambda x: x[1])
    random.shuffle(file_list)
    N = len(file_list)

    if dataset_id == "14":
        if N < 2:
            print("Dataset 14 should have exactly 2 files. Found:", N)
            return None

        return {
            "train": (file_list[:1], 1),
            "val": ([], 0),
            "test": (file_list[1:2], 1),
        }
    else:
        n_train = int(N * split_ratios["train"])
        n_val = int(N * split_ratios["val"])
        n_test = N - n_train - n_val

        val_start = n_train
        test_start = n_train + n_val

        return {
            "train": (file_list[:val_start], n_train),
            "val": (file_list[val_start:test_start], n_val),
            "test": (file_list[test_start:], n_test),
        }


def main():
    random.seed(SEED)
    base_dir = OUTPUT_DIR

    print("=" * 50)
    print("Processing BMP files")
    print(f"Source file pattern: {source_file}")
    print(f"Random seed: {SEED} (deterministic)")
    print(f"Output directory: {base_dir}")
    print("=" * 50)

    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(train_dir)
    os.makedirs(val_dir)
    os.makedirs(test_dir)

    dataset_files = collect_dataset_files(dataset_ids, source_file, extension)
    total_copied_files = 0

    for id, file_list in dataset_files.items():
        if not file_list:
            print(f"No files found for dataset ID {id}. Skipping split.")
            continue

        N = len(file_list)
        print(f"\n--- Processing Dataset ID {id} ({N} files) ---")

        splits = create_splits(file_list, split_ratios, id)
        if splits is None:
            continue

        if id == "14":
            print(" Special handling for Dataset 14:")
            print("  - 1 file → train")
            print("  - 1 file → test")
            print("  - 0 files → val")

        split_dirs = {
            "train": train_dir,
            "val": val_dir,
            "test": test_dir,
        }

        for split_name, (files, count) in splits.items():
            print(f"  - Assigning {count} files to {split_name}")
            dest_dir = split_dirs[split_name]

            for source_path, new_name in files:
                destination_path = os.path.join(dest_dir, new_name)
                shutil.copy(source_path, destination_path)
                total_copied_files += 1

    print("\n" + "=" * 50)
    print("Data splitting complete for BMP files")
    print(f"Total files copied: {total_copied_files}")
    print(f"Train directory: {train_dir} ({len(os.listdir(train_dir))} files)")
    print(f"Validation directory: {val_dir} ({len(os.listdir(val_dir))} files)")
    print(f"Test directory: {test_dir} ({len(os.listdir(test_dir))} files)")
    print(f"Random seed used: {SEED} (deterministic split)")
    print("=" * 50)


if __name__ == "__main__":
    main()
