#!/usr/bin/env python3
"""
YOLO Dataset Train/Validation Splitter

This script splits a YOLO format dataset from a single 'train' folder into
train/val splits based on a specified percentage and updates dataset.yaml accordingly.

Expected input structure:
dataset/
├── images/
│   └── train/
│       ├── img1.jpg
│       └── ...
├── labels/
│   └── train/
│       ├── img1.txt
│       └── ...
└── dataset.yaml

Output structure:
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml (updated)
"""

import argparse
import os
import random
import shutil
import yaml
from pathlib import Path
from typing import List, Tuple


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset from train-only to train/val splits"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset root directory"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Percentage for validation split (default: 0.2 for 20%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)"
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them (default: move)"
    )

    return parser.parse_args()


def validate_dataset_structure(dataset_path: Path) -> Tuple[Path, Path]:
    """Validate the dataset structure and return paths to images and labels."""
    images_train_path = dataset_path / "images" / "train"
    labels_train_path = dataset_path / "labels" / "train"

    if not images_train_path.exists():
        raise FileNotFoundError(f"Images train directory not found: {images_train_path}")

    if not labels_train_path.exists():
        raise FileNotFoundError(f"Labels train directory not found: {labels_train_path}")

    return images_train_path, labels_train_path


def get_image_files(images_path: Path) -> List[Path]:
    """Get all image files from the train directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    for file_path in images_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)

    if not image_files:
        raise ValueError(f"No image files found in {images_path}")

    return image_files


def split_files(image_files: List[Path], val_split: float, seed: int) -> Tuple[List[Path], List[Path]]:
    """Split image files into train and validation sets."""
    random.seed(seed)
    random.shuffle(image_files)

    val_count = int(len(image_files) * val_split)
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]

    return train_files, val_files


def create_directories(dataset_path: Path):
    """Create val directories for images and labels."""
    (dataset_path / "images" / "val").mkdir(exist_ok=True)
    (dataset_path / "labels" / "val").mkdir(exist_ok=True)


def move_or_copy_files(files: List[Path], src_type: str, dst_type: str,
                       dataset_path: Path, copy_files: bool = False):
    """Move or copy files from source to destination directory."""
    operation = shutil.copy2 if copy_files else shutil.move
    operation_name = "Copying" if copy_files else "Moving"

    for file_path in files:
        # Handle image files
        img_src = dataset_path / "images" / src_type / file_path.name
        img_dst = dataset_path / "images" / dst_type / file_path.name

        if img_src.exists():
            operation(str(img_src), str(img_dst))

        # Handle corresponding label files
        label_name = file_path.stem + '.txt'
        lbl_src = dataset_path / "labels" / src_type / label_name
        lbl_dst = dataset_path / "labels" / dst_type / label_name

        if lbl_src.exists():
            operation(str(lbl_src), str(lbl_dst))
        else:
            print(f"Warning: Label file not found for {file_path.name}: {lbl_src}")


def update_dataset_yaml(dataset_path: Path, val_split: float):
    """Update or create dataset.yaml file with train/val paths."""
    yaml_path = dataset_path / "dataset.yaml"

    # Load existing yaml or create new structure
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    # Update paths
    data['train'] = str(dataset_path / "images" / "train")
    data['val'] = str(dataset_path / "images" / "val")

    # If test path doesn't exist, don't include it
    if 'test' not in data:
        data['test'] = None

    # Add metadata about the split
    data['split_info'] = {
        'val_percentage': val_split,
        'split_method': 'random'
    }

    # If no class names exist, add placeholder
    if 'names' not in data:
        print("Warning: No class names found in dataset.yaml. You may need to add them manually.")
        data['names'] = ['class0']  # Placeholder

    # Write updated yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Updated {yaml_path}")


def main():
    """Main function to execute the dataset splitting."""
    args = parse_arguments()

    # Validate arguments
    if not 0 < args.val_split < 1:
        raise ValueError("val_split must be between 0 and 1")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    print(f"Processing dataset: {dataset_path}")
    print(f"Validation split: {args.val_split:.1%}")
    print(f"Random seed: {args.seed}")
    print(f"Operation: {'Copy' if args.copy else 'Move'}")

    # Validate dataset structure
    images_train_path, labels_train_path = validate_dataset_structure(dataset_path)

    # Get all image files
    image_files = get_image_files(images_train_path)
    print(f"Found {len(image_files)} images")

    # Split files
    train_files, val_files = split_files(image_files, args.val_split, args.seed)
    print(f"Split: {len(train_files)} train, {len(val_files)} val")

    # Create validation directories
    create_directories(dataset_path)

    # Move/copy validation files
    print("Processing validation files...")
    move_or_copy_files(val_files, "train", "val", dataset_path, args.copy)

    # Update dataset.yaml
    update_dataset_yaml(dataset_path, args.val_split)

    print("Dataset splitting completed successfully!")
    print(f"Train samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")


if __name__ == "__main__":
    main()