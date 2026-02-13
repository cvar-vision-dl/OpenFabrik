#!/usr/bin/env python3
"""
YOLO Dataset Pruning Script

This script randomly prunes a YOLO dataset by removing a specified percentage of images
and their corresponding labels. Useful for creating smaller datasets for testing,
reducing overfitting, or experimenting with different dataset sizes.

Supports both flat structure (images/, labels/) and split structure (images/train, images/val, etc.)

Requirements:
    pip install tqdm pyyaml

Usage:
    # Prune 20% from all splits
    python yolo_prune.py --input /path/to/dataset --output /path/to/pruned_dataset --prune 20

    # Prune 30% from training split only
    python yolo_prune.py --input /path/to/dataset --output /path/to/pruned --prune 30 --split train

    # Prune 50% from validation split
    python yolo_prune.py --input /path/to/dataset --output /path/to/pruned --prune 50 --split val

    # Prune with specific random seed for reproducibility
    python yolo_prune.py --input /path/to/dataset --output /path/to/pruned --prune 25 --seed 42

    # Dry run to see what would be kept/removed without actually copying files
    python yolo_prune.py --input /path/to/dataset --output /path/to/pruned --prune 20 --dry-run
"""

import argparse
import os
import shutil
import random
import glob
import sys
from pathlib import Path
import yaml
from tqdm import tqdm


class YOLODatasetPruner:
    def __init__(self, input_dataset, output_dataset, prune_percentage, split=None, seed=None, dry_run=False):
        self.input_dataset = Path(input_dataset)
        self.output_dataset = Path(output_dataset)
        self.prune_percentage = prune_percentage
        self.keep_percentage = 100 - prune_percentage
        self.split = split
        self.dry_run = dry_run

        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            print(f"Random seed set to: {seed}")

        # Paths
        self.input_images_dir = self.input_dataset / 'images'
        self.input_labels_dir = self.input_dataset / 'labels'
        self.output_images_dir = self.output_dataset / 'images'
        self.output_labels_dir = self.output_dataset / 'labels'

        # Validate input directories
        if not self.input_images_dir.exists():
            raise ValueError(f"Images directory not found: {self.input_images_dir}")
        if not self.input_labels_dir.exists():
            raise ValueError(f"Labels directory not found: {self.input_labels_dir}")

        # Statistics
        self.stats = {}

    def _find_all_images(self, images_dir, split_name=None):
        """
        Find all image files in a directory or split subdirectory.
        Returns a list of tuples: (absolute_path, relative_path_from_images_dir)
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []

        search_dir = images_dir / split_name if split_name else images_dir

        if not search_dir.exists():
            return []

        for ext in image_extensions:
            found_images = glob.glob(str(search_dir / ext)) + glob.glob(str(search_dir / ext.upper()))
            for img_path in found_images:
                img_path = Path(img_path)
                rel_path = img_path.relative_to(images_dir)
                image_files.append((str(img_path), str(rel_path)))

        return sorted(image_files)

    def _get_label_path(self, image_rel_path):
        """Get the corresponding label file path for an image using its relative path."""
        label_rel_path = Path(image_rel_path).with_suffix('.txt')
        return self.input_labels_dir / label_rel_path

    def _prune_split(self, split_name):
        """Prune a specific split or flat structure."""
        print(f"\nProcessing split: {split_name if split_name else 'flat structure'}")
        print("=" * 60)

        # Find all images in this split
        image_files = self._find_all_images(self.input_images_dir, split_name)

        if not image_files:
            print(f"No images found in {split_name if split_name else 'root directory'}")
            return

        total_images = len(image_files)
        images_to_keep = int(total_images * (self.keep_percentage / 100.0))
        images_to_remove = total_images - images_to_keep

        print(f"Total images: {total_images}")
        print(f"Images to keep ({self.keep_percentage}%): {images_to_keep}")
        print(f"Images to remove ({self.prune_percentage}%): {images_to_remove}")

        # Randomly select images to keep
        images_to_keep_list = random.sample(image_files, images_to_keep)

        # Store statistics
        split_key = split_name if split_name else 'root'
        self.stats[split_key] = {
            'total': total_images,
            'kept': images_to_keep,
            'removed': images_to_remove
        }

        if self.dry_run:
            print("DRY RUN: No files will be copied")
            return

        # Create output directories
        if split_name:
            output_img_split_dir = self.output_images_dir / split_name
            output_lbl_split_dir = self.output_labels_dir / split_name
        else:
            output_img_split_dir = self.output_images_dir
            output_lbl_split_dir = self.output_labels_dir

        output_img_split_dir.mkdir(parents=True, exist_ok=True)
        output_lbl_split_dir.mkdir(parents=True, exist_ok=True)

        # Copy selected images and labels
        copied_images = 0
        copied_labels = 0
        missing_labels = 0

        print(f"\nCopying files...")
        for img_abs_path, img_rel_path in tqdm(images_to_keep_list, desc="Progress", unit="files"):
            # Copy image
            output_img_path = self.output_images_dir / img_rel_path
            try:
                shutil.copy2(img_abs_path, output_img_path)
                copied_images += 1
            except Exception as e:
                tqdm.write(f"Error copying image {img_abs_path}: {e}")
                continue

            # Copy label if it exists
            input_label_path = self._get_label_path(img_rel_path)
            if input_label_path.exists():
                output_label_path = self.output_labels_dir / Path(img_rel_path).with_suffix('.txt')
                try:
                    shutil.copy2(input_label_path, output_label_path)
                    copied_labels += 1
                except Exception as e:
                    tqdm.write(f"Error copying label {input_label_path}: {e}")
            else:
                missing_labels += 1

        print(f"Copied: {copied_images} images, {copied_labels} labels")
        if missing_labels > 0:
            print(f"Warning: {missing_labels} images had no corresponding label files")

    def _copy_yaml_files(self):
        """Copy dataset.yaml or data.yaml if they exist."""
        yaml_files = ['dataset.yaml', 'data.yaml']

        for yaml_file in yaml_files:
            input_yaml = self.input_dataset / yaml_file
            if input_yaml.exists():
                output_yaml = self.output_dataset / yaml_file
                try:
                    # Read, potentially modify, and write
                    with open(input_yaml, 'r') as f:
                        yaml_data = yaml.safe_load(f)

                    # Update paths if they exist
                    if yaml_data and 'path' in yaml_data:
                        yaml_data['path'] = str(self.output_dataset)

                    # Add note about pruning
                    if yaml_data and isinstance(yaml_data, dict):
                        yaml_data['_pruning_info'] = {
                            'original_dataset': str(self.input_dataset),
                            'prune_percentage': self.prune_percentage,
                            'keep_percentage': self.keep_percentage,
                            'pruned_split': self.split if self.split else 'all'
                        }

                    if not self.dry_run:
                        with open(output_yaml, 'w') as f:
                            yaml.dump(yaml_data, f, default_flow_style=False)
                        print(f"\nCopied and updated: {yaml_file}")
                except Exception as e:
                    print(f"Error processing {yaml_file}: {e}")

    def prune(self):
        """Main pruning process."""
        print("\n" + "=" * 60)
        print("YOLO Dataset Pruning")
        print("=" * 60)
        print(f"Input dataset: {self.input_dataset}")
        print(f"Output dataset: {self.output_dataset}")
        print(f"Prune percentage: {self.prune_percentage}% (keep {self.keep_percentage}%)")
        if self.dry_run:
            print("MODE: DRY RUN (no files will be copied)")

        # Check if split structure exists
        common_splits = ['train', 'val', 'test', 'valid', 'validation']
        subdirs = [d for d in self.input_images_dir.iterdir()
                   if d.is_dir() and d.name in common_splits]

        is_split_structure = len(subdirs) > 0

        if is_split_structure:
            print(f"Dataset structure: SPLIT (found: {[d.name for d in subdirs]})")

            if self.split:
                # Prune only specified split
                if self.split not in [d.name for d in subdirs]:
                    raise ValueError(
                        f"Split '{self.split}' not found in dataset. Available: {[d.name for d in subdirs]}")
                self._prune_split(self.split)

                # Copy other splits completely
                if not self.dry_run:
                    print(f"\nCopying other splits without pruning...")
                    for subdir in subdirs:
                        if subdir.name != self.split:
                            # Copy images
                            src_images = self.input_images_dir / subdir.name
                            dst_images = self.output_images_dir / subdir.name
                            if src_images.exists():
                                # Count files for progress bar
                                image_files = list(src_images.glob('*.*'))
                                image_files = [f for f in image_files if
                                               f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']]

                                dst_images.mkdir(parents=True, exist_ok=True)
                                for img_file in tqdm(image_files, desc=f"Copying {subdir.name} images", unit="files"):
                                    shutil.copy2(img_file, dst_images / img_file.name)

                            # Copy labels
                            src_labels = self.input_labels_dir / subdir.name
                            dst_labels = self.output_labels_dir / subdir.name
                            if src_labels.exists():
                                label_files = list(src_labels.glob('*.txt'))

                                dst_labels.mkdir(parents=True, exist_ok=True)
                                for lbl_file in tqdm(label_files, desc=f"Copying {subdir.name} labels", unit="files"):
                                    shutil.copy2(lbl_file, dst_labels / lbl_file.name)
            else:
                # Prune all splits
                for subdir in subdirs:
                    self._prune_split(subdir.name)
        else:
            # Flat structure
            print("Dataset structure: FLAT")
            self._prune_split(None)

        # Copy YAML configuration files
        self._copy_yaml_files()

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("PRUNING SUMMARY")
        print("=" * 60)

        total_original = 0
        total_kept = 0
        total_removed = 0

        for split_name, stats in self.stats.items():
            print(f"\n{split_name}:")
            print(f"  Original: {stats['total']} images")
            print(f"  Kept:     {stats['kept']} images ({stats['kept'] / stats['total'] * 100:.1f}%)")
            print(f"  Removed:  {stats['removed']} images ({stats['removed'] / stats['total'] * 100:.1f}%)")

            total_original += stats['total']
            total_kept += stats['kept']
            total_removed += stats['removed']

        if len(self.stats) > 1:
            print(f"\nTOTAL ACROSS ALL SPLITS:")
            print(f"  Original: {total_original} images")
            print(f"  Kept:     {total_kept} images ({total_kept / total_original * 100:.1f}%)")
            print(f"  Removed:  {total_removed} images ({total_removed / total_original * 100:.1f}%)")

        if not self.dry_run:
            print(f"\nOutput dataset saved to: {self.output_dataset}")
        else:
            print("\nDRY RUN COMPLETED - No files were copied")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly prune a YOLO dataset by removing a specified percentage of images and labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove 20% of data from all splits
  python yolo_prune.py --input ./dataset --output ./dataset_pruned --prune 20

  # Remove 30% from training split only
  python yolo_prune.py --input ./dataset --output ./dataset_pruned --prune 30 --split train

  # Remove 50% from validation split with specific seed
  python yolo_prune.py --input ./dataset --output ./dataset_pruned --prune 50 --split val --seed 42

  # Dry run to see statistics without copying
  python yolo_prune.py --input ./dataset --output ./dataset_pruned --prune 25 --dry-run

Dataset Structures Supported:
  1. Flat structure:
     input_dataset/
       ├── images/
       │   ├── img1.jpg
       │   └── img2.jpg
       └── labels/
           ├── img1.txt
           └── img2.txt

  2. Split structure:
     input_dataset/
       ├── images/
       │   ├── train/
       │   ├── val/
       │   └── test/
       └── labels/
           ├── train/
           ├── val/
           └── test/

Note: The output dataset will maintain the same structure as the input.
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input YOLO dataset directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output pruned dataset directory'
    )

    parser.add_argument(
        '--prune',
        type=float,
        required=True,
        help='Percentage of data to remove (0-100). E.g., 20 means remove 20%%, keep 80%%'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'valid', 'validation'],
        help='Prune only a specific split. If not specified, prunes all splits/data.'
    )

    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )

    args = parser.parse_args()

    # Validate prune percentage
    if args.prune < 0 or args.prune > 100:
        print("Error: Prune percentage must be between 0 and 100")
        sys.exit(1)

    if args.prune == 0:
        print("Warning: Prune percentage is 0. This will copy the entire dataset.")

    if args.prune == 100:
        print("Warning: Prune percentage is 100. This will create an empty dataset.")

    # Check if output exists
    output_path = Path(args.output)
    if output_path.exists() and not args.dry_run:
        response = input(f"Output directory '{args.output}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
        # Remove existing output
        shutil.rmtree(output_path)

    try:
        pruner = YOLODatasetPruner(
            args.input,
            args.output,
            args.prune,
            args.split,
            args.seed,
            args.dry_run
        )
        pruner.prune()

        print("\n✓ Pruning completed successfully!")

    except KeyboardInterrupt:
        print("\n\nPruning interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()