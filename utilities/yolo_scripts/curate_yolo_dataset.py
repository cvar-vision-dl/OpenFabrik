#!/usr/bin/env python3
"""
YOLO Segmentation Dataset Review Script

This script allows you to visually review a YOLO segmentation dataset by displaying images
with their polygon mask annotations overlaid. You can navigate through images
and delete incorrectly labeled ones.

Supports both flat structure (images/, labels/) and split structure (images/train, images/val, etc.)

Controls:
- SPACE or RIGHT ARROW: Next image
- LEFT ARROW: Previous image
- 'd': Delete current image and its label file
- 'f': Toggle filled masks
- 'o': Toggle outlines
- 'q' or ESC: Quit

Usage:
    # Flat structure
    python yolo_review.py --dataset /path/to/dataset

    # Split structure - review all splits
    python yolo_review.py --dataset /path/to/dataset

    # Split structure - review specific split
    python yolo_review.py --dataset /path/to/dataset --split train
    python yolo_review.py --dataset /path/to/dataset --split val

    # Start from specific image
    python yolo_review.py --dataset /path/to/dataset --start 500

    # Custom paths
    python yolo_review.py --images /path/to/images --labels /path/to/labels
"""

import argparse
import os
import cv2
import numpy as np
import glob
import sys
from pathlib import Path
import random
import yaml


class YOLODatasetReviewer:
    def __init__(self, images_dir, labels_dir, class_names=None, start_index=0, split=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names or {}
        self.deleted_count = 0
        self.show_filled = True  # Toggle for filled masks
        self.show_outline = True  # Toggle for outlines
        self.split = split

        # Detect dataset structure and find all image files
        self.image_files, self.relative_paths = self._find_all_images()

        if not self.image_files:
            raise ValueError(f"No image files found in {self.images_dir}")

        # Validate and set start index AFTER we know how many images we have
        if start_index >= len(self.image_files):
            print(
                f"Warning: Start index {start_index} is beyond dataset size ({len(self.image_files)}). Starting from last image.")
            self.current_index = len(self.image_files) - 1
        elif start_index < 0:
            print(f"Warning: Start index {start_index} is negative. Starting from first image.")
            self.current_index = 0
        else:
            self.current_index = start_index

        print(f"Found {len(self.image_files)} images")
        if self.current_index > 0:
            print(f"Starting from image {self.current_index + 1}/{len(self.image_files)}")

        # Debug print to verify start index
        print(f"Current index set to: {self.current_index}")

        # Colors for different classes (BGR format for OpenCV)
        self.colors = [
            (0, 255, 0),  # Green
            (255, 0, 0),  # Blue
            (0, 0, 255),  # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 128, 0),  # Olive
        ]

    def _find_all_images(self):
        """
        Find all image files, supporting both flat and split directory structures.
        Returns a list of absolute paths and corresponding relative paths from images_dir.
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        relative_paths = []

        # Check if this is a split structure (has train/val/test subdirectories)
        common_splits = ['train', 'val', 'test', 'valid', 'validation']
        subdirs = [d for d in self.images_dir.iterdir() if d.is_dir() and d.name in common_splits]

        if subdirs and not self.split:
            # Split structure detected, review all splits
            print(f"Split structure detected. Found splits: {[d.name for d in subdirs]}")
            print("Reviewing all splits together. Use --split to review specific split.")

            for subdir in subdirs:
                for ext in image_extensions:
                    found_images = glob.glob(str(subdir / ext)) + glob.glob(str(subdir / ext.upper()))
                    for img_path in found_images:
                        image_files.append(img_path)
                        # Store relative path from images_dir (e.g., "train/image001.jpg")
                        rel_path = Path(img_path).relative_to(self.images_dir)
                        relative_paths.append(rel_path)

        elif subdirs and self.split:
            # Review specific split
            split_dir = self.images_dir / self.split
            if not split_dir.exists():
                raise ValueError(f"Split directory not found: {split_dir}")

            print(f"Reviewing split: {self.split}")

            for ext in image_extensions:
                found_images = glob.glob(str(split_dir / ext)) + glob.glob(str(split_dir / ext.upper()))
                for img_path in found_images:
                    image_files.append(img_path)
                    rel_path = Path(img_path).relative_to(self.images_dir)
                    relative_paths.append(rel_path)

        else:
            # Flat structure - search directly in images_dir
            print("Flat structure detected.")
            for ext in image_extensions:
                found_images = glob.glob(str(self.images_dir / ext)) + glob.glob(str(self.images_dir / ext.upper()))
                for img_path in found_images:
                    image_files.append(img_path)
                    # Relative path is just the filename
                    rel_path = Path(img_path).name
                    relative_paths.append(rel_path)

        # Sort both lists together to maintain correspondence
        paired = sorted(zip(image_files, relative_paths))
        if paired:
            image_files, relative_paths = zip(*paired)
            image_files = list(image_files)
            relative_paths = list(relative_paths)
        else:
            image_files = []
            relative_paths = []

        return image_files, relative_paths

    def load_yolo_labels(self, label_path):
        """Load YOLO format segmentation labels from file."""
        if not os.path.exists(label_path):
            return []

        labels = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 7:  # At least class_id + 3 coordinate pairs (6 values)
                            class_id = int(parts[0])
                            # Parse polygon coordinates (x1, y1, x2, y2, ..., xn, yn)
                            coordinates = [float(coord) for coord in parts[1:]]
                            # Ensure we have pairs of coordinates
                            if len(coordinates) % 2 == 0:
                                polygon_points = []
                                for i in range(0, len(coordinates), 2):
                                    polygon_points.append((coordinates[i], coordinates[i + 1]))
                                labels.append((class_id, polygon_points))
        except Exception as e:
            print(f"Error reading label file {label_path}: {e}")

        return labels

    def draw_masks(self, image, labels, show_filled=True, show_outline=True):
        """Draw polygon masks on the image."""
        img_height, img_width = image.shape[:2]

        # Create overlay for semi-transparent masks
        overlay = image.copy()

        for i, (class_id, polygon_points) in enumerate(labels):
            # Convert normalized coordinates to pixel coordinates
            pixel_points = []
            for x_norm, y_norm in polygon_points:
                x_pixel = int(x_norm * img_width)
                y_pixel = int(y_norm * img_height)
                pixel_points.append([x_pixel, y_pixel])

            if len(pixel_points) < 3:
                continue  # Need at least 3 points for a polygon

            # Convert to numpy array for OpenCV
            pts = np.array(pixel_points, dtype=np.int32)

            # Get color for this class
            color = self.colors[class_id % len(self.colors)]

            # Draw filled polygon (mask) on overlay
            if show_filled:
                cv2.fillPoly(overlay, [pts], color)

            # Draw polygon outline
            if show_outline:
                cv2.polylines(image, [pts], True, color, 2)

            # Add class label near the polygon
            if len(pixel_points) > 0:
                # Find centroid for label placement
                cx = int(sum(p[0] for p in pixel_points) / len(pixel_points))
                cy = int(sum(p[1] for p in pixel_points) / len(pixel_points))

                class_name = self.class_names.get(class_id, f"Class {class_id}")
                label_text = f"{class_name} ({class_id})"

                # Get text size for background rectangle
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    image,
                    (cx - text_width // 2, cy - text_height - 5),
                    (cx + text_width // 2, cy + 5),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    image,
                    label_text,
                    (cx - text_width // 2, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )

        # Blend overlay with original image for semi-transparent effect
        if show_filled:
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def get_label_path(self, image_index):
        """Get the corresponding label file path for an image using its relative path."""
        # Get the relative path of the image
        rel_path = self.relative_paths[image_index]

        # Change extension to .txt
        if isinstance(rel_path, str):
            rel_path = Path(rel_path)

        label_rel_path = rel_path.with_suffix('.txt')

        # Construct full label path
        label_path = self.labels_dir / label_rel_path

        return label_path

    def delete_current_image(self):
        """Delete the current image and its label file."""
        if self.current_index >= len(self.image_files):
            return False

        image_path = self.image_files[self.current_index]
        label_path = self.get_label_path(self.current_index)

        try:
            # Delete image file
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Deleted image: {image_path}")

            # Delete label file
            if os.path.exists(label_path):
                os.remove(label_path)
                print(f"Deleted label: {label_path}")

            # Remove from lists
            self.image_files.pop(self.current_index)
            self.relative_paths.pop(self.current_index)
            self.deleted_count += 1

            # Adjust current index
            if self.current_index >= len(self.image_files):
                self.current_index = max(0, len(self.image_files) - 1)

            return True

        except Exception as e:
            print(f"Error deleting files: {e}")
            return False

    def run(self):
        """Main review loop."""
        if not self.image_files:
            print("No images to review!")
            return

        print("\nControls:")
        print("  SPACE or RIGHT ARROW: Next image")
        print("  LEFT ARROW: Previous image")
        print("  'd': Delete current image and label")
        print("  'q' or ESC: Quit")
        print("  'f': Toggle filled masks")
        print("  'o': Toggle outlines")
        print("  'h': Show help")
        print()

        cv2.namedWindow('YOLO Dataset Review', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('YOLO Dataset Review', 1200, 800)

        while self.current_index < len(self.image_files):
            # Load current image
            image_path = self.image_files[self.current_index]
            image = cv2.imread(image_path)

            if image is None:
                print(f"Could not load image: {image_path}")
                self.current_index += 1
                continue

            # Load labels
            label_path = self.get_label_path(self.current_index)
            labels = self.load_yolo_labels(label_path)

            # Create a copy for drawing
            display_image = image.copy()

            # Draw polygon masks
            if labels:
                self.draw_masks(display_image, labels, self.show_filled, self.show_outline)

            # Add info text
            mask_info = []
            if self.show_filled:
                mask_info.append("Filled")
            if self.show_outline:
                mask_info.append("Outline")
            mask_mode = " + ".join(mask_info) if mask_info else "Hidden"

            info_text = f"Image {self.current_index + 1}/{len(self.image_files)} | " \
                        f"Deleted: {self.deleted_count} | " \
                        f"Masks: {len(labels)} | " \
                        f"Mode: {mask_mode}"

            cv2.putText(
                display_image,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Add filename (including split if applicable)
            rel_path = self.relative_paths[self.current_index]
            filename_display = str(rel_path)
            cv2.putText(
                display_image,
                filename_display,
                (10, display_image.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            # Display image
            cv2.imshow('YOLO Dataset Review', display_image)

            # Handle keyboard input
            raw_key = cv2.waitKey(0)
            key = raw_key & 0xFF

            # Arrow key detection (codes can vary by system)
            # Common codes: Left=81/2424832, Right=83/2555904, Up=82/2490368, Down=84/2621440
            if key == ord(' ') or key == 83 or raw_key == 2555904:  # Space or Right arrow - next image
                self.current_index += 1
            elif key == 81 or raw_key == 2424832:  # Left arrow - previous image
                self.current_index = max(0, self.current_index - 1)
            elif key == ord('d'):  # Delete current image
                if self.delete_current_image():
                    print(f"Deleted! Remaining images: {len(self.image_files)}")
                    if len(self.image_files) == 0:
                        print("No more images to review!")
                        break
                else:
                    self.current_index += 1
            elif key == ord('f'):  # Toggle filled masks
                self.show_filled = not self.show_filled
                print(f"Filled masks: {'ON' if self.show_filled else 'OFF'}")
            elif key == ord('o'):  # Toggle outlines
                self.show_outline = not self.show_outline
                print(f"Outlines: {'ON' if self.show_outline else 'OFF'}")
            elif key == ord('q') or key == 27:  # Quit
                break
            elif key == ord('h'):  # Help
                print("\nControls:")
                print("  SPACE or RIGHT ARROW: Next image")
                print("  LEFT ARROW: Previous image")
                print("  'd': Delete current image and label")
                print("  'f': Toggle filled masks")
                print("  'o': Toggle outlines")
                print("  'q' or ESC: Quit")
                print("  'h': Show help")

        cv2.destroyAllWindows()
        print(f"\nReview completed! Deleted {self.deleted_count} images.")


def load_class_names(classes_file):
    """Load class names from a file."""
    class_names = {}
    if classes_file and os.path.exists(classes_file):
        try:
            with open(classes_file, 'r') as f:
                for i, line in enumerate(f):
                    class_names[i] = line.strip()
        except Exception as e:
            print(f"Error reading classes file: {e}")
    return class_names


def load_class_names_from_yaml(yaml_file):
    """Load class names from dataset.yaml file."""
    class_names = {}
    if not os.path.exists(yaml_file):
        return class_names

    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)

        if data and 'names' in data:
            names_data = data['names']

            # Handle different formats:
            # 1. List format: names: ['class1', 'class2', 'class3']
            if isinstance(names_data, list):
                for i, name in enumerate(names_data):
                    class_names[i] = name

            # 2. Dict format: names: {0: 'class1', 1: 'class2', 2: 'class3'}
            elif isinstance(names_data, dict):
                for idx, name in names_data.items():
                    class_names[int(idx)] = name

            print(f"Loaded {len(class_names)} class names from {yaml_file}")

    except Exception as e:
        print(f"Error reading YAML file {yaml_file}: {e}")

    return class_names


def main():
    parser = argparse.ArgumentParser(
        description="Review and clean a YOLO segmentation dataset by visually inspecting images with polygon mask annotations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review segmentation dataset with standard YOLO structure (flat or split)
  python yolo_review.py --dataset /path/to/yolo_dataset

  # Review only the training split
  python yolo_review.py --dataset /path/to/dataset --split train

  # Review only the validation split
  python yolo_review.py --dataset /path/to/dataset --split val

  # Start from image 500 (useful for resuming review)
  python yolo_review.py --dataset /path/to/dataset --start 500

  # Specify custom image and label directories
  python yolo_review.py --images /path/to/images --labels /path/to/labels

  # Include class names file
  python yolo_review.py --dataset /path/to/dataset --classes classes.txt --start 100

  # Note: If dataset.yaml or data.yaml exists in the dataset directory,
  # class names will be automatically loaded from it.

Dataset Structures Supported:
  1. Flat structure:
     dataset/
       ├── images/
       │   ├── img1.jpg
       │   └── img2.jpg
       └── labels/
           ├── img1.txt
           └── img2.txt

  2. Split structure:
     dataset/
       ├── images/
       │   ├── train/
       │   │   ├── img1.jpg
       │   │   └── img2.jpg
       │   └── val/
       │       ├── img3.jpg
       │       └── img4.jpg
       └── labels/
           ├── train/
           │   ├── img1.txt
           │   └── img2.txt
           └── val/
               ├── img3.txt
               └── img4.txt
        """
    )

    # Dataset path (assumes images/ and labels/ subdirectories)
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to YOLO dataset directory (should contain images/ and labels/ subdirs)'
    )

    # Individual directories
    parser.add_argument(
        '--images',
        type=str,
        help='Path to images directory'
    )

    parser.add_argument(
        '--labels',
        type=str,
        help='Path to labels directory'
    )

    # Optional class names file
    parser.add_argument(
        '--classes',
        type=str,
        help='Path to classes.txt file with class names (optional, overrides dataset.yaml if specified)'
    )

    # Starting image index
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Starting image index (0-based). Useful for resuming review from a specific point.'
    )

    # Split selection
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'val', 'test', 'valid', 'validation'],
        help='Review only a specific split (train/val/test). If not specified, reviews all images.'
    )

    args = parser.parse_args()

    # Determine image and label directories
    if args.dataset:
        dataset_path = Path(args.dataset)
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'

        # Check if directories exist
        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            # Try common alternatives
            alt_images = dataset_path / 'imgs'
            if alt_images.exists():
                images_dir = alt_images
            else:
                print("Please specify --images and --labels directories")
                sys.exit(1)

        if not labels_dir.exists():
            print(f"Labels directory not found: {labels_dir}")
            sys.exit(1)

    elif args.images and args.labels:
        images_dir = Path(args.images)
        labels_dir = Path(args.labels)

        if not images_dir.exists():
            print(f"Images directory not found: {images_dir}")
            sys.exit(1)

        if not labels_dir.exists():
            print(f"Labels directory not found: {labels_dir}")
            sys.exit(1)
    else:
        print("Please specify either --dataset or both --images and --labels")
        parser.print_help()
        sys.exit(1)

    # Load class names if provided
    class_names = {}

    # First, try to load from dataset.yaml if it exists
    if args.dataset:
        yaml_path = Path(args.dataset) / 'dataset.yaml'
        if yaml_path.exists():
            print(f"Found dataset.yaml at {yaml_path}")
            class_names = load_class_names_from_yaml(yaml_path)
        else:
            # Also check for data.yaml (alternative common name)
            yaml_path = Path(args.dataset) / 'data.yaml'
            if yaml_path.exists():
                print(f"Found data.yaml at {yaml_path}")
                class_names = load_class_names_from_yaml(yaml_path)

    # If --classes is specified, it takes precedence and overrides yaml
    if args.classes:
        classes_from_file = load_class_names(args.classes)
        if classes_from_file:
            class_names = classes_from_file
            print(f"Loaded {len(class_names)} class names from {args.classes} (overriding yaml)")

    try:
        # Create and run reviewer
        reviewer = YOLODatasetReviewer(images_dir, labels_dir, class_names, args.start, args.split)
        reviewer.run()

    except KeyboardInterrupt:
        print("\nReview interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()