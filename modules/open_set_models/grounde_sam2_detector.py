import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict


class GroundedSAM2Detector:
    """
    A class for object detection and segmentation using Grounding DINO + SAM2.

    This class combines Grounding DINO for object detection with SAM2 for segmentation,
    providing methods to process images and save results in various formats including YOLO.

    ENHANCED VERSION:
    - Maintains global class consistency across all images
    - Smart fuzzy matching for compound classes (e.g., "equipment radio" → "equipment")
    """

    def __init__(self,
                 sam2_checkpoint: str = "./checkpoints/sam2.1_hiera_large.pt",
                 sam2_model_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
                 grounding_dino_config: str = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                 grounding_dino_checkpoint: str = "gdino_checkpoints/groundingdino_swint_ogc.pth",
                 device: Optional[str] = None,
                 global_classes: Optional[List[str]] = None):
        """
        Initialize the GroundedSAM2Detector.

        Args:
            sam2_checkpoint: Path to SAM2 checkpoint
            sam2_model_config: Path to SAM2 model config
            grounding_dino_config: Path to Grounding DINO config
            grounding_dino_checkpoint: Path to Grounding DINO checkpoint
            device: Device to use ('cuda' or 'cpu'). Auto-detect if None.
            global_classes: List of all possible classes for consistent mapping across images
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.grounding_dino_config = grounding_dino_config
        self.grounding_dino_checkpoint = grounding_dino_checkpoint

        # Model instances
        self.sam2_model = None
        self.sam2_predictor = None
        self.grounding_model = None

        # Last processed results
        self.last_image_path = None
        self.last_image_shape = None
        self.last_detections = None
        self.last_class_names = None
        self.last_confidences = None
        self.last_masks = None
        self.last_boxes = None

        # FIXED: Global class mapping for consistency across all images
        self.global_classes = global_classes or []
        self.class_to_id = {}
        self.id_to_class = {}
        self.classes_file_saved = False  # Track if we've already saved classes.txt

        # Initialize global class mapping if provided
        if self.global_classes:
            self._initialize_global_class_mapping()

        print(f"Initialized GroundedSAM2Detector on device: {self.device}")
        if self.global_classes:
            print(f"Using global class list with {len(self.global_classes)} classes")

    def _initialize_global_class_mapping(self) -> None:
        """Initialize global class mapping from the provided class list."""
        # Remove duplicates while preserving order
        unique_classes = []
        seen = set()
        for cls in self.global_classes:
            if cls not in seen:
                unique_classes.append(cls)
                seen.add(cls)

        # Sort for consistency and create sequential mapping
        sorted_classes = sorted(unique_classes)

        for idx, class_name in enumerate(sorted_classes):
            self.class_to_id[class_name] = idx  # Sequential: 0, 1, 2, ...
            self.id_to_class[idx] = class_name

        # Integrity check
        expected_ids = set(range(len(sorted_classes)))
        actual_ids = set(self.id_to_class.keys())

        if expected_ids != actual_ids:
            raise ValueError("Class ID mapping integrity check failed!")

        self.class_to_id = {name: idx for idx, name in enumerate(sorted_classes)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}

        print(f"Initialized global class mapping:")
        print(f"  Classes: {sorted_classes}")
        print(f"  Total: {len(sorted_classes)} classes")

    def set_global_classes(self, classes: List[str]) -> None:
        """
        Set or update the global class list.

        Args:
            classes: List of all possible classes
        """
        self.global_classes = classes
        self._initialize_global_class_mapping()
        self.classes_file_saved = False  # Reset flag when classes change

    def load_global_classes_from_file(self, classes_file: str) -> None:
        """
        Load global classes from a JSON file (e.g., classes.json from prompt generation).

        Args:
            classes_file: Path to JSON file containing the classes list
        """
        try:
            with open(classes_file, 'r') as f:
                classes = json.load(f)

            if isinstance(classes, list):
                self.set_global_classes(classes)
                print(f"Loaded {len(classes)} global classes from {classes_file}")
            else:
                print(f"ERROR: Classes file should contain a list, got {type(classes)}")
        except Exception as e:
            print(f"ERROR: Failed to load classes from {classes_file}: {e}")

    def _match_detected_class_to_global(self, detected_class: str) -> Optional[str]:
        """
        Match a detected class to global classes with fuzzy matching.

        This handles compound classes like:
        - "equipment radio" → "equipment" (if "equipment" is in global classes)
        - "wire wire" → "wire" (if "wire" is in global classes)
        - "deck floor" → "deck" (if "deck" is in global classes, takes first match)

        Args:
            detected_class: Class detected by Grounding DINO

        Returns:
            Matched global class name or None if no match found
        """
        # Direct exact match first
        if detected_class in self.class_to_id:
            return detected_class

        # Try fuzzy matching by splitting compound classes
        detected_words = detected_class.split()

        # Look for individual words that match global classes
        matched_classes = []
        for word in detected_words:
            word_clean = word.strip().lower()
            # Check if this word matches any global class (case-insensitive)
            for global_class in self.class_to_id.keys():
                if word_clean == global_class.lower():
                    matched_classes.append(global_class)
                    break

        if matched_classes:
            # If multiple matches found, prefer the first one
            best_match = matched_classes[0]
            print(f"  Fuzzy match: '{detected_class}' → '{best_match}' (from words: {detected_words})")
            return best_match

        # No match found
        return None

    def load_models(self) -> None:
        """Load SAM2 and Grounding DINO models."""
        print("Loading models...")

        # Configure CUDA optimizations
        self._configure_cuda_optimizations()

        # Load SAM2
        print("Loading SAM2 model...")
        self.sam2_model = build_sam2(
            self.sam2_model_config,
            self.sam2_checkpoint,
            device=self.device
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

        # Load Grounding DINO
        print("Loading Grounding DINO model...")
        self.grounding_model = load_model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device
        )

        print("Models loaded successfully!")

    def _configure_cuda_optimizations(self) -> None:
        """Configure CUDA optimizations for better performance."""
        if self.device == "cuda":
            # Don't use bfloat16 autocast due to Grounding DINO compatibility issues
            # torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

            # Enable TF32 for Ampere GPUs (this is still safe to use)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 optimizations for Ampere GPU")

            print("CUDA optimizations configured (without bfloat16 autocast)")

    def _create_class_mapping(self, class_names: List[str]) -> None:
        """
        FIXED: Use global class mapping or create one if not available.
        """
        if not self.global_classes:
            # Fallback to old behavior if no global classes provided
            unique_classes = list(set(class_names))
            unique_classes.sort()  # Ensure consistent ordering

            self.class_to_id = {name: idx for idx, name in enumerate(unique_classes)}
            self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}

            print(f"WARNING: No global classes set, created mapping for {len(unique_classes)} classes")

        # If global classes are set, mapping is already initialized
        # Just verify that detected classes exist in the mapping
        unknown_classes = [cls for cls in class_names if cls not in self.class_to_id]
        if unknown_classes:
            print(f"WARNING: Found unknown classes not in global mapping: {unknown_classes}")
            print(f"These will be skipped during annotation!")

    def process_image(self,
                      image_path: str,
                      text_prompt: str,
                      box_threshold: float = 0.20,
                      text_threshold: float = 0.20) -> Dict:
        """
        Process an image with Grounding DINO + SAM2.

        Args:
            image_path: Path to the image
            text_prompt: Text description of objects to detect
            box_threshold: Box confidence threshold for Grounding DINO
            text_threshold: Text confidence threshold for Grounding DINO

        Returns:
            Dictionary with detection results
        """
        if not self.sam2_model or not self.grounding_model:
            raise RuntimeError("Models must be loaded first. Call load_models().")

        print(f"Processing image: {image_path}")

        # Load and prepare image
        image_source, image = load_image(image_path)
        self.sam2_predictor.set_image(image_source)

        # Store image info
        self.last_image_path = image_path
        self.last_image_shape = image_source.shape
        h, w, _ = image_source.shape

        # Grounding DINO detection
        print("Running Grounding DINO detection...")
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=image,
            caption=text_prompt.lower(),  # Ensure lowercase
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(boxes) == 0:
            print("No objects detected!")
            return self._create_empty_results()

        # ENHANCED: Smart fuzzy matching for compound classes
        if self.global_classes:
            valid_indices = []
            valid_labels = []
            valid_boxes = []
            valid_confidences = []

            print(f"Filtering detected classes against global classes with smart matching...")
            for i, label in enumerate(labels):
                matched_class = self._match_detected_class_to_global(label)

                if matched_class:
                    valid_indices.append(i)
                    valid_labels.append(matched_class)  # Use the matched global class name
                    valid_boxes.append(boxes[i])
                    valid_confidences.append(confidences[i])

                    if matched_class != label:
                        print(f"  ✓ Mapped: '{label}' → '{matched_class}'")
                    else:
                        print(f"  ✓ Exact match: '{label}'")
                else:
                    print(f"  ✗ Skipping unknown class: '{label}' (no fuzzy match found)")

            if not valid_labels:
                print("No valid classes detected after filtering!")
                return self._create_empty_results()

            # Update with filtered results
            labels = valid_labels
            boxes = torch.stack(valid_boxes) if valid_boxes else torch.tensor([])
            confidences = torch.stack(valid_confidences) if valid_confidences else torch.tensor([])

            print(f"Final filtered classes: {labels}")
        else:
            print("No global classes set, using all detected classes")

        # Convert boxes to the right format for SAM2
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # SAM2 segmentation
        print(f"Running SAM2 segmentation on {len(input_boxes)} detections...")
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # Process masks
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # Store results
        self.last_boxes = input_boxes
        self.last_masks = masks
        self.last_class_names = labels
        self.last_confidences = confidences.numpy().tolist()

        # Create/verify class mapping
        self._create_class_mapping(labels)

        # Create supervision detections with GLOBAL class IDs
        class_ids = np.array([self.class_to_id[name] for name in labels])
        self.last_detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=class_ids
        )

        print(f"Successfully processed {len(input_boxes)} detections")
        print(f"Class IDs used: {class_ids.tolist()}")

        return {
            'boxes': input_boxes,
            'masks': masks,
            'class_names': labels,
            'confidences': self.last_confidences,
            'class_ids': class_ids,
            'image_shape': (h, w),
            'detections': self.last_detections
        }

    def _create_empty_results(self) -> Dict:
        """Create empty results structure when no objects are detected."""
        h, w, _ = self.last_image_shape
        return {
            'boxes': np.array([]),
            'masks': np.array([]),
            'class_names': [],
            'confidences': [],
            'class_ids': np.array([]),
            'image_shape': (h, w),
            'detections': sv.Detections.empty()
        }

    def save_yolo_format(self,
                         output_dir: str,
                         dataset_name: str = "dataset",
                         split: str = "train") -> Dict[str, str]:
        """
        Save the last processed results in YOLO format.

        Args:
            output_dir: Base directory for the dataset
            dataset_name: Name of the dataset
            split: Data split ('train', 'val', 'test')

        Returns:
            Dictionary with paths to saved files
        """
        if self.last_detections is None:
            raise RuntimeError("No processed results to save. Call process_image() first.")

        # Create directory structure
        dataset_dir = Path(output_dir) / dataset_name
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Get image name without extension
        image_name = Path(self.last_image_path).stem
        image_ext = Path(self.last_image_path).suffix

        # Copy image to dataset
        import shutil
        target_image_path = images_dir / f"{image_name}{image_ext}"
        shutil.copy2(self.last_image_path, target_image_path)

        # Save YOLO annotation
        h, w = self.last_image_shape[:2]
        annotation_path = labels_dir / f"{image_name}.txt"

        with open(annotation_path, 'w') as f:
            for i, (box, mask, class_name) in enumerate(zip(
                    self.last_boxes, self.last_masks, self.last_class_names
            )):
                # FIXED: Use consistent global class ID
                class_id = self.class_to_id[class_name]

                # Convert bounding box to YOLO format (normalized)
                x1, y1, x2, y2 = box
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h

                # For segmentation, convert mask to polygon
                polygon = self._mask_to_polygon(mask, w, h)

                if polygon is not None:
                    # YOLO segmentation format: class_id + normalized polygon points
                    polygon_str = ' '.join([f"{coord:.6f}" for coord in polygon])
                    f.write(f"{class_id} {polygon_str}\n")
                else:
                    # Fallback to bounding box format
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # FIXED: Only save classes.txt once and include ALL global classes
        classes_path = dataset_dir / "classes.txt"
        if not self.classes_file_saved:
            with open(classes_path, 'w') as f:
                for class_id in sorted(self.id_to_class.keys()):
                    f.write(f"{self.id_to_class[class_id]}\n")
            self.classes_file_saved = True
            print(f"Saved classes.txt with {len(self.id_to_class)} classes")

        # Save dataset.yaml for YOLO
        yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.exists():  # Only create once
            self._save_yolo_yaml(yaml_path, dataset_name, len(self.class_to_id))

        print(f"YOLO annotation saved: {annotation_path}")
        print(f"Used class IDs: {[self.class_to_id[name] for name in self.last_class_names]}")

        return {
            'dataset_dir': str(dataset_dir),
            'image_path': str(target_image_path),
            'annotation_path': str(annotation_path),
            'classes_path': str(classes_path),
            'yaml_path': str(yaml_path)
        }

    def _mask_to_polygon(self, mask: np.ndarray, img_w: int, img_h: int) -> Optional[List[float]]:
        """
        Convert a binary mask to a normalized polygon.

        Args:
            mask: Binary mask array
            img_w: Image width
            img_h: Image height

        Returns:
            List of normalized polygon coordinates or None if conversion fails
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return None

            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Simplify contour to reduce points
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

            # Convert to normalized coordinates
            polygon = []
            for point in simplified_contour:
                x, y = point[0]
                # Normalize coordinates
                x_norm = x / img_w
                y_norm = y / img_h
                polygon.extend([x_norm, y_norm])

            # YOLO requires at least 6 coordinates (3 points)
            if len(polygon) >= 6:
                return polygon
            else:
                return None

        except Exception as e:
            print(f"Error converting mask to polygon: {e}")
            return None

    def _save_yolo_yaml(self, yaml_path: Path, dataset_name: str, num_classes: int) -> None:
        """Save YOLO dataset configuration file."""
        yaml_content = f"""# YOLO dataset configuration
# Generated by GroundedSAM2Detector (Enhanced Version with Smart Matching)

# Dataset info
name: {dataset_name}
nc: {num_classes}  # number of classes

# Paths (relative to this file)
train: images/train
val: images/val
test: images/test

# Class names
names:
"""

        # Verify sequential IDs before saving
        expected_ids = set(range(num_classes))
        actual_ids = set(self.id_to_class.keys())

        if expected_ids != actual_ids:
            raise ValueError("Cannot create YOLO config with non-sequential class IDs!")

        # Generate with guaranteed sequential order
        for class_id in range(num_classes):  # 0, 1, 2, ...
            class_name = self.id_to_class[class_id]
            yaml_content += f"  {class_id}: {class_name}\n"

        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

    def save_visualizations(self, output_dir: str) -> Dict[str, str]:
        """
        Save annotated visualizations of the last processed image.

        Args:
            output_dir: Directory to save visualizations

        Returns:
            Dictionary with paths to saved visualizations
        """
        if self.last_detections is None:
            raise RuntimeError("No processed results to visualize. Call process_image() first.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Load original image
        img = cv2.imread(self.last_image_path)
        image_name = Path(self.last_image_path).stem

        # Create labels for visualization
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(self.last_class_names, self.last_confidences)
        ]

        # Annotate with boxes
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=self.last_detections)

        # Add labels
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=self.last_detections,
            labels=labels
        )
        boxes_path = os.path.join(output_dir, f"{image_name}_boxes.jpg")
        cv2.imwrite(boxes_path, annotated_frame)

        # Annotate with masks
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=self.last_detections)
        masks_path = os.path.join(output_dir, f"{image_name}_masks.jpg")
        cv2.imwrite(masks_path, annotated_frame)

        print(f"Visualizations saved to: {output_dir}")

        return {
            'boxes_image': boxes_path,
            'masks_image': masks_path
        }

    def save_json_results(self, output_path: str) -> str:
        """
        Save detection results in JSON format.

        Args:
            output_path: Path to save JSON file

        Returns:
            Path to saved JSON file
        """
        if self.last_detections is None:
            raise RuntimeError("No processed results to save. Call process_image() first.")

        h, w = self.last_image_shape[:2]

        # Convert masks to RLE format
        def single_mask_to_rle(mask):
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        mask_rles = [single_mask_to_rle(mask) for mask in self.last_masks]

        # Create results dictionary
        results = {
            "image_path": self.last_image_path,
            "image_width": w,
            "image_height": h,
            "box_format": "xyxy",
            "class_mapping": self.class_to_id,
            "global_classes": self.global_classes,
            "smart_matching_enabled": len(self.global_classes) > 0,
            "annotations": [
                {
                    "class_name": class_name,
                    "class_id": self.class_to_id[class_name],
                    "bbox": box.tolist(),
                    "segmentation": mask_rle,
                    "confidence": confidence,
                }
                for class_name, box, mask_rle, confidence in zip(
                    self.last_class_names, self.last_boxes, mask_rles, self.last_confidences
                )
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"JSON results saved to: {output_path}")
        return output_path

    def process_and_save_all(self,
                             image_path: str,
                             text_prompt: str,
                             output_dir: str,
                             dataset_name: str = "grounded_sam2_dataset",
                             split: str = "train",
                             box_threshold: float = 0.20,
                             text_threshold: float = 0.20) -> Dict[str, Union[str, Dict]]:
        """
        Complete workflow: process image and save in all formats.

        Args:
            image_path: Path to input image
            text_prompt: Text description of objects to detect
            output_dir: Base output directory
            dataset_name: Name for the dataset
            split: Data split for YOLO format
            box_threshold: Box confidence threshold
            text_threshold: Text confidence threshold

        Returns:
            Dictionary with all saved file paths and processing results
        """
        # Process the image
        results = self.process_image(
            image_path=image_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        if len(results['boxes']) == 0:
            print("No objects detected, skipping save operations.")
            return {'results': results, 'saved_files': {}}

        # Save in all formats
        saved_files = {}

        # YOLO format
        yolo_paths = self.save_yolo_format(output_dir, dataset_name, split)
        saved_files['yolo'] = yolo_paths

        # Visualizations
        viz_dir = os.path.join(output_dir, "visualizations")
        viz_paths = self.save_visualizations(viz_dir)
        saved_files['visualizations'] = viz_paths

        # JSON results
        json_path = os.path.join(output_dir, f"{Path(image_path).stem}_results.json")
        json_file = self.save_json_results(json_path)
        saved_files['json'] = json_file

        return {
            'results': results,
            'saved_files': saved_files
        }

    def get_summary(self) -> Dict:
        """Get summary of the detector state and last results."""
        return {
            'device': self.device,
            'models_loaded': self.sam2_model is not None and self.grounding_model is not None,
            'last_image_processed': self.last_image_path,
            'last_detections_count': len(self.last_boxes) if self.last_boxes is not None else 0,
            'available_classes': list(self.class_to_id.keys()) if self.class_to_id else [],
            'class_count': len(self.class_to_id),
            'global_classes_set': len(self.global_classes) > 0,
            'classes_file_saved': self.classes_file_saved,
            'smart_matching_enabled': len(self.global_classes) > 0
        }


# Example usage with smart fuzzy matching
if __name__ == '__main__':
    # Example global classes (from your prompt generation)
    global_classes = ["equipment", "radio", "wire", "computer", "tank", "container", "deck", "floor", "valve", "pipe"]

    # ENHANCED: Initialize detector with global classes for smart matching
    detector = GroundedSAM2Detector(
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
        global_classes=global_classes  # Enable smart matching
    )

    # Load models
    detector.load_models()

    # Define your text prompt and image
    text_prompt = ".".join(global_classes)
    image_path = "/tmp/test_image.png"

    # Process and save everything - now with smart matching!
    all_results = detector.process_and_save_all(
        image_path=image_path,
        text_prompt=text_prompt,
        output_dir="outputs/example_dataset",
        dataset_name="object_segmentation",
        split="train"
    )

    # Print summary
    summary = detector.get_summary()
    print(f"\nProcessing complete!")
    print(f"Detected {summary['last_detections_count']} objects")
    print(f"Classes found: {summary['available_classes']}")
    print(f"Smart matching enabled: {summary['smart_matching_enabled']}")