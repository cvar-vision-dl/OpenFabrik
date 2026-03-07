import os
import gc
import cv2
import json
import torch
import numpy as np
import supervision as sv
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional, Union


class SAM3Detector:
    """
    Object detection and segmentation using SAM3 (Segment Anything Model 3).

    Uses SAM3's Promptable Concept Segmentation (PCS) with a sequential
    per-class strategy: the image encoder runs once per image, and the
    detection head runs once per class. This keeps VRAM usage well below
    the 24 GB budget and is the community-proven approach for multi-class
    annotation.

    Provides the same interface as GroundedSAM2Detector so it can be used
    as a drop-in annotator in the scene generation pipeline.
    """

    def __init__(self,
                 sam3_checkpoint: str,
                 device: Optional[str] = None,
                 global_classes: Optional[List[str]] = None):
        """
        Args:
            sam3_checkpoint: Path to the SAM3 checkpoint file (sam3.pt).
            device: 'cuda' or 'cpu'. Auto-detected if None.
            global_classes: Ordered class list for consistent YOLO class IDs.
        """
        self.sam3_checkpoint = sam3_checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.is_loaded = False

        # Last processed results
        self.last_image_path = None
        self.last_image_shape = None
        self.last_detections = None
        self.last_class_names = None
        self.last_confidences = None
        self.last_masks = None
        self.last_boxes = None

        # Global class mapping for consistent IDs across images
        self.global_classes = global_classes or []
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: Dict[int, str] = {}
        self.classes_file_saved = False

        if self.global_classes:
            self._initialize_global_class_mapping()

        print(f"Initialized SAM3Detector on device: {self.device}")
        if self.global_classes:
            print(f"Using global class list with {len(self.global_classes)} classes")

    # -------------------------------------------------------------------------
    # Class mapping
    # -------------------------------------------------------------------------

    def _initialize_global_class_mapping(self) -> None:
        unique = sorted(set(self.global_classes))
        self.class_to_id = {name: idx for idx, name in enumerate(unique)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
        print(f"Initialized global class mapping: {list(self.class_to_id.keys())}")

    def set_global_classes(self, classes: List[str]) -> None:
        self.global_classes = classes
        self._initialize_global_class_mapping()
        self.classes_file_saved = False

    def load_global_classes_from_file(self, classes_file: str) -> None:
        try:
            with open(classes_file, 'r') as f:
                classes = json.load(f)
            if isinstance(classes, list):
                self.set_global_classes(classes)
                print(f"Loaded {len(classes)} global classes from {classes_file}")
        except Exception as e:
            print(f"ERROR: Failed to load classes from {classes_file}: {e}")

    # -------------------------------------------------------------------------
    # Model lifecycle
    # -------------------------------------------------------------------------

    def load_model(self) -> None:
        """Load the SAM3 model. Enables TF32 on Ampere GPUs for efficiency."""
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError:
            raise ImportError(
                "SAM3 is not installed. Clone and install it:\n"
                "  git clone https://github.com/facebookresearch/sam3\n"
                "  cd sam3 && pip install -e . && cd .."
            )

        if self.device == "cuda" and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("Enabled TF32 optimizations for Ampere GPU")

        print(f"Loading SAM3 from: {self.sam3_checkpoint}")
        model = build_sam3_image_model(checkpoint_path=self.sam3_checkpoint)
        model = model.to(self.device)
        self.processor = Sam3Processor(model)
        self.is_loaded = True
        print("SAM3 loaded successfully.")

    def unload_model(self) -> None:
        """Unload the model and free GPU memory."""
        self.processor = None
        self.is_loaded = False
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    def process_image(self,
                      image_path: str,
                      text_prompt: str,
                      box_threshold: float = 0.45,
                      text_threshold: float = 0.45) -> Dict:
        """
        Annotate a single image using SAM3's per-class sequential prompting.

        The image encoder runs once; the detection head iterates over each class.
        SAM3 returns per-instance confidence scores; detections below
        `box_threshold` are discarded. `text_threshold` is accepted for
        interface compatibility but is not used separately.

        Args:
            image_path: Path to input image.
            text_prompt: Dot-separated class string, e.g. "cup.bottle.plate"
                         (same convention as the scene generation pipeline).
            box_threshold: Minimum confidence score to keep a detection (default 0.45).
            text_threshold: Kept for interface compatibility with GroundedSAM2Detector.

        Returns:
            Dictionary with 'boxes', 'masks', 'class_names', 'confidences',
            'class_ids', 'image_shape', 'detections' — same schema as
            GroundedSAM2Detector.
        """
        if not self.is_loaded or self.processor is None:
            raise RuntimeError("Model must be loaded first. Call load_model().")

        print(f"Processing image: {image_path}")

        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        self.last_image_path = image_path
        self.last_image_shape = (h, w, 3)

        # Derive classes to query
        classes = [c.strip() for c in text_prompt.split('.') if c.strip()]
        if self.global_classes:
            # Preserve global ordering and filter to known classes only
            classes = [c for c in self.global_classes if c in self.class_to_id]

        if not classes:
            return self._create_empty_results(h, w)

        # Encode image once, then query each class sequentially
        all_masks, all_boxes, all_scores, all_labels = [], [], [], []

        use_autocast = (self.device == "cuda")
        with torch.no_grad():
            with torch.autocast(device_type="cuda" if use_autocast else "cpu",
                                dtype=torch.bfloat16,
                                enabled=use_autocast):

                inference_state = self.processor.set_image(image)

                for cls in classes:
                    try:
                        output = self.processor.set_text_prompt(
                            state=inference_state,
                            prompt=cls
                        )
                    except Exception as e:
                        print(f"  SAM3 failed for class '{cls}': {e}")
                        continue

                    masks = output.get("masks")
                    boxes = output.get("boxes")
                    scores = output.get("scores")

                    if masks is None or (hasattr(masks, '__len__') and len(masks) == 0):
                        continue

                    # Normalise to numpy (cast to float32 first — bfloat16 is not numpy-compatible)
                    if isinstance(masks, torch.Tensor):
                        masks = masks.cpu().float().numpy()
                    if isinstance(boxes, torch.Tensor):
                        boxes = boxes.cpu().float().numpy()
                    if isinstance(scores, torch.Tensor):
                        scores = scores.cpu().float().numpy()

                    masks = np.array(masks)
                    if masks.ndim == 2:
                        masks = masks[np.newaxis]       # (1, H, W)
                    elif masks.ndim == 4:
                        masks = masks.squeeze(1)        # (N, H, W)

                    # Scale boxes to pixel space if they appear normalised
                    if boxes is not None and len(boxes) > 0:
                        boxes = np.array(boxes, dtype=np.float32)
                        if boxes.ndim == 1:
                            boxes = boxes[np.newaxis]
                        if boxes.max() <= 1.0:
                            boxes = boxes * np.array([w, h, w, h], dtype=np.float32)

                    scores_arr = (
                        scores[:len(masks)]
                        if (scores is not None and len(scores) >= len(masks))
                        else np.ones(len(masks), dtype=np.float32)
                    )

                    # Apply confidence threshold
                    keep = scores_arr >= box_threshold
                    masks = masks[keep]
                    boxes = boxes[keep] if (boxes is not None and len(boxes) > 0) else boxes
                    scores_arr = scores_arr[keep]

                    n = len(masks)
                    if n == 0:
                        print(f"  {cls}: 0 instance(s) above threshold {box_threshold}")
                        continue

                    all_masks.extend(masks)
                    all_boxes.extend(
                        boxes[:n].tolist() if (boxes is not None and len(boxes) >= n)
                        else [[0.0, 0.0, float(w), float(h)]] * n
                    )
                    all_scores.extend(scores_arr.tolist())
                    all_labels.extend([cls] * n)
                    print(f"  {cls}: {n} instance(s) (score >= {box_threshold})")

        if not all_labels:
            return self._create_empty_results(h, w)

        all_masks = np.stack(all_masks).astype(bool)         # (N, H, W)
        all_boxes = np.array(all_boxes, dtype=np.float32)    # (N, 4) xyxy pixels
        class_ids = np.array([self.class_to_id[lbl] for lbl in all_labels])

        self.last_masks = all_masks
        self.last_boxes = all_boxes
        self.last_class_names = all_labels
        self.last_confidences = all_scores
        self.last_detections = sv.Detections(
            xyxy=all_boxes,
            mask=all_masks,
            class_id=class_ids
        )

        print(f"SAM3: {len(all_labels)} instance(s) across "
              f"{len(set(all_labels))} class(es)")
        print(f"Class IDs used: {class_ids.tolist()}")

        return {
            'boxes': all_boxes,
            'masks': all_masks,
            'class_names': all_labels,
            'confidences': all_scores,
            'class_ids': class_ids,
            'image_shape': (h, w),
            'detections': self.last_detections,
        }

    def _create_empty_results(self, h: int, w: int) -> Dict:
        self.last_detections = None
        return {
            'boxes': np.array([]),
            'masks': np.array([]),
            'class_names': [],
            'confidences': [],
            'class_ids': np.array([]),
            'image_shape': (h, w),
            'detections': sv.Detections.empty(),
        }

    # -------------------------------------------------------------------------
    # Save utilities (same schema as GroundedSAM2Detector)
    # -------------------------------------------------------------------------

    def save_yolo_format(self,
                         output_dir: str,
                         dataset_name: str = "dataset",
                         split: str = "train") -> Dict[str, str]:
        if self.last_detections is None:
            raise RuntimeError("No results to save. Call process_image() first.")

        import shutil

        dataset_dir = Path(output_dir) / dataset_name
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        image_name = Path(self.last_image_path).stem
        image_ext = Path(self.last_image_path).suffix
        target_image_path = images_dir / f"{image_name}{image_ext}"
        shutil.copy2(self.last_image_path, target_image_path)

        h, w = self.last_image_shape[:2]
        annotation_path = labels_dir / f"{image_name}.txt"

        with open(annotation_path, 'w') as f:
            for mask, box, class_name in zip(
                    self.last_masks, self.last_boxes, self.last_class_names):
                class_id = self.class_to_id[class_name]
                polygon = self._mask_to_polygon(mask, w, h)
                if polygon is not None:
                    polygon_str = ' '.join([f"{c:.6f}" for c in polygon])
                    f.write(f"{class_id} {polygon_str}\n")
                # else:
                #     x1, y1, x2, y2 = box
                #     xc = ((x1 + x2) / 2) / w
                #     yc = ((y1 + y2) / 2) / h
                #     bw = (x2 - x1) / w
                #     bh = (y2 - y1) / h
                #     f.write(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        classes_path = dataset_dir / "classes.txt"
        if not self.classes_file_saved:
            with open(classes_path, 'w') as f:
                for idx in sorted(self.id_to_class.keys()):
                    f.write(f"{self.id_to_class[idx]}\n")
            self.classes_file_saved = True
            print(f"Saved classes.txt with {len(self.id_to_class)} classes")

        yaml_path = dataset_dir / "dataset.yaml"
        if not yaml_path.exists():
            self._save_yolo_yaml(yaml_path, dataset_name, len(self.class_to_id))

        print(f"YOLO annotation saved: {annotation_path}")

        return {
            'dataset_dir': str(dataset_dir),
            'image_path': str(target_image_path),
            'annotation_path': str(annotation_path),
            'classes_path': str(classes_path),
            'yaml_path': str(yaml_path),
        }

    def _mask_to_polygon(self, mask, img_w, img_h):
        try:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None
            largest = max(contours, key=cv2.contourArea)
            epsilon = 0.001 * cv2.arcLength(largest, True)  # much smoother
            simplified = cv2.approxPolyDP(largest, epsilon, True)
            polygon = []
            for point in simplified:
                x, y = point[0]
                polygon.extend([x / img_w, y / img_h])
            return polygon if len(polygon) >= 6 else None
        except Exception as e:
            print(f"Error converting mask to polygon: {e}")
            return None

    def _save_yolo_yaml(self,
                        yaml_path: Path,
                        dataset_name: str,
                        num_classes: int) -> None:
        lines = [
            "# YOLO dataset configuration",
            "# Generated by SAM3Detector",
            "",
            f"name: {dataset_name}",
            f"nc: {num_classes}",
            "",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            "names:",
        ]
        for class_id in range(num_classes):
            lines.append(f"  {class_id}: {self.id_to_class[class_id]}")
        yaml_path.write_text('\n'.join(lines) + '\n')

    def save_visualizations(self, output_dir: str) -> Dict[str, str]:
        if self.last_detections is None:
            raise RuntimeError("No results to visualize. Call process_image() first.")

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        img = cv2.imread(self.last_image_path)
        image_name = Path(self.last_image_path).stem

        labels = [
            f"{cls} {conf:.2f}"
            for cls, conf in zip(self.last_class_names, self.last_confidences)
        ]

        annotated = sv.BoxAnnotator().annotate(
            scene=img.copy(), detections=self.last_detections)
        annotated = sv.LabelAnnotator().annotate(
            scene=annotated, detections=self.last_detections, labels=labels)
        boxes_path = os.path.join(output_dir, f"{image_name}_boxes.jpg")
        cv2.imwrite(boxes_path, annotated)

        annotated = sv.MaskAnnotator().annotate(
            scene=annotated, detections=self.last_detections)
        masks_path = os.path.join(output_dir, f"{image_name}_masks.jpg")
        cv2.imwrite(masks_path, annotated)

        return {'boxes_image': boxes_path, 'masks_image': masks_path}

    def save_json_results(self, output_path: str) -> str:
        if self.last_detections is None:
            raise RuntimeError("No results to save. Call process_image() first.")

        import pycocotools.mask as mask_util

        h, w = self.last_image_shape[:2]

        def mask_to_rle(mask):
            rle = mask_util.encode(
                np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle

        results = {
            "image_path": self.last_image_path,
            "image_width": w,
            "image_height": h,
            "annotator": "sam3",
            "class_mapping": self.class_to_id,
            "annotations": [
                {
                    "class_name": cls,
                    "class_id": self.class_to_id[cls],
                    "bbox": box.tolist(),
                    "segmentation": mask_to_rle(mask),
                    "confidence": conf,
                }
                for cls, box, mask, conf in zip(
                    self.last_class_names, self.last_boxes,
                    self.last_masks, self.last_confidences
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
                              dataset_name: str = "sam3_dataset",
                              split: str = "train",
                              box_threshold: float = 0.45,
                              text_threshold: float = 0.45) -> Dict[str, Union[str, Dict]]:
        results = self.process_image(
            image_path=image_path,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

        if len(results['boxes']) == 0:
            print("No objects detected, skipping save operations.")
            return {'results': results, 'saved_files': {}}

        saved_files = {}
        saved_files['yolo'] = self.save_yolo_format(output_dir, dataset_name, split)

        viz_dir = os.path.join(output_dir, "visualizations")
        saved_files['visualizations'] = self.save_visualizations(viz_dir)

        json_path = os.path.join(output_dir, f"{Path(image_path).stem}_results.json")
        saved_files['json'] = self.save_json_results(json_path)

        return {'results': results, 'saved_files': saved_files}

    def get_summary(self) -> Dict:
        return {
            'device': self.device,
            'models_loaded': self.is_loaded,
            'last_image_processed': self.last_image_path,
            'last_detections_count': (
                len(self.last_boxes) if self.last_boxes is not None else 0),
            'available_classes': list(self.class_to_id.keys()),
            'class_count': len(self.class_to_id),
            'global_classes_set': len(self.global_classes) > 0,
            'classes_file_saved': self.classes_file_saved,
            'annotator': 'sam3',
        }
