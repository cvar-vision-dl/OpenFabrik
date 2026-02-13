import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import cv2
import argparse
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend globally to prevent tkinter issues
import matplotlib.pyplot as plt
import warnings
import os
from tqdm import tqdm
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import yaml

warnings.filterwarnings('ignore')

from show import *

# Try to import both SAM and SAM 2
try:
    from per_segment_anything import sam_model_registry, SamPredictor

    PERSAM_AVAILABLE = True
except ImportError:
    print("Warning: PerSAM not available, trying regular SAM")
    try:
        from segment_anything import sam_model_registry, SamPredictor

        PERSAM_AVAILABLE = False
    except ImportError:
        print("Error: Neither PerSAM nor SAM is available")
        exit(1)

# Try to import SAM 2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    SAM2_AVAILABLE = True
except ImportError:
    print("SAM 2 not available - will use SAM only")
    SAM2_AVAILABLE = False


class Mask_Weights(nn.Module):
    """Learnable mask weights for PerSAM-F training refinement"""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


class PerSAMProcessor:
    """
    Personalized SAM processor with SAM 2 integration and YOLO dataset support
    """

    def __init__(self, args):
        """
        Initialize the PerSAM processor

        Args:
            args: Parsed arguments containing model configurations
        """
        self.args = args
        self.sam_predictor = None
        self.sam2_predictor = None

        # Initialize models based on workflow
        self._initialize_models()

    def _initialize_models(self):
        """Initialize SAM and/or SAM 2 models based on workflow"""
        if self.args.workflow in ['persam', 'persam_to_sam2']:
            self.sam_predictor = self._initialize_sam_model(self.args.sam_type, self.args.sam_ckpt)

            # Freeze SAM parameters if training is enabled
            if self.args.enable_training:
                for name, param in self.sam_predictor.model.named_parameters():
                    param.requires_grad = False
                print("SAM parameters frozen for PerSAM-F training")

        if self.args.workflow in ['sam2', 'persam_to_sam2']:
            if not SAM2_AVAILABLE:
                raise ImportError("SAM 2 is not available but required for this workflow")
            self.sam2_predictor = self._initialize_sam2_model(self.args.sam2_config, self.args.sam2_ckpt)

    def _initialize_sam_model(self, sam_type, sam_ckpt):
        """Initialize SAM model"""
        print(f"======> Loading SAM {sam_type}")

        if sam_type == 'vit_h':
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
        elif sam_type in ['vit_l', 'vit_b']:
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
        elif sam_type == 'vit_t':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
            sam.eval()
        else:
            raise ValueError(f"Unsupported SAM type: {sam_type}")

        return SamPredictor(sam)

    def _initialize_sam2_model(self, sam2_config, sam2_ckpt):
        """Initialize SAM 2 model"""
        print(f"======> Loading SAM 2 with config {sam2_config}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam2_model = build_sam2(sam2_config, sam2_ckpt, device=device)
        return SAM2ImagePredictor(sam2_model)

    def load_images(self, ref_image_path, ref_mask_path, test_image_path):
        """
        Load reference image, reference mask, and test image

        Returns:
            tuple: (ref_image, ref_mask, test_image) in RGB format
        """
        print("======> Loading images")
        ref_image = self._load_image(ref_image_path)
        ref_mask = self._load_image(ref_mask_path)
        test_image = self._load_image(test_image_path)
        return ref_image, ref_mask, test_image

    @staticmethod
    def _load_image(image_path):
        """Load and convert image from BGR to RGB"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def extract_target_embedding(self, ref_image, ref_mask, enable_training=False):
        """
        Extract target embedding from reference image and mask

        Returns:
            tuple: (target_feat, target_embedding, gt_mask) if enable_training=True
                   (target_feat, target_embedding) if enable_training=False
        """
        print("======> Extracting target embedding from reference")

        # Process reference mask and image
        ref_mask_tensor = self.sam_predictor.set_image(ref_image, ref_mask)
        ref_feat = self.sam_predictor.features.squeeze().permute(1, 2, 0)

        # Resize mask to feature map size
        ref_mask_tensor = F.interpolate(ref_mask_tensor, size=ref_feat.shape[0:2], mode="bilinear")
        ref_mask_tensor = ref_mask_tensor.squeeze()[0]

        # Extract target features from masked region
        target_feat = ref_feat[ref_mask_tensor > 0]

        if enable_training:
            # PerSAM-F enhanced feature extraction
            target_feat_mean = target_feat.mean(0)
            target_feat_max = torch.max(target_feat, dim=0)[0]
            target_embedding = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

            # Prepare ground truth mask for training
            gt_mask_binary = ref_mask[:, :, 0] > 128
            gt_mask = torch.tensor(gt_mask_binary, dtype=torch.float32).cuda()

            # Resize ground truth mask to match SAM's prediction resolution
            sam_output_size = (ref_image.shape[0] // 4, ref_image.shape[1] // 4)
            gt_mask_resized = F.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0),
                size=sam_output_size,
                mode="bilinear"
            )
            gt_mask = (gt_mask_resized > 0.5).float().flatten(1)

            print(f"PerSAM-F mode: Enhanced feature extraction enabled")
            print(f"Original image size: {ref_image.shape[:2]}")
            print(f"SAM output size: {sam_output_size}")
            print(f"GT mask shape: {gt_mask.shape}, pixels: {gt_mask.sum().item()}")
        else:
            # Original PerSAM feature extraction
            target_embedding = target_feat.mean(0).unsqueeze(0)
            gt_mask = None

        target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
        target_embedding = target_embedding.unsqueeze(0)

        if enable_training:
            return target_feat, target_embedding, gt_mask
        else:
            return target_feat, target_embedding

    def train_mask_weights(self, ref_image, ref_mask, target_feat, target_embedding, gt_mask):
        """Train learnable mask weights using PerSAM-F approach"""
        print('======> Starting PerSAM-F Training')

        # Get similarity map and points for reference image
        self.sam_predictor.set_image(ref_image, ref_mask)
        ref_feat = self.sam_predictor.features.squeeze().permute(1, 2, 0)

        # Compute cosine similarity
        h, w, C = ref_feat.shape
        target_feat_norm = target_feat / target_feat.norm(dim=-1, keepdim=True)
        ref_feat_norm = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat_flat = ref_feat_norm.permute(2, 0, 1).reshape(C, h * w)
        sim = target_feat_norm @ ref_feat_flat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.sam_predictor.model.postprocess_masks(
            sim,
            input_size=self.sam_predictor.input_size,
            original_size=self.sam_predictor.original_size).squeeze()

        # Get positive location prior
        topk_xy, topk_label = self._point_selection_positive_only(sim, topk=1)

        # Initialize learnable mask weights
        mask_weights = Mask_Weights().cuda()
        mask_weights.train()

        optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=self.args.lr, eps=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.train_epoch)

        print(f"Training for {self.args.train_epoch} epochs with lr={self.args.lr}")
        print(f"Ground truth mask shape: {gt_mask.shape}")

        for train_idx in tqdm(range(self.args.train_epoch), desc="Training mask weights"):
            # Run the decoder
            masks, scores, logits, logits_high = self.sam_predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=True)

            # Check tensor sizes for debugging
            if train_idx == 0:
                print(f"Logits high shape: {logits_high.shape}")
                print(f"GT mask shape: {gt_mask.shape}")

            logits_high = logits_high.flatten(1)

            # Ensure gt_mask matches logits_high size
            if logits_high.shape[1] != gt_mask.shape[1]:
                print(f"Resizing GT mask from {gt_mask.shape} to match logits {logits_high.shape}")
                gt_h = int(np.sqrt(gt_mask.shape[1]))
                gt_w = gt_mask.shape[1] // gt_h
                gt_mask_2d = gt_mask.reshape(1, 1, gt_h, gt_w)

                logits_h = int(np.sqrt(logits_high.shape[1]))
                logits_w = logits_high.shape[1] // logits_h

                gt_mask_resized = F.interpolate(gt_mask_2d, size=(logits_h, logits_w), mode="bilinear")
                gt_mask = (gt_mask_resized > 0.5).float().flatten(1)
                print(f"Resized GT mask to: {gt_mask.shape}")

            # Weighted sum three-scale masks
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            logits_high = logits_high * weights
            logits_high = logits_high.sum(0).unsqueeze(0)

            dice_loss = self._calculate_dice_loss(logits_high, gt_mask)
            focal_loss = self._calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if train_idx % self.args.log_epoch == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(
                    f'Epoch {train_idx}/{self.args.train_epoch} - LR: {current_lr:.6f}, Dice: {dice_loss.item():.4f}, Focal: {focal_loss.item():.4f}')

        mask_weights.eval()
        final_weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
        weights_np = final_weights.detach().cpu().numpy()
        print(f'======> Learned mask weights: {weights_np.flatten()}')

        return final_weights

    def segment_test_image(self, test_image, target_feat, target_embedding, mask_weights=None):
        """Segment target object in test image using target embedding"""
        print("======> Segmenting test image")

        # Encode test image
        self.sam_predictor.set_image(test_image)
        test_feat = self.sam_predictor.features.squeeze()

        # Compute cosine similarity between target and test features
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat

        # Reshape and upscale similarity map
        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = self.sam_predictor.model.postprocess_masks(
            sim,
            input_size=self.sam_predictor.input_size,
            original_size=self.sam_predictor.original_size).squeeze()

        # Select positive and negative points based on similarity
        topk_xy_i, topk_label_i, last_xy_i, last_label_i = self._point_selection(sim, topk=1)
        topk_xy = np.concatenate([topk_xy_i, last_xy_i], axis=0)
        topk_label = np.concatenate([topk_label_i, last_label_i], axis=0)

        # Prepare attention guidance
        sim = (sim - sim.mean()) / torch.std(sim)
        sim = F.interpolate(sim.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim.sigmoid_().unsqueeze(0).flatten(3)

        if mask_weights is not None:
            # PerSAM-F mode with learned weights
            print("Using PerSAM-F learned weights")

            # First prediction with target guidance and learned weights
            masks, scores, logits, logits_high = self.sam_predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=True,
                attn_sim=attn_sim,
                target_embedding=target_embedding
            )

            # Apply learned weights
            weights_np = mask_weights.detach().cpu().numpy()
            logits_high = logits_high * mask_weights.unsqueeze(-1)
            logit_high = logits_high.sum(0)
            mask = (logit_high > 0).detach().cpu().numpy()

            logits = logits * weights_np[..., None]
            logit = logits.sum(0)

            # Cascaded refinement with weighted logits
            y, x = np.nonzero(mask)
            if len(y) > 0 and len(x) > 0:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])

                masks, scores, logits, _ = self.sam_predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logit[None, :, :],
                    multimask_output=True)
                best_idx = np.argmax(scores)

                # Second refinement
                y, x = np.nonzero(masks[best_idx])
                if len(y) > 0 and len(x) > 0:
                    x_min, x_max = x.min(), x.max()
                    y_min, y_max = y.min(), y.max()
                    input_box = np.array([x_min, y_min, x_max, y_max])

                    masks, scores, logits, _ = self.sam_predictor.predict(
                        point_coords=topk_xy,
                        point_labels=topk_label,
                        box=input_box[None, :],
                        mask_input=logits[best_idx: best_idx + 1, :, :],
                        multimask_output=True)
                    best_idx = np.argmax(scores)
            else:
                best_idx = 0
        else:
            # Original PerSAM mode
            print("Using original PerSAM approach")

            # First prediction with target guidance
            masks, scores, logits, _ = self.sam_predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                multimask_output=False,
                attn_sim=attn_sim,
                target_embedding=target_embedding
            )
            best_idx = 0

            # Cascaded refinement step 1
            masks, scores, logits, _ = self.sam_predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                mask_input=logits[best_idx: best_idx + 1, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)

            # Cascaded refinement step 2 (with bounding box)
            y, x = np.nonzero(masks[best_idx])
            if len(y) > 0 and len(x) > 0:
                x_min, x_max = x.min(), x.max()
                y_min, y_max = y.min(), y.max()
                input_box = np.array([x_min, y_min, x_max, y_max])

                masks, scores, logits, _ = self.sam_predictor.predict(
                    point_coords=topk_xy,
                    point_labels=topk_label,
                    box=input_box[None, :],
                    mask_input=logits[best_idx: best_idx + 1, :, :],
                    multimask_output=True)
                best_idx = np.argmax(scores)

        return masks[best_idx], topk_xy, topk_label

    def segment_with_sam2(self, test_image, input_points=None, input_labels=None, input_box=None):
        """Segment with SAM 2 using provided prompts"""
        print("======> Segmenting with SAM 2")

        # Set image for SAM 2
        self.sam2_predictor.set_image(test_image)

        # Prepare inputs
        kwargs = {'multimask_output': True}

        if input_points is not None and input_labels is not None:
            kwargs['point_coords'] = input_points
            kwargs['point_labels'] = input_labels
            print(f"Using {len(input_points)} points from PerSAM")

        if input_box is not None:
            kwargs['box'] = input_box[None, :]  # SAM 2 expects box in shape (1, 4)
            print(f"Using bounding box from PerSAM: {input_box}")

        # Predict with SAM 2
        masks, scores, logits = self.sam2_predictor.predict(**kwargs)

        # Select best mask and normalize
        best_idx = np.argmax(scores)
        best_mask = self._normalize_mask(masks[best_idx])

        print(f"SAM 2 generated {len(masks)} masks, best score: {scores[best_idx]:.3f}")
        print(f"Best mask dtype: {masks[best_idx].dtype} -> normalized to: {best_mask.dtype}")

        return masks, scores, best_mask

    def run_persam_workflow(self, ref_image, ref_mask, test_image):
        """Run the complete PerSAM workflow"""

        # Extract target embedding from reference
        if self.args.enable_training:
            target_feat, target_embedding, gt_mask = self.extract_target_embedding(
                ref_image, ref_mask, enable_training=True
            )

            # Train mask weights
            mask_weights = self.train_mask_weights(
                ref_image, ref_mask, target_feat, target_embedding, gt_mask
            )

            # Segment test image with learned weights
            final_mask, topk_xy, topk_label = self.segment_test_image(
                test_image, target_feat, target_embedding, mask_weights
            )

            print("======> PerSAM-F segmentation completed!")

        else:
            target_feat, target_embedding = self.extract_target_embedding(
                ref_image, ref_mask, enable_training=False
            )

            # Segment test image with original method
            final_mask, topk_xy, topk_label = self.segment_test_image(
                test_image, target_feat, target_embedding
            )

            print("======> PerSAM segmentation completed!")

        return final_mask, topk_xy, topk_label

    def run_pipeline(self, ref_image_path, ref_mask_path, test_image_path):
        """
        Run the complete pipeline based on the specified workflow

        Returns:
            dict: Results containing masks, points, and other outputs
        """
        # Load images
        ref_image, ref_mask, test_image = self.load_images(ref_image_path, ref_mask_path, test_image_path)

        results = {}

        if self.args.workflow == 'persam':
            print("======> Executing PerSAM workflow")
            final_mask, topk_xy, topk_label = self.run_persam_workflow(ref_image, ref_mask, test_image)
            results = {
                'final_mask': final_mask,
                'points': topk_xy,
                'labels': topk_label,
                'method': 'PerSAM'
            }
            self.save_results(test_image, final_mask, topk_xy, topk_label,
                              self.args.output_dir, "persam_result", "PerSAM")

        elif self.args.workflow == 'sam2':
            print("======> Executing SAM 2 only workflow")
            # For SAM 2 only, use center point as fallback
            center_point = np.array([[test_image.shape[1] // 2, test_image.shape[0] // 2]])
            center_label = np.array([1])

            masks, scores, final_mask = self.segment_with_sam2(test_image, center_point, center_label)
            results = {
                'final_mask': final_mask,
                'points': center_point,
                'labels': center_label,
                'method': 'SAM 2'
            }
            self.save_results(test_image, final_mask, center_point, center_label,
                              self.args.output_dir, "sam2_result", "SAM 2")

        elif self.args.workflow == 'persam_to_sam2':
            print("======> Executing PerSAM → SAM 2 pipeline")

            # Step 1: Run PerSAM to get initial segmentation and prompts
            persam_mask, persam_points, persam_labels = self.run_persam_workflow(
                ref_image, ref_mask, test_image
            )

            # Step 2: Extract prompts for SAM 2
            sam2_points = None
            sam2_labels = None
            sam2_box = None

            if self.args.use_persam_points:
                sam2_points = persam_points
                sam2_labels = persam_labels

            if self.args.use_persam_box:
                sam2_box = self._extract_bounding_box(persam_mask)

            # If no specific prompts requested, use both points and box
            if not self.args.use_persam_points and not self.args.use_persam_box:
                sam2_points = persam_points
                sam2_labels = persam_labels
                sam2_box = self._extract_bounding_box(persam_mask)

            # Step 3: Run SAM 2 with PerSAM prompts
            masks, scores, sam2_mask = self.segment_with_sam2(
                test_image, sam2_points, sam2_labels, sam2_box
            )

            # Step 4: Save results
            self.save_results(test_image, persam_mask, persam_points, persam_labels,
                              self.args.output_dir, "persam_result", "PerSAM")
            self.save_results(test_image, sam2_mask, sam2_points, sam2_labels,
                              self.args.output_dir, "sam2_refined_result", "SAM 2 (PerSAM-guided)")

            if self.args.compare_results:
                self.save_comparison(test_image, persam_mask, sam2_mask,
                                     persam_points, persam_labels, self.args.output_dir)

            results = {
                'persam_mask': persam_mask,
                'sam2_mask': sam2_mask,
                'persam_points': persam_points,
                'persam_labels': persam_labels,
                'sam2_points': sam2_points,
                'sam2_labels': sam2_labels,
                'sam2_box': sam2_box,
                'method': 'PerSAM → SAM 2'
            }

            print(f"======> Pipeline completed successfully!")
            print(f"PerSAM mask pixels: {np.sum(persam_mask)}")
            print(f"SAM 2 refined mask pixels: {np.sum(sam2_mask)}")

        return results

    def _save_mask_visualization(self, image, mask, output_path):
        """
        Save visualization of mask overlaid on image

        Args:
            image: Original image (RGB format)
            mask: Binary mask
            output_path: Path to save visualization
        """
        # Normalize mask to boolean type
        mask = self._normalize_mask(mask)

        # Create figure with explicit size
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Display image and mask
        ax.imshow(image)
        show_mask(mask, ax)
        ax.axis('off')

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save with high quality
        fig.savefig(output_path, format='jpg', bbox_inches='tight', dpi=150, pad_inches=0)

        # Explicitly close figure and clear memory
        plt.close(fig)
        plt.clf()  # Clear current figure
        plt.cla()  # Clear current axes

    # ============================================================================
    # YOLO DATASET ANNOTATION METHODS
    # ============================================================================

    def mask_to_yolo_polygon(self, mask, image_width, image_height):
        """
        Convert binary mask to YOLO polygon format (normalized coordinates)

        Args:
            mask: Binary mask array
            image_width: Original image width
            image_height: Original image height

        Returns:
            List of normalized polygon coordinates [x1, y1, x2, y2, ...]
        """
        # Find contours in the mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify the contour to reduce points
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Convert to normalized coordinates
        polygon = []
        for point in simplified_contour:
            x, y = point[0]
            # Normalize coordinates
            x_norm = x / image_width
            y_norm = y / image_height
            # Clamp to [0, 1]
            x_norm = max(0, min(1, x_norm))
            y_norm = max(0, min(1, y_norm))
            polygon.extend([x_norm, y_norm])

        return polygon

    def save_yolo_annotation(self, mask, class_id, image_width, image_height, output_path):
        """
        Save single YOLO annotation file

        Args:
            mask: Binary mask
            class_id: Class ID for the object
            image_width: Original image width
            image_height: Original image height
            output_path: Path to save the .txt annotation file
        """
        polygon = self.mask_to_yolo_polygon(mask, image_width, image_height)

        if not polygon:
            # Create empty annotation file if no valid polygon
            with open(output_path, 'w') as f:
                pass
            return

        # Write YOLO annotation
        with open(output_path, 'w') as f:
            # Format: class_id x1 y1 x2 y2 x3 y3 ...
            polygon_str = ' '.join([f'{coord:.6f}' for coord in polygon])
            f.write(f'{class_id} {polygon_str}\n')

    def create_yolo_dataset_structure(self, output_dir, class_names, split_ratios=None):
        """
        Create YOLO dataset directory structure and metadata files

        Args:
            output_dir: Output directory for YOLO dataset
            class_names: List of class names
            split_ratios: Dict with train/val/test ratios (optional)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for split in ['train', 'val']:
            (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
            (output_path / split / 'visualizations').mkdir(parents=True, exist_ok=True)

        # Create classes.txt
        with open(output_path / 'classes.txt', 'w') as f:
            for class_name in class_names:
                f.write(f'{class_name}\n')

        # Create dataset.yaml
        dataset_config = {
            'path': str(output_path.absolute()),
            'train': 'train',
            'val': 'val',
            'nc': len(class_names),
            'names': class_names
        }

        with open(output_path / 'dataset.yaml', 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)

        print(f"Created YOLO dataset structure in: {output_path}")
        print(f"Includes: images/, labels/, and visualizations/ folders for train/val splits")
        print(f"Classes: {class_names}")

    def annotate_dataset(self, dataset_dir, ref_image_path, ref_mask_path,
                         output_dir, class_name, split_ratio=0.8):
        """
        Annotate all images in dataset directory using PerSAM → SAM 2

        Args:
            dataset_dir: Directory containing images to annotate
            ref_image_path: Reference image path for PerSAM
            ref_mask_path: Reference mask path for PerSAM
            output_dir: Output directory for YOLO dataset
            class_name: Name of the class
            split_ratio: Ratio for train/val split
        """
        print(f"======> Starting dataset annotation")
        print(f"Dataset directory: {dataset_dir}")
        print(f"Reference image: {ref_image_path}")
        print(f"Reference mask: {ref_mask_path}")
        print(f"Output directory: {output_dir}")
        print(f"Class name: {class_name}")
        print(f"Note: Mask visualizations will be saved alongside annotations")

        # Load reference image and mask
        ref_image, ref_mask, _ = self.load_images(ref_image_path, ref_mask_path, ref_image_path)

        # Extract target embedding (only once for all images)
        if self.args.enable_training:
            target_feat, target_embedding, gt_mask = self.extract_target_embedding(
                ref_image, ref_mask, enable_training=True
            )
            # Train mask weights
            mask_weights = self.train_mask_weights(
                ref_image, ref_mask, target_feat, target_embedding, gt_mask
            )
        else:
            target_feat, target_embedding = self.extract_target_embedding(
                ref_image, ref_mask, enable_training=False
            )
            mask_weights = None

        # Create YOLO dataset structure
        self.create_yolo_dataset_structure(output_dir, [class_name])

        # Get all image files
        dataset_path = Path(dataset_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []

        for ext in image_extensions:
            image_files.extend(dataset_path.glob(f'*{ext}'))
            image_files.extend(dataset_path.glob(f'*{ext.upper()}'))

        if not image_files:
            raise ValueError(f"No images found in {dataset_dir}")

        print(f"Found {len(image_files)} images to annotate")

        # Split images into train/val
        random.shuffle(image_files)
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        print(f"Train images: {len(train_files)}")
        print(f"Val images: {len(val_files)}")

        # Process images
        output_path = Path(output_dir)
        class_id = 0  # Single class

        successful_annotations = 0
        failed_annotations = 0

        for split_name, files in [('train', train_files), ('val', val_files)]:
            print(f"\n======> Processing {split_name} split ({len(files)} images)")

            for i, image_file in enumerate(tqdm(files, desc=f"Annotating {split_name}")):
                try:
                    # Load test image
                    test_image = self._load_image(str(image_file))
                    image_height, image_width = test_image.shape[:2]

                    # Run PerSAM → SAM 2 pipeline
                    if self.args.workflow == 'persam_to_sam2':
                        # Step 1: PerSAM segmentation
                        persam_mask, persam_points, persam_labels = self.segment_test_image(
                            test_image, target_feat, target_embedding, mask_weights
                        )

                        # Step 2: SAM 2 refinement
                        sam2_box = self._extract_bounding_box(persam_mask)
                        _, _, final_mask = self.segment_with_sam2(
                            test_image, persam_points, persam_labels, sam2_box
                        )
                    else:
                        # Use only PerSAM
                        final_mask, _, _ = self.segment_test_image(
                            test_image, target_feat, target_embedding, mask_weights
                        )

                    # Normalize mask
                    final_mask = self._normalize_mask(final_mask)

                    # Copy image to YOLO dataset
                    image_filename = f"{image_file.stem}.jpg"
                    image_dst = output_path / split_name / 'images' / image_filename

                    # Convert and save image
                    image_pil = Image.fromarray(test_image)
                    image_pil.save(image_dst, 'JPEG', quality=95)

                    # Save YOLO annotation
                    annotation_filename = f"{image_file.stem}.txt"
                    annotation_dst = output_path / split_name / 'labels' / annotation_filename

                    self.save_yolo_annotation(
                        final_mask, class_id, image_width, image_height, annotation_dst
                    )

                    # Save visualization (mask overlay on image)
                    vis_filename = f"{image_file.stem}_mask_overlay.jpg"
                    vis_dst = output_path / split_name / 'visualizations' / vis_filename
                    self._save_mask_visualization(test_image, final_mask, vis_dst)

                    successful_annotations += 1

                except Exception as e:
                    print(f"Failed to process {image_file.name}: {e}")
                    failed_annotations += 1
                    continue

        # Save annotation summary
        summary = {
            'total_images': len(image_files),
            'successful_annotations': successful_annotations,
            'failed_annotations': failed_annotations,
            'class_name': class_name,
            'class_id': class_id,
            'train_images': len(train_files),
            'val_images': len(val_files),
            'reference_image': str(ref_image_path),
            'reference_mask': str(ref_mask_path),
            'visualizations_saved': successful_annotations,  # Same as successful annotations
            'visualization_format': 'jpg'
        }

        with open(output_path / 'annotation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n======> Dataset annotation completed!")
        print(f"Successful: {successful_annotations}")
        print(f"Failed: {failed_annotations}")
        print(f"Success rate: {successful_annotations / (successful_annotations + failed_annotations) * 100:.1f}%")
        print(f"Dataset saved to: {output_path}")
        print(f"Visualizations saved to: train/visualizations and val/visualizations")

        return str(output_path)

    def save_results(self, test_image, final_mask, topk_xy, topk_label, output_dir,
                     filename_prefix="result", method_name=""):
        """Save visualization and mask results"""
        os.makedirs(output_dir, exist_ok=True)

        # Normalize mask to boolean type
        final_mask = self._normalize_mask(final_mask)

        # Save visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(test_image)
        show_mask(final_mask, ax)
        if topk_xy is not None and topk_label is not None:
            show_points(topk_xy, topk_label, ax)

        title = f"{method_name} Segmentation Result" if method_name else "Segmentation Result"
        ax.set_title(title, fontsize=18)
        ax.axis('off')

        vis_path = os.path.join(output_dir, f'{filename_prefix}_visualization.jpg')
        fig.savefig(vis_path, format='jpg', bbox_inches='tight', dpi=150)

        # Properly close figure
        plt.close(fig)
        plt.clf()
        plt.cla()

        # Save binary mask
        mask_colors = np.zeros((final_mask.shape[0], final_mask.shape[1], 3), dtype=np.uint8)
        mask_colors[final_mask, :] = np.array([255, 255, 255])  # White mask
        mask_path = os.path.join(output_dir, f'{filename_prefix}_mask.png')
        cv2.imwrite(mask_path, mask_colors)

        print(f"Results saved to {output_dir}")
        print(f"  Visualization: {vis_path}")
        print(f"  Mask: {mask_path}")

        return vis_path, mask_path

    def save_comparison(self, test_image, persam_mask, sam2_mask, persam_points, persam_labels, output_dir):
        """Save side-by-side comparison of PerSAM and SAM 2 results"""

        # Normalize masks to boolean type
        persam_mask = self._normalize_mask(persam_mask)
        sam2_mask = self._normalize_mask(sam2_mask)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(test_image)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis('off')

        # PerSAM result
        axes[1].imshow(test_image)
        show_mask(persam_mask, axes[1])
        if persam_points is not None and persam_labels is not None:
            show_points(persam_points, persam_labels, axes[1])
        axes[1].set_title("PerSAM Result", fontsize=14)
        axes[1].axis('off')

        # SAM 2 result
        axes[2].imshow(test_image)
        show_mask(sam2_mask, axes[2])
        if persam_points is not None and persam_labels is not None:
            show_points(persam_points, persam_labels, axes[2])
        axes[2].set_title("SAM 2 Result", fontsize=14)
        axes[2].axis('off')

        plt.tight_layout()
        comparison_path = os.path.join(output_dir, 'comparison_persam_vs_sam2.jpg')
        fig.savefig(comparison_path, format='jpg', bbox_inches='tight', dpi=150)

        # Properly close figure
        plt.close(fig)
        plt.clf()
        plt.cla()

        print(f"Comparison saved to: {comparison_path}")
        return comparison_path

    # Utility methods
    @staticmethod
    def _normalize_mask(mask):
        """Normalize mask to boolean type regardless of input format"""
        if mask.dtype == bool:
            return mask
        elif mask.dtype in [np.uint8, np.int32, np.int64]:
            return mask > 0
        elif mask.dtype in [np.float32, np.float64]:
            return mask > 0.5
        else:
            print(f"Warning: Unknown mask dtype {mask.dtype}, converting with > 0.5 threshold")
            return mask > 0.5

    @staticmethod
    def _extract_bounding_box(mask):
        """Extract bounding box from mask"""
        y, x = np.nonzero(mask)
        if len(y) == 0 or len(x) == 0:
            return None

        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        return np.array([x_min, y_min, x_max, y_max])

    @staticmethod
    def _point_selection(mask_sim, topk=1):
        """Select positive and negative points based on similarity map"""
        w, h = mask_sim.shape

        # Top-k point selection (positive points)
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()

        # Bottom-k point selection (negative points)
        last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
        last_x = (last_xy // h).unsqueeze(0)
        last_y = (last_xy - last_x * h)
        last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
        last_label = np.array([0] * topk)
        last_xy = last_xy.cpu().numpy()

        return topk_xy, topk_label, last_xy, last_label

    @staticmethod
    def _point_selection_positive_only(mask_sim, topk=1):
        """Select only positive points (for training mode)"""
        w, h = mask_sim.shape
        topk_xy = mask_sim.flatten(0).topk(topk)[1]
        topk_x = (topk_xy // h).unsqueeze(0)
        topk_y = (topk_xy - topk_x * h)
        topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
        topk_label = np.array([1] * topk)
        topk_xy = topk_xy.cpu().numpy()
        return topk_xy, topk_label

    @staticmethod
    def _calculate_dice_loss(inputs, targets, num_masks=1):
        """Compute the DICE loss, similar to generalized IOU for masks"""
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(-1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_masks

    @staticmethod
    def _calculate_sigmoid_focal_loss(inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2):
        """Loss used in RetinaNet for dense detection"""
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_masks


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Personalized SAM for single image segmentation with SAM 2 integration and YOLO dataset support')

    parser.add_argument('--ref_image', type=str, required=True,
                        help='Path to reference image')
    parser.add_argument('--ref_mask', type=str, required=True,
                        help='Path to reference mask (binary mask of target object)')
    parser.add_argument('--test_image', type=str, required=True,
                        help='Path to test image to segment')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Output directory for results')

    # Model selection - Changed default to persam_to_sam2
    parser.add_argument('--workflow', type=str, default='persam_to_sam2',
                        choices=['persam', 'sam2', 'persam_to_sam2'],
                        help='Workflow: persam only, sam2 only, or persam→sam2 pipeline (default: persam_to_sam2)')
    parser.add_argument('--sam_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
                        help='SAM model type')
    parser.add_argument('--sam_ckpt', type=str, default='sam_vit_h_4b8939.pth',
                        help='Path to SAM checkpoint')

    # SAM 2 options
    parser.add_argument('--sam2_config', type=str, default='sam2_hiera_l.yaml',
                        help='SAM 2 model config')
    parser.add_argument('--sam2_ckpt', type=str, default='sam2_hiera_large.pt',
                        help='Path to SAM 2 checkpoint')

    # PerSAM-F training parameters
    parser.add_argument('--enable_training', action='store_true',
                        help='Enable PerSAM-F training refinement')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for training refinement')
    parser.add_argument('--train_epoch', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--log_epoch', type=int, default=200,
                        help='Logging interval for training')

    # Pipeline options
    parser.add_argument('--use_persam_box', action='store_true',
                        help='Use PerSAM-generated bounding box for SAM 2')
    parser.add_argument('--use_persam_points', action='store_true',
                        help='Use PerSAM-generated points for SAM 2')
    parser.add_argument('--compare_results', action='store_true',
                        help='Save comparison between PerSAM and SAM 2 results')

    args = parser.parse_args()
    return args


def main():
    """Main function with argparse handling"""
    args = get_arguments()

    print("Arguments:", args)
    print(f"======> Workflow: {args.workflow}")

    try:
        # Initialize processor with arguments
        processor = PerSAMProcessor(args)

        # Run the pipeline
        results = processor.run_pipeline(args.ref_image, args.ref_mask, args.test_image)

        print("======> All workflows completed successfully!")
        print(f"======> Results: {results['method']}")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()