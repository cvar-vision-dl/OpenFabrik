#!/usr/bin/env python3
"""
YOLOv11 Training Script with Augmentations, Multi-Scale Training, and TensorBoard
Usage: python train_yolov11.py --data path/to/dataset.yaml --model yolo11n.pt --epochs 100
"""

import argparse
import os
import random
import shutil
import yaml
import glob
from pathlib import Path
from ultralytics import YOLO
import torch
from torch.utils.tensorboard import SummaryWriter


def parse_arguments():
    """Parse command line arguments for YOLO training."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 with augmentations, multi-scale training and TensorBoard logging",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file"
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Model to use (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)"
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Image size for training"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Epochs to wait for no observable improvement for early stopping"
    )

    # Output and logging
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Project directory"
    )

    parser.add_argument(
        "--name",
        type=str,
        default="exp",
        help="Experiment name"
    )

    parser.add_argument(
        "--save-period",
        type=int,
        default=1,
        help="Save checkpoint every x epochs"
    )

    # Multi-scale validation parameters
    parser.add_argument("--val-scales", type=int, nargs="*", default=None,
                        help="Multiple validation sizes to evaluate each epoch (e.g., --val-scales 320 640 832)")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Run multi-scale validation every N epochs (default 1)")
    parser.add_argument("--val-split", type=str, default="val", choices=["val", "test", "train"],
                        help="Dataset split for the multi-scale validation & sample predictions")
    parser.add_argument("--val-metric", type=str, default="map50",
                        choices=["map", "map50"],
                        help="Metric to track for best models at each scale")
    parser.add_argument("--override-best", action="store_true",
                        help="Also overwrite Ultralytics best.pt when any alt metric improves (optional)")
    parser.add_argument("--tb-samples", type=int, default=3,
                        help="Number of sample images to visualize in TensorBoard (default 3)")
    parser.add_argument("--tb-sample-images", type=str, nargs="*", default=None,
                        help="Specific image paths to use for TensorBoard visualization (overrides random sampling)")

    # Multi-scale training parameters
    parser.add_argument(
        "--multiscale",
        action="store_true",
        help="Enable multi-scale training (YOLO uses fixed 0.5x-1.5x range of base imgsz)"
    )

    parser.add_argument(
        "--multiscale-range",
        type=float,
        default=0.5,
        help="DEPRECATED: YOLO ignores this and uses fixed 0.5x-1.5x range"
    )

    parser.add_argument(
        "--multiscale-min",
        type=int,
        default=None,
        help="DEPRECATED: YOLO ignores this and uses fixed 0.5x-1.5x range"
    )

    parser.add_argument(
        "--multiscale-max",
        type=int,
        default=None,
        help="DEPRECATED: YOLO ignores this and uses fixed 0.5x-1.5x range"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to use (cuda:0, cpu, etc.). Empty string for auto-detection"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers"
    )

    # Augmentation parameters
    parser.add_argument(
        "--hsv-h",
        type=float,
        default=0.015,
        help="HSV-Hue augmentation (fraction)"
    )

    parser.add_argument(
        "--hsv-s",
        type=float,
        default=0.7,
        help="HSV-Saturation augmentation (fraction)"
    )

    parser.add_argument(
        "--hsv-v",
        type=float,
        default=0.4,
        help="HSV-Value augmentation (fraction)"
    )

    parser.add_argument(
        "--degrees",
        type=float,
        default=40,
        help="Rotation augmentation (degrees)"
    )

    parser.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Translation augmentation (fraction)"
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=0.9,
        help="Scale augmentation (fraction)"
    )

    parser.add_argument(
        "--shear",
        type=float,
        default=15,
        help="Shear augmentation (degrees)"
    )

    parser.add_argument(
        "--perspective",
        type=float,
        default=0.0004,
        help="Perspective augmentation (fraction)"
    )

    parser.add_argument(
        "--flipud",
        type=float,
        default=0.5,
        help="Vertical flip augmentation (probability)"
    )

    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Horizontal flip augmentation (probability)"
    )

    parser.add_argument(
        "--mosaic",
        type=float,
        default=0.9,
        help="Mosaic augmentation (probability)"
    )

    parser.add_argument(
        "--mixup",
        type=float,
        default=0.5,
        help="MixUp augmentation (probability)"
    )

    parser.add_argument(
        "--copy-paste",
        type=float,
        default=0.0,
        help="Copy-paste augmentation (probability)"
    )

    # Advanced options
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume training from checkpoint"
    )

    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained weights"
    )

    parser.add_argument(
        "--freeze",
        type=int,
        default=0,
        help="Number of layers to freeze (0 = no freezing)"
    )

    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache images for faster training"
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use Automatic Mixed Precision training"
    )

    parser.add_argument(
        "--plots",
        action="store_true",
        default=True,
        help="Generate training plots"
    )

    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    # Check if data file exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset file not found: {args.data}")

    # Validate model name
    valid_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
    if args.model not in valid_models and not os.path.exists(args.model):
        print(f"Warning: Model {args.model} not in standard models {valid_models}")
        print("Make sure the model path is correct or it will be downloaded automatically")

    # Validate ranges
    if args.epochs <= 0:
        raise ValueError("Epochs must be positive")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if args.imgsz <= 0:
        raise ValueError("Image size must be positive")

    # Validate multi-scale parameters
    if args.multiscale:
        print("Note: YOLO's built-in multi-scale training uses a fixed 0.5x-1.5x range")
        print("Custom range parameters (--multiscale-range, --multiscale-min/max) are ignored")
        print("Use --val-scales for multi-scale validation at specific sizes")

        if args.multiscale_range != 0.5:
            print(f"Warning: --multiscale-range {args.multiscale_range} will be ignored")

        if args.multiscale_min or args.multiscale_max:
            print(f"Warning: --multiscale-min/max will be ignored")

    # Validate validation scales
    if args.val_scales:
        for scale in args.val_scales:
            if scale < 32:
                raise ValueError(f"Validation scale {scale} must be at least 32")
            if scale % 32 != 0:
                print(f"Warning: Validation scale {scale} is not a multiple of 32, will be rounded")

    # Check GPU availability if device not specified
    if not args.device:
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available, using CPU")

    return True


def detect_task_type(model_or_metrics):
    """Detect the task type from model or metrics object."""
    if hasattr(model_or_metrics, 'task'):
        return model_or_metrics.task

    # Fallback: detect from available metrics
    if hasattr(model_or_metrics, 'seg'):
        return 'segment'
    elif hasattr(model_or_metrics, 'pose'):
        return 'pose'
    elif hasattr(model_or_metrics, 'box'):
        return 'detect'
    else:
        return 'detect'  # default assumption


def main():
    """Main training function."""
    args = parse_arguments()

    try:
        validate_arguments(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    print("=" * 60)
    print("YOLOv11 Training Configuration")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Image Size: {args.imgsz}")
    print(f"Learning Rate: {args.lr}")
    print(f"Device: {args.device or 'auto'}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")

    # Multi-scale training info
    if args.multiscale:
        yolo_min = int(args.imgsz * 0.5)
        yolo_max = int(args.imgsz * 1.5)
        yolo_min = ((yolo_min + 31) // 32) * 32
        yolo_max = ((yolo_max + 31) // 32) * 32
        print(f"Multi-scale Training: Enabled (YOLO internal: {yolo_min}-{yolo_max}px)")
    else:
        print(f"Multi-scale Training: Disabled")

    # Multi-scale validation info
    if args.val_scales:
        print(f"Multi-scale Validation: {args.val_scales}")
    else:
        print(f"Multi-scale Validation: Disabled")

    print("=" * 60)

    # Initialize model
    try:
        model = YOLO(args.model)
        print(f"Model loaded successfully: {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Detect task type
    task_type = detect_task_type(model)
    print(f"Detected task type: {task_type}")

    # Prepare training arguments
    train_args = {
        # Basic training parameters
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.imgsz,
        "lr0": args.lr,
        "patience": args.patience,

        # Output and logging
        "project": args.project,
        "name": args.name,
        "save_period": args.save_period,
        "plots": args.plots,

        # Hardware
        "workers": args.workers,
        "cache": args.cache,
        "amp": args.amp,

        # Augmentation parameters
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,

        # Advanced options
        "pretrained": args.pretrained,
        "freeze": args.freeze
    }

    # Multi-scale training configuration
    if args.multiscale:
        # WARNING: Ultralytics YOLO's multi_scale=True uses internal scaling (0.5x to 1.5x base size)
        # and does not respect custom min/max bounds. The parameters below are for display only.
        train_args["multi_scale"] = True

        # Calculate YOLO's actual range (hardcoded in Ultralytics)
        yolo_min = int(args.imgsz * 0.5)
        yolo_max = int(args.imgsz * 1.5)

        # Round to multiples of 32
        yolo_min = ((yolo_min + 31) // 32) * 32
        yolo_max = ((yolo_max + 31) // 32) * 32

        print(f"Multi-scale training enabled (YOLO internal): {yolo_min} - {yolo_max} pixels")
        print(f"  Note: YOLO uses fixed 0.5x-1.5x range, custom min/max parameters are ignored")

        if args.multiscale_min or args.multiscale_max:
            print(f"  Warning: --multiscale-min/max parameters have no effect with YOLO's built-in multi-scale")
            print(f"  Consider using multi-scale validation (--val-scales) for custom size testing")
    else:
        train_args["multi_scale"] = False

    # Add device if specified
    if args.device:
        train_args["device"] = args.device

    # Add resume if specified
    if args.resume:
        train_args["resume"] = args.resume
        print(f"Resuming training from: {args.resume}")

    # Setup validation scales
    val_scales = args.val_scales or []
    if val_scales:
        # Ensure all scales are multiples of 32
        val_scales = [((scale + 31) // 32) * 32 for scale in val_scales]

    # --- Setup for multi-scale validation and TensorBoard logging ---
    best_models = {}  # Track best model for each validation scale
    for scale in val_scales:
        best_models[scale] = {"score": -1.0, "epoch": -1}

    tb_writer = {"writer": None}
    sample_images = {"files": None, "tags": {}}  # Store consistent sample images for each scale

    def _pick_score(metrics, which="map"):
        """Extract score based on detected task type."""
        task = detect_task_type(metrics)

        if task == 'detect' and hasattr(metrics, "box"):
            return float(metrics.box.map if which == "map" else metrics.box.map50)
        elif task == 'segment' and hasattr(metrics, "seg"):
            return float(metrics.seg.map if which == "map" else metrics.seg.map50)
        elif task == 'pose' and hasattr(metrics, "pose"):
            return float(metrics.pose.map if which == "map" else metrics.pose.map50)

        # Fallback: try in order of preference
        for attr in ['box', 'seg', 'pose']:
            if hasattr(metrics, attr):
                metric_obj = getattr(metrics, attr)
                return float(metric_obj.map if which == "map" else metric_obj.map50)

        return None

    def _log_scalars_tb(writer, step, metrics, prefix):
        """Log metrics to TensorBoard based on actual task type."""
        if not writer:
            return

        task = detect_task_type(metrics)

        # Only log metrics relevant to the current task
        if task == 'detect' and hasattr(metrics, "box"):
            writer.add_scalar(f"{prefix}/mAP50-95", float(metrics.box.map), step)
            writer.add_scalar(f"{prefix}/mAP50", float(metrics.box.map50), step)
        elif task == 'segment':
            if hasattr(metrics, "box"):
                writer.add_scalar(f"{prefix}/mAP50-95(box)", float(metrics.box.map), step)
                writer.add_scalar(f"{prefix}/mAP50(box)", float(metrics.box.map50), step)
            if hasattr(metrics, "seg"):
                writer.add_scalar(f"{prefix}/mAP50-95(seg)", float(metrics.seg.map), step)
                writer.add_scalar(f"{prefix}/mAP50(seg)", float(metrics.seg.map50), step)
        elif task == 'pose':
            if hasattr(metrics, "box"):
                writer.add_scalar(f"{prefix}/mAP50-95(box)", float(metrics.box.map), step)
                writer.add_scalar(f"{prefix}/mAP50(box)", float(metrics.box.map50), step)
            if hasattr(metrics, "pose"):
                writer.add_scalar(f"{prefix}/mAP50-95(pose)", float(metrics.pose.map), step)
                writer.add_scalar(f"{prefix}/mAP50(pose)", float(metrics.pose.map50), step)

        writer.flush()

    def _open_tb_writer(save_dir):
        """Initialize TensorBoard writer."""
        if tb_writer["writer"] is None:
            try:
                tb_writer["writer"] = SummaryWriter(log_dir=str(save_dir))
                print(f"TensorBoard logging initialized: {save_dir}")
            except Exception as e:
                print(f"TensorBoard writer init failed: {e}")

    def _as_list(x):
        """Convert to list if not already."""
        if x is None:
            return []
        return x if isinstance(x, (list, tuple)) else [x]

    def _maybe_to_images_dir(p):
        """Convert labels directory path to images directory path."""
        parts = Path(p).parts
        if "labels" in parts:
            parts = list(parts)
            parts[parts.index("labels")] = "images"
            return str(Path(*parts))
        return p

    def _expand_path(p):
        """Expand path to list of image files."""
        # txt file with image paths?
        if p.endswith(".txt") and os.path.isfile(p):
            with open(p) as fh:
                return [ln.strip() for ln in fh if ln.strip()]

        # directory of images?
        if os.path.isdir(p):
            p_img = _maybe_to_images_dir(p)
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
            out = []
            for e in exts:
                out.extend(glob.glob(os.path.join(p_img, "**", e), recursive=True))
            return out

        # single file?
        if os.path.isfile(p):
            return [p]
        return []

    def _resolve_dataset_split_from_yaml(yaml_path, split_key):
        """Resolve dataset split paths from YAML configuration."""
        try:
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Error reading YAML file: {e}")
            return []

        root = cfg.get("path", None)

        # try common keys (val/test/train)
        entry = cfg.get(split_key) or cfg.get("val")
        if not entry:
            return []

        paths = []
        for item in _as_list(entry):
            # join with root if relative and root exists
            if root and not os.path.isabs(item):
                item = os.path.join(root, item)
            # if they pointed at a labels dir, hop to images sibling
            item = _maybe_to_images_dir(item)
            paths.extend(_expand_path(item))
        return [p for p in paths if os.path.isfile(p)]

    def _sample_val_images(trainer, k=None):
        """Sample validation images for prediction visualization."""
        if k is None:
            k = args.tb_samples

        # 1) try to pull from the active validator's dataset (if present)
        try:
            val_obj = getattr(trainer, "validator", None)
            ds = getattr(getattr(val_obj, "dataloader", None), "dataset", None)
            files = getattr(ds, "im_files", None) or getattr(ds, "imgs", None)
            if files:
                files = [f for f in files if isinstance(f, str) and os.path.isfile(f)]
                if files:
                    # Sort for consistency, then sample
                    files = sorted(files)
                    return files[:k] if len(files) >= k else files
        except Exception:
            pass

        # 2) resolve from data.yaml, respecting 'path:' and swapping labels->images
        try:
            files = _resolve_dataset_split_from_yaml(args.data, args.val_split)
            if files:
                # Sort for consistency, then sample
                files = sorted(files)
                return files[:k] if len(files) >= k else files
        except Exception:
            pass

        return []

    def _initialize_sample_images(trainer):
        """Initialize consistent sample images for TensorBoard visualization."""
        if sample_images["files"] is not None:
            return  # Already initialized

        # Use user-specified images if provided
        if args.tb_sample_images:
            files = []
            for img_path in args.tb_sample_images:
                if os.path.isfile(img_path):
                    files.append(img_path)
                else:
                    print(f"Warning: Specified sample image not found: {img_path}")
        else:
            # Auto-sample from validation set
            files = _sample_val_images(trainer, args.tb_samples)

        if files:
            sample_images["files"] = files
            # Create consistent tags for each scale
            for scale in val_scales:
                sample_images["tags"][scale] = []
                for i, file_path in enumerate(files):
                    filename = Path(file_path).stem
                    tag = f"pred@{scale}/sample_{i:02d}_{filename}"
                    sample_images["tags"][scale].append(tag)

            print(f"[TensorBoard] Using {len(files)} consistent sample images for visualization:")
            for i, file_path in enumerate(files):
                print(f"  {i + 1}. {Path(file_path).name}")
        else:
            print("[TensorBoard] No sample images found for visualization.")

    def _multi_scale_validation_and_log(trainer):
        """Perform multi-scale validation and log results."""
        if not val_scales or args.val_every < 1:
            return
        if (trainer.epoch + 1) % args.val_every:
            return

        print(f"\n[Epoch {trainer.epoch + 1}] Running multi-scale validation...")

        # Use the just-saved weights
        ckpt_path = trainer.last if getattr(trainer, "last", None) and trainer.last.exists() else trainer.best

        try:
            step = trainer.epoch + 1
            _open_tb_writer(trainer.save_dir)

            # Initialize sample images for TensorBoard
            _initialize_sample_images(trainer)

            # Validate at each scale
            for scale in val_scales:
                print(f"  Validating at {scale}px...")

                try:
                    y = YOLO(str(ckpt_path))

                    # Validation at current scale
                    metrics = y.val(
                        data=args.data,
                        imgsz=scale,
                        batch=args.batch_size,
                        split=args.val_split,
                        device=(args.device or None),
                        plots=False,
                        verbose=False
                    )

                    # Log task-aware scalars
                    _log_scalars_tb(tb_writer["writer"], step, metrics, prefix=f"val@{scale}")

                    # Score to track for best@scale
                    score = _pick_score(metrics, which=args.val_metric)

                    if score is not None:
                        print(f"    {args.val_metric}: {score:.5f}")

                        # Save "best at this scale" if improved
                        if score > best_models[scale]["score"]:
                            best_models[scale]["score"] = float(score)
                            best_models[scale]["epoch"] = step

                            # Explicit weights dir (same as original experiment weights)
                            weights_dir = trainer.save_dir / "weights"
                            try:
                                weights_dir.mkdir(parents=True, exist_ok=True)
                            except Exception:
                                pass

                            dst = weights_dir / f"best_img{scale}.pt"
                            try:
                                shutil.copy2(ckpt_path, dst)

                                # Optional: overwrite Ultralytics' best.pt with the best across all scales
                                if args.override_best:
                                    # Find the scale with the highest score
                                    best_scale = max(best_models.keys(),
                                                     key=lambda s: best_models[s]["score"])
                                    if scale == best_scale:
                                        shutil.copy2(dst, weights_dir / "best.pt")

                                # Write a summary file
                                with open(weights_dir / f"best_img{scale}.txt", "w") as f:
                                    f.write(
                                        f"epoch={step}\n"
                                        f"metric={args.val_metric}\n"
                                        f"score={best_models[scale]['score']:.6f}\n"
                                        f"task={task_type}\n"
                                        f"from={ckpt_path}\n"
                                    )

                                print(
                                    f"    → New best at {scale}px ({args.val_metric}={score:.5f}) saved to {dst.name}")
                            except Exception as e:
                                print(f"    → Saving {dst.name} failed: {e}")

                    # Sample predictions for TensorBoard
                    if sample_images["files"] and scale in sample_images["tags"]:
                        try:
                            print(f"    → Running sample predictions...")

                            preds = y.predict(
                                source=sample_images["files"],
                                imgsz=scale,
                                conf=0.25,
                                device=(args.device or None),
                                save=False,
                                stream=True,
                                verbose=False
                            )

                            for i, res in enumerate(preds):
                                try:
                                    # Render annotated image (returns BGR np.ndarray)
                                    im = res.plot()
                                    # Convert BGR -> RGB for TensorBoard
                                    im = im[..., ::-1]

                                    # Use consistent tag for TensorBoard slider
                                    tag = sample_images["tags"][scale][i] if i < len(
                                        sample_images["tags"][scale]) else f"pred@{scale}/sample_{i:02d}"

                                    if tb_writer["writer"]:
                                        tb_writer["writer"].add_image(tag, im, step, dataformats="HWC")

                                    # Brief console summary
                                    n_detections = 0
                                    if getattr(res, "boxes", None) is not None:
                                        n_detections = len(res.boxes) if res.boxes is not None else 0
                                    # Don't print for every image to reduce clutter

                                except Exception as e:
                                    print(f"    → Sample prediction logging failed for image {i}: {e}")

                        except Exception as e:
                            print(f"    → Sample prediction failed: {e}")

                except Exception as e:
                    print(f"  → Validation at {scale}px failed: {e}")

            # Flush TensorBoard writer
            if tb_writer["writer"]:
                tb_writer["writer"].flush()

            # Print summary of best models
            print(f"  Multi-scale validation summary:")
            for scale in val_scales:
                if best_models[scale]["score"] > -1:
                    print(
                        f"    {scale}px: best {args.val_metric}={best_models[scale]['score']:.5f} (epoch {best_models[scale]['epoch']})")

        except Exception as e:
            print(f"Multi-scale validation failed: {e}")

    # Add callback for multi-scale validation after each epoch
    model.add_callback("on_train_epoch_end", _multi_scale_validation_and_log)

    print("\nStarting training...")
    if args.multiscale:
        print("Multi-scale training enabled - image sizes will vary during training")
    if val_scales:
        print(f"Multi-scale validation will run at {val_scales}px every {args.val_every} epoch(s)")
        print(f"TensorBoard will visualize {args.tb_samples} sample images with slider functionality")
    print("TensorBoard logs will be saved alongside training results")
    print("You can monitor training with: tensorboard --logdir <project>/<name>/tensorboard")
    print("-" * 60)

    try:
        # Start training
        results = model.train(**train_args)

        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print(f"Best weights saved to: {model.trainer.best}")
        print(f"Last weights saved to: {model.trainer.last}")

        # Print summary of all best models
        if val_scales and any(best_models[scale]["score"] > -1 for scale in val_scales):
            print("\nBest models at different scales:")
            weights_dir = Path(model.trainer.save_dir) / "weights"
            for scale in val_scales:
                if best_models[scale]["score"] > -1:
                    best_path = weights_dir / f"best_img{scale}.pt"
                    if best_path.exists():
                        print(f"  {scale}px: {args.val_metric}={best_models[scale]['score']:.5f} → {best_path}")

        print("=" * 60)

        # Close TensorBoard writer
        if tb_writer["writer"]:
            tb_writer["writer"].close()

        return 0

    except Exception as e:
        print(f"\nError during training: {e}")
        if tb_writer["writer"]:
            tb_writer["writer"].close()
        return 1


if __name__ == "__main__":
    exit(main())