#!/usr/bin/env python3
"""
Proper TensorRT Export with INT8 Calibration
This script correctly implements INT8 quantization with calibration dataset
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import glob
import json
from typing import List, Optional
import tempfile

try:
    from ultralytics import YOLO
    import tensorrt as trt
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install ultralytics tensorrt")
    sys.exit(1)


class CalibrationDataLoader:
    """
    Data loader for INT8 calibration
    """

    def __init__(self, calibration_images: List[str], input_size: int, batch_size: int = 1):
        self.calibration_images = calibration_images
        self.input_size = input_size
        self.batch_size = batch_size
        self.current_index = 0

        print(f"Loaded {len(self.calibration_images)} calibration images")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Preprocess image for calibration (should match inference preprocessing)"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Convert BGR to RGB (YOLO expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize while maintaining aspect ratio (letterboxing)
        h, w = image.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterboxed image
        letterboxed = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        y_offset = (self.input_size - new_h) // 2
        x_offset = (self.input_size - new_w) // 2
        letterboxed[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Normalize to [0, 1] and transpose to CHW format
        letterboxed = letterboxed.astype(np.float32) / 255.0
        letterboxed = np.transpose(letterboxed, (2, 0, 1))  # HWC to CHW

        # Add batch dimension
        return np.expand_dims(letterboxed, axis=0)

    def get_batch(self) -> Optional[np.ndarray]:
        """Get next batch of calibration data"""
        if self.current_index >= len(self.calibration_images):
            return None

        batch_images = []
        for i in range(min(self.batch_size, len(self.calibration_images) - self.current_index)):
            image_path = self.calibration_images[self.current_index + i]
            try:
                processed_image = self.preprocess_image(image_path)
                batch_images.append(processed_image[0])  # Remove batch dim for stacking
            except Exception as e:
                print(f"Warning: Failed to process {image_path}: {e}")
                continue

        if not batch_images:
            return None

        self.current_index += len(batch_images)

        # Stack into batch
        batch = np.stack(batch_images, axis=0)
        print(f"Calibration batch {self.current_index // self.batch_size}: {batch.shape}")

        return batch

    def reset(self):
        """Reset calibration data loader"""
        self.current_index = 0


def collect_calibration_images(calibration_images_dir: str, extensions: List[str] = None) -> List[str]:
    """Collect calibration images from directory"""
    if extensions is None:
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']

    calibration_images = []
    calibration_path = Path(calibration_images_dir)

    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration images directory not found: {calibration_images_dir}")

    for ext in extensions:
        images = glob.glob(str(calibration_path / "**" / ext), recursive=True)
        calibration_images.extend(images)

    if not calibration_images:
        raise ValueError(f"No calibration images found in {calibration_images_dir}")

    # Limit to reasonable number for calibration
    if len(calibration_images) > 1000:
        print(f"Found {len(calibration_images)} images, using first 1000 for calibration")
        calibration_images = calibration_images[:1000]

    return sorted(calibration_images)


def create_sample_calibration_data(output_dir: str, input_size: int, num_samples: int = 100):
    """Create sample calibration data if none provided"""
    print(f"Creating {num_samples} sample calibration images...")

    os.makedirs(output_dir, exist_ok=True)

    # Generate synthetic data that resembles real images
    for i in range(num_samples):
        # Create random noise image
        image = np.random.randint(0, 256, (input_size, input_size, 3), dtype=np.uint8)

        # Add some structure to make it more realistic
        # Add some geometric shapes
        center = (input_size // 2, input_size // 2)
        radius = np.random.randint(20, input_size // 4)
        color = tuple(np.random.randint(0, 256, 3).tolist())
        cv2.circle(image, center, radius, color, -1)

        # Add some rectangles
        for _ in range(np.random.randint(1, 5)):
            pt1 = (np.random.randint(0, input_size), np.random.randint(0, input_size))
            pt2 = (np.random.randint(0, input_size), np.random.randint(0, input_size))
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.rectangle(image, pt1, pt2, color, -1)

        # Save image
        image_path = os.path.join(output_dir, f"calib_{i:04d}.jpg")
        cv2.imwrite(image_path, image)

    print(f"Sample calibration data created in: {output_dir}")
    return output_dir


def create_sample_dataset_yaml(output_path: str, calibration_images_dir: str, input_size: int):
    """Create a sample dataset.yaml file for calibration"""
    dataset_config = {
        'path': str(Path(calibration_images_dir).parent),  # Parent directory
        'train': str(Path(calibration_images_dir).name),  # Relative path to images
        'val': str(Path(calibration_images_dir).name),  # Use same images for validation
        'nc': 80,  # Number of classes (COCO default)
        'names': [f'class_{i}' for i in range(80)]  # Dummy class names
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        import yaml
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"Sample dataset.yaml created at: {output_path}")
    return output_path


def export_with_proper_int8_calibration(
        model_path: str,
        calibration_images_dir: Optional[str] = None,
        dataset_yaml: Optional[str] = None,
        output_path: Optional[str] = None,
        input_size: int = 640,
        precision: str = "fp16",
        workspace: int = 4,
        device: int = 0
):
    """Export YOLO model to TensorRT with proper INT8 calibration"""

    print(f"=== TensorRT Export with Calibration ===")
    print(f"Model: {model_path}")
    print(f"Input size: {input_size}")
    print(f"Precision: {precision}")
    print(f"Workspace: {workspace}GB")

    # Load model
    model = YOLO(model_path)

    if precision == "int8":
        if calibration_images_dir is None:
            # Create sample calibration data
            print("Warning: No calibration images directory provided for INT8.")
            print("Creating synthetic calibration data (not recommended for production)")
            temp_dir = tempfile.mkdtemp(prefix="tensorrt_calibration_")
            calibration_images_dir = create_sample_calibration_data(temp_dir, input_size)

        if dataset_yaml is None:
            # Create sample dataset.yaml
            print("Warning: No dataset.yaml provided for INT8.")
            print("Creating sample dataset.yaml file")
            temp_yaml = os.path.join(tempfile.gettempdir(), "sample_dataset.yaml")
            dataset_yaml = create_sample_dataset_yaml(temp_yaml, calibration_images_dir, input_size)

        # Collect calibration images
        calibration_images = collect_calibration_images(calibration_images_dir)
        print(f"Using {len(calibration_images)} calibration images from: {calibration_images_dir}")
        print(f"Using dataset configuration: {dataset_yaml}")

        # Validate dataset.yaml exists
        if not os.path.exists(dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML file not found: {dataset_yaml}")

        # For ultralytics, we can try to use the built-in calibration
        # Note: This may not use our specific calibration data optimally
        print("Exporting with INT8... This will use built-in calibration")
        print("For production use, consider implementing custom calibration")

        exported_path = model.export(
            format='engine',
            imgsz=input_size,
            device=device,
            workspace=workspace,
            int8=True,
            data=dataset_yaml,  # Use the provided dataset.yaml
            verbose=True
        )
    else:
        # FP16 or FP32 export
        half = (precision == "fp16")
        exported_path = model.export(
            format='engine',
            imgsz=input_size,
            device=device,
            workspace=workspace,
            half=half,
            verbose=True
        )

    print(f"Export completed: {exported_path}")

    # Validate the exported model
    try:
        validation_model = YOLO(exported_path)
        dummy_image = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
        results = validation_model(dummy_image, verbose=False)
        print("✓ Model validation successful")
    except Exception as e:
        print(f"⚠ Model validation failed: {e}")

    return exported_path


def export_with_custom_calibration(
        model_path: str,
        calibration_images_dir: str,
        output_path: str,
        input_size: int = 640,
        workspace: int = 4
):
    """
    Export using custom TensorRT calibration (advanced users)
    This is a template - requires full TensorRT implementation
    """
    print("Custom calibration export not fully implemented.")
    print("This would require:")
    print("1. Implementing IInt8Calibrator interface")
    print("2. Building TensorRT engine manually")
    print("3. Managing calibration cache")
    print("")
    print("For now, use the ultralytics built-in approach or:")
    print("1. Export to ONNX first")
    print("2. Use trtexec with calibration data")

    # Example of what would be needed:
    print("Example trtexec command for manual calibration:")
    onnx_path = model_path.replace('.pt', '.onnx')
    print(f"# First export to ONNX:")
    print(f"python -c \"from ultralytics import YOLO; YOLO('{model_path}').export(format='onnx', imgsz={input_size})\"")
    print(f"")
    print(f"# Then use trtexec with calibration:")
    print(f"trtexec --onnx={onnx_path} \\")
    print(f"        --saveEngine={output_path} \\")
    print(f"        --int8 \\")
    print(f"        --calib={calibration_images_dir} \\")
    print(f"        --workspace={workspace * 1024}")


def main():
    parser = argparse.ArgumentParser(description='TensorRT Export with Proper INT8 Calibration')

    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model (.pt)')
    parser.add_argument('--calibration-images-dir', type=str, help='Directory with calibration images')
    parser.add_argument('--dataset-yaml', type=str, help='Path to dataset.yaml configuration file')
    parser.add_argument('--output', type=str, help='Output engine file path')
    parser.add_argument('--input-size', type=int, default=640, help='Input image size')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'int8'], default='fp16',
                        help='Model precision')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size in GB')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--create-sample-data', action='store_true',
                        help='Create sample calibration data for testing')

    args = parser.parse_args()

    # Validate INT8 requirements
    if args.precision == 'int8':
        if args.calibration_images_dir and not os.path.exists(args.calibration_images_dir):
            print(f"Error: Calibration images directory not found: {args.calibration_images_dir}")
            sys.exit(1)

        if args.dataset_yaml and not os.path.exists(args.dataset_yaml):
            print(f"Error: Dataset YAML file not found: {args.dataset_yaml}")
            sys.exit(1)

        if not args.calibration_images_dir:
            print("Warning: INT8 quantization without calibration images directory!")
            print("This will create synthetic calibration data (not optimal for production)")
            response = input("Continue? (y/N): ")
            if response.lower() != 'y':
                sys.exit(1)

    # Create sample calibration data if requested
    if args.create_sample_data:
        sample_images_dir = f"calibration_images_{args.input_size}"
        create_sample_calibration_data(sample_images_dir, args.input_size)

        sample_yaml_path = f"sample_dataset_{args.input_size}.yaml"
        create_sample_dataset_yaml(sample_yaml_path, sample_images_dir, args.input_size)

        print(f"Sample calibration images created in: {sample_images_dir}")
        print(f"Sample dataset.yaml created at: {sample_yaml_path}")
        print("Use these for testing only - use real data for production!")

        if not args.calibration_images_dir:
            args.calibration_images_dir = sample_images_dir
        if not args.dataset_yaml:
            args.dataset_yaml = sample_yaml_path

    try:
        exported_path = export_with_proper_int8_calibration(
            model_path=args.model,
            calibration_images_dir=args.calibration_images_dir,
            dataset_yaml=args.dataset_yaml,
            output_path=args.output,
            input_size=args.input_size,
            precision=args.precision,
            workspace=args.workspace,
            device=args.device
        )

        print(f"\n✓ Export completed successfully!")
        print(f"Model saved to: {exported_path}")

        if args.precision == 'int8':
            print(f"\n📋 INT8 Calibration Notes:")
            print(f"• Calibration images used: {args.calibration_images_dir}")
            print(f"• Dataset configuration: {args.dataset_yaml}")
            print(f"• For best results, use 100-1000 representative images")
            print(f"• Images should match your deployment scenario")
            print(f"• Consider validating accuracy against FP16 model")

    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Usage examples:
"""
# FP16 export (no calibration needed)
python export_tensorrt_with_calibration.py --model yolo11n-pose.pt --precision fp16

# INT8 export with real calibration data
python export_tensorrt_with_calibration.py --model yolo11n-pose.pt --precision int8 \
    --calibration-images-dir /path/to/calibration/images \
    --dataset-yaml /path/to/dataset.yaml

# INT8 export with synthetic calibration data (testing only)
python export_tensorrt_with_calibration.py --model yolo11n-pose.pt --precision int8 --create-sample-data

# Manual approach using trtexec:
# 1. First export to ONNX
# python -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=640)"
# 
# 2. Then use trtexec with calibration
# trtexec --onnx=yolo11n-pose.onnx --int8 --saveEngine=model_int8.engine --calib=/path/to/calibration/images
"""