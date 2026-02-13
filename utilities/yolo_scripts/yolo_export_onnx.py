#!/usr/bin/env python3
"""
YOLO11 ONNX Export Guide and Utilities
This script provides various methods to export YOLO11 models to ONNX format
with different optimization options including float16, dynamic shapes, etc.
"""

import torch
from ultralytics import YOLO
import onnx
import onnxruntime
import numpy as np
import time
import argparse
from pathlib import Path


def export_yolo11_to_onnx(
        model_path: str,
        output_path: str = None,
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        opset: int = 12,
        simplify: bool = True,
        optimize: bool = True,
        batch_size: int = 1
):
    """
    Export YOLO11 model to ONNX format with various optimization options.

    Args:
        model_path (str): Path to the YOLO11 model (.pt file)
        output_path (str): Output path for ONNX model (optional)
        imgsz (int): Image size for export (default: 640)
        half (bool): Export in FP16 precision (default: False)
        dynamic (bool): Enable dynamic input shapes (default: False)
        opset (int): ONNX opset version (default: 12)
        simplify (bool): Simplify ONNX model (default: True)
        optimize (bool): Optimize ONNX model (default: True)
        batch_size (int): Batch size for export (default: 1)

    Returns:
        str: Path to exported ONNX model
    """
    print(f"Loading YOLO11 model from: {model_path}")

    # Load the model
    model = YOLO(model_path)

    # Generate output path if not provided
    if output_path is None:
        model_name = Path(model_path).stem
        precision = "fp16" if half else "fp32"
        dynamic_suffix = "_dynamic" if dynamic else ""
        output_path = f"{model_name}_{precision}_bs{batch_size}{dynamic_suffix}.onnx"

    print(f"Exporting to: {output_path}")
    print(f"Configuration:")
    print(f"  - Image size: {imgsz}")
    print(f"  - Precision: {'FP16' if half else 'FP32'}")
    print(f"  - Dynamic shapes: {dynamic}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - ONNX opset: {opset}")
    print(f"  - Simplify: {simplify}")
    print(f"  - Optimize: {optimize}")

    # Export to ONNX
    try:
        model.export(
            format='onnx',
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            opset=opset,
            simplify=simplify,
            optimize=optimize,
            batch=batch_size if not dynamic else 1
        )

        # The export method automatically generates the filename
        # We need to find the generated file and optionally rename it
        model_dir = Path(model_path).parent
        generated_files = list(model_dir.glob(f"{Path(model_path).stem}*.onnx"))

        if generated_files:
            generated_file = max(generated_files, key=lambda x: x.stat().st_mtime)

            if str(generated_file) != output_path:
                generated_file.rename(output_path)

            print(f"✓ Export successful: {output_path}")
            return output_path
        else:
            print("✗ Export failed: No ONNX file generated")
            return None

    except Exception as e:
        print(f"✗ Export failed: {e}")
        return None


def validate_onnx_model(onnx_path: str, original_model_path: str = None, test_image_size: tuple = (640, 640)):
    """
    Validate the exported ONNX model by running inference and comparing with original model.

    Args:
        onnx_path (str): Path to ONNX model
        original_model_path (str): Path to original PyTorch model (optional)
        test_image_size (tuple): Size of test image (width, height)
    """
    print(f"\n--- Validating ONNX model: {onnx_path} ---")

    try:
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model structure is valid")

        # Get model info
        print(f"✓ ONNX opset version: {onnx_model.opset_import[0].version}")

        # Check input/output shapes
        for input_tensor in onnx_model.graph.input:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"✓ Input shape: {input_tensor.name} -> {shape}")

        for output_tensor in onnx_model.graph.output:
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"✓ Output shape: {output_tensor.name} -> {shape}")

        # Test inference with ONNXRuntime
        print("\n--- Testing ONNX Runtime inference ---")

        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = onnxruntime.InferenceSession(onnx_path, providers=providers)

        print(f"✓ Using providers: {session.get_providers()}")

        # Create dummy input
        input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
        if any(s <= 0 for s in input_shape):  # Handle dynamic shapes
            input_shape = [1, 3, test_image_size[1], test_image_size[0]]

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        print(f"✓ Test input shape: {dummy_input.shape}")

        # Run inference
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        start_time = time.time()
        outputs = session.run(output_names, {input_name: dummy_input})
        inference_time = time.time() - start_time

        print(f"✓ ONNX inference successful in {inference_time:.4f}s")
        for i, output in enumerate(outputs):
            print(f"  Output {i}: {output.shape}")

        # Compare with original model if provided
        if original_model_path:
            print(f"\n--- Comparing with original model ---")
            try:
                original_model = YOLO(original_model_path)

                # Convert dummy input to the format expected by YOLO
                dummy_image = (dummy_input[0].transpose(1, 2, 0) * 255).astype(np.uint8)

                start_time = time.time()
                results = original_model(dummy_image, verbose=False)
                pytorch_time = time.time() - start_time

                print(f"✓ PyTorch inference in {pytorch_time:.4f}s")
                print(f"✓ ONNX speedup: {pytorch_time / inference_time:.2f}x")

            except Exception as e:
                print(f"⚠ Could not compare with original model: {e}")

        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


def benchmark_model(model_path: str, num_runs: int = 100, warmup_runs: int = 10):
    """
    Benchmark ONNX model performance.

    Args:
        model_path (str): Path to ONNX model
        num_runs (int): Number of benchmark runs
        warmup_runs (int): Number of warmup runs
    """
    print(f"\n--- Benchmarking model: {model_path} ---")

    try:
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)

        # Get input shape
        input_shape = session.get_inputs()[0].shape
        if any(isinstance(s, str) for s in input_shape):  # Dynamic shapes
            input_shape = [1, 3, 640, 640]

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        # Warmup
        print(f"Warming up with {warmup_runs} runs...")
        for _ in range(warmup_runs):
            session.run(output_names, {input_name: dummy_input})

        # Benchmark
        print(f"Running {num_runs} benchmark iterations...")
        times = []

        for i in range(num_runs):
            start_time = time.time()
            session.run(output_names, {input_name: dummy_input})
            times.append(time.time() - start_time)

            if (i + 1) % 20 == 0:
                print(f"  Completed {i + 1}/{num_runs} runs")

        # Statistics
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1.0 / mean_time

        print(f"\n--- Benchmark Results ---")
        print(f"Mean inference time: {mean_time:.4f}s (±{std_time:.4f}s)")
        print(f"Min inference time:  {min_time:.4f}s")
        print(f"Max inference time:  {max_time:.4f}s")
        print(f"Average FPS:         {fps:.1f}")
        print(f"Provider:            {session.get_providers()[0]}")

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")


def optimize_onnx_model(input_path: str, output_path: str = None):
    """
    Apply additional optimizations to ONNX model using ONNXRuntime tools.

    Args:
        input_path (str): Path to input ONNX model
        output_path (str): Path to optimized output model
    """
    try:
        from onnxruntime.tools import optimizer

        if output_path is None:
            output_path = input_path.replace('.onnx', '_optimized.onnx')

        print(f"Optimizing ONNX model: {input_path} -> {output_path}")

        # Apply optimizations
        optimized_model = optimizer.optimize_model(
            input_path,
            model_type='bert',  # Use 'bert' for general optimizations
            opt_level=99  # Maximum optimization level
        )

        optimized_model.save_model_to_file(output_path)
        print(f"✓ Optimized model saved to: {output_path}")

        return output_path

    except ImportError:
        print("⚠ ONNXRuntime optimization tools not available")
        print("  Install with: pip install onnxruntime-tools")
        return None
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLO11 ONNX Export and Optimization Tool")

    # Required arguments
    parser.add_argument('model', type=str, help='Path to YOLO11 model (.pt file)')

    # Export options
    parser.add_argument('--output', '-o', type=str, help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for export (default: 640)')
    parser.add_argument('--half', action='store_true', help='Export in FP16 precision')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic input shapes')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version (default: 12)')
    parser.add_argument('--no-simplify', action='store_true', help='Disable ONNX simplification')
    parser.add_argument('--no-optimize', action='store_true', help='Disable ONNX optimization')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for export (default: 1)')

    # Validation and benchmarking
    parser.add_argument('--validate', action='store_true', help='Validate exported ONNX model')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark ONNX model performance')
    parser.add_argument('--num-runs', type=int, default=100, help='Number of benchmark runs (default: 100)')
    parser.add_argument('--warmup-runs', type=int, default=10, help='Number of warmup runs (default: 10)')

    # Additional optimization
    parser.add_argument('--extra-optimize', action='store_true',
                        help='Apply additional ONNXRuntime optimizations')

    args = parser.parse_args()

    # Export model
    print("=== YOLO11 ONNX Export Tool ===\n")

    onnx_path = export_yolo11_to_onnx(
        model_path=args.model,
        output_path=args.output,
        imgsz=args.imgsz,
        half=args.half,
        dynamic=args.dynamic,
        opset=args.opset,
        simplify=not args.no_simplify,
        optimize=not args.no_optimize,
        batch_size=args.batch_size
    )

    if onnx_path is None:
        print("Export failed. Exiting.")
        return

    # Apply extra optimizations
    if args.extra_optimize:
        onnx_path = optimize_onnx_model(onnx_path) or onnx_path

    # Validate model
    if args.validate:
        validate_onnx_model(onnx_path, args.model)

    # Benchmark model
    if args.benchmark:
        benchmark_model(onnx_path, args.num_runs, args.warmup_runs)

    print(f"\n=== Export Complete ===")
    print(f"ONNX model: {onnx_path}")


if __name__ == '__main__':
    main()


# Example usage functions
def export_examples():
    """Example exports for different use cases."""

    model_path = "yolo11n-pose.pt"

    print("=== YOLO11 ONNX Export Examples ===\n")

    # 1. Standard FP32 export
    print("1. Standard FP32 export:")
    export_yolo11_to_onnx(
        model_path=model_path,
        output_path="yolo11n-pose-fp32.onnx",
        imgsz=640,
        half=False,
        dynamic=False
    )

    # 2. FP16 export for better performance
    print("\n2. FP16 export:")
    export_yolo11_to_onnx(
        model_path=model_path,
        output_path="yolo11n-pose-fp16.onnx",
        imgsz=640,
        half=True,
        dynamic=False
    )

    # 3. Dynamic shapes for flexible input sizes
    print("\n3. Dynamic shapes export:")
    export_yolo11_to_onnx(
        model_path=model_path,
        output_path="yolo11n-pose-dynamic.onnx",
        imgsz=640,
        half=False,
        dynamic=True
    )

    # 4. High-performance FP16 with dynamic shapes
    print("\n4. High-performance FP16 with dynamic shapes:")
    export_yolo11_to_onnx(
        model_path=model_path,
        output_path="yolo11n-pose-fp16-dynamic.onnx",
        imgsz=640,
        half=True,
        dynamic=True
    )

    # 5. Batch processing export
    print("\n5. Batch processing export:")
    export_yolo11_to_onnx(
        model_path=model_path,
        output_path="yolo11n-pose-batch4.onnx",
        imgsz=640,
        half=False,
        dynamic=False,
        batch_size=4
    )


# Command line examples as comments:
"""
# Basic export
python yolo11_onnx_export.py yolo11n-pose.pt

# FP16 export with validation and benchmarking
python yolo11_onnx_export.py yolo11n-pose.pt --half --validate --benchmark

# Dynamic shapes with extra optimization
python yolo11_onnx_export.py yolo11n-pose.pt --dynamic --extra-optimize

# Custom output path with specific settings
python yolo11_onnx_export.py yolo11n-pose.pt -o custom_model.onnx --imgsz 832 --half --dynamic

# Batch export with validation
python yolo11_onnx_export.py yolo11n-pose.pt --batch-size 4 --validate --benchmark --num-runs 50
"""