#!/usr/bin/env python3
"""
Improved Model Performance and Speed Benchmark Analyzer
Loads all models in a folder, extracts model architecture details,
exports to ONNX and TensorRT with comprehensive naming scheme.
Supports both trtexec (default) and Python TensorRT as fallback.
"""

import argparse
import os
import sys
import json
import time
import random
import re
import gc
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from PIL import Image
    import torch
    import yaml
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install ultralytics matplotlib seaborn pandas pillow torch")
    sys.exit(1)


class TensorRTConverter:
    """Handles TensorRT conversion using system trtexec or Python TensorRT (FP32/FP16 only)"""

    def __init__(self, trtexec_path: str = "/usr/src/tensorrt/bin/trtexec",
                 use_python_trt: bool = False, device: int = 0, verbose: bool = True):
        self.trtexec_path = trtexec_path
        self.use_python_trt = use_python_trt
        self.device = device
        self.verbose = verbose
        self.trtexec_available = False
        self.python_trt_available = False

        self._check_availability()

    def _check_availability(self):
        """Check availability of TensorRT conversion methods"""
        # Check trtexec
        if not self.use_python_trt:
            if shutil.which(self.trtexec_path) or os.path.exists(self.trtexec_path):
                self.trtexec_available = True
                print(f"✓ Found trtexec at: {self.trtexec_path}")
            else:
                print(f"⚠️ trtexec not found at {self.trtexec_path}")

        # Check Python TensorRT
        try:
            import tensorrt as trt
            self.python_trt_available = True
            print(f"✓ Python TensorRT available (version: {trt.__version__})")
        except ImportError:
            print(f"⚠️ Python TensorRT not available")

        # Determine which method to use
        if not self.use_python_trt and self.trtexec_available:
            print("Using trtexec for TensorRT conversion")
        elif self.python_trt_available:
            print("Using Python TensorRT for conversion")
            self.use_python_trt = True
        else:
            raise RuntimeError("No TensorRT conversion method available")

    def convert_onnx_to_engine(self, onnx_path: str, engine_path: Optional[str] = None,
                               precision: str = "fp16", max_workspace_size: int = 1024) -> str:
        """Convert ONNX model to TensorRT engine (FP32/FP16 only)"""
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

        # Validate precision
        if precision not in ["fp32", "fp16"]:
            print(f"⚠️ Unsupported precision '{precision}', using FP16 instead")
            precision = "fp16"

        # Generate engine path if not provided
        if engine_path is None:
            engine_path = onnx_path.replace('.onnx', f'_{precision}.engine')

        print(f"🔄 Converting ONNX to TensorRT engine:")
        print(f"   Input:  {onnx_path}")
        print(f"   Output: {engine_path}")
        print(f"   Precision: {precision}")
        print(f"   Method: {'trtexec' if not self.use_python_trt else 'Python TensorRT'}")

        # Use the explicitly chosen method
        if not self.use_python_trt:
            if not self.trtexec_available:
                raise RuntimeError(f"trtexec not available at {self.trtexec_path}")
            return self._convert_with_trtexec(onnx_path, engine_path, precision, max_workspace_size)
        else:
            if not self.python_trt_available:
                raise RuntimeError("Python TensorRT not available")
            return self._convert_with_python_trt(onnx_path, engine_path, precision, max_workspace_size)

    def _convert_with_trtexec(self, onnx_path: str, engine_path: str, precision: str,
                              max_workspace_size: int) -> str:
        """Convert using system trtexec (FP32/FP16 only)"""
        cmd = [
            self.trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
        ]

        # Handle workspace parameter based on trtexec version
        # Newer versions (10.x+) use --memPoolSize, older versions use --workspace
        cmd.append(f"--memPoolSize=workspace:{max_workspace_size}M")

        if precision == "fp16":
            cmd.append("--fp16")
        # fp32 is default, no flag needed

        # Add device specification
        cmd.append(f"--device={self.device}")

        if self.verbose:
            cmd.append("--verbose")

        # Prepare environment - inherit current environment completely
        env = os.environ.copy()

        # Ensure CUDA is visible
        env['CUDA_VISIBLE_DEVICES'] = str(self.device)

        # Ensure CUDA paths are in PATH
        cuda_paths = ['/usr/local/cuda/bin', '/usr/local/cuda-12/bin', '/usr/local/cuda-11/bin']
        current_path = env.get('PATH', '')
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path) and cuda_path not in current_path:
                env['PATH'] = f"{cuda_path}:{current_path}"

        # Ensure CUDA libraries are in LD_LIBRARY_PATH
        cuda_lib_paths = ['/usr/local/cuda/lib64', '/usr/local/cuda-12/lib64', '/usr/local/cuda-11/lib64',
                          '/usr/lib/x86_64-linux-gnu']
        current_ld_path = env.get('LD_LIBRARY_PATH', '')
        for cuda_lib_path in cuda_lib_paths:
            if os.path.exists(cuda_lib_path) and cuda_lib_path not in current_ld_path:
                if current_ld_path:
                    env['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{current_ld_path}"
                else:
                    env['LD_LIBRARY_PATH'] = cuda_lib_path

        print(f"   Executing: {' '.join(cmd)}")
        print(f"   CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
        print(f"   PATH includes CUDA: {any('cuda' in path for path in env['PATH'].split(':'))}")
        start_time = time.time()

        try:
            # Clear GPU memory before running trtexec
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                print(f"   Cleared GPU memory before trtexec")

            print(f"   Starting TensorRT conversion...")
            print(f"   " + "=" * 60)

            # Run with shell=True to inherit full environment like terminal
            result = subprocess.run(
                ' '.join(cmd),  # Join command as single string for shell execution
                env=env,
                shell=True,  # This is key - use shell like terminal
                text=True,
                capture_output=False  # Let output go directly to console
            )
            conversion_time = time.time() - start_time

            print(f"   " + "=" * 60)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, "", "")

            print(f"✅ TensorRT conversion completed in {conversion_time:.1f}s")

            if not os.path.exists(engine_path):
                raise RuntimeError("Engine file was not created")

            engine_size = os.path.getsize(engine_path) / (1024 * 1024)
            print(f"   Engine file size: {engine_size:.1f} MB")

            return engine_path

        except subprocess.CalledProcessError as e:
            # Try with legacy --workspace parameter for older trtexec versions
            if "--memPoolSize" in ' '.join(cmd):
                print(f"     Trying legacy --workspace parameter...")
                cmd_legacy = [
                    self.trtexec_path,
                    f"--onnx={onnx_path}",
                    f"--saveEngine={engine_path}",
                    f"--workspace={max_workspace_size}",
                    f"--device={self.device}"
                ]

                if precision == "fp16":
                    cmd_legacy.append("--fp16")

                if self.verbose:
                    cmd_legacy.append("--verbose")

                print(f"   Executing (legacy): {' '.join(cmd_legacy)}")
                print(f"   " + "=" * 60)

                result_legacy = subprocess.run(
                    ' '.join(cmd_legacy),
                    env=env,
                    shell=True,
                    text=True,
                    capture_output=False
                )
                conversion_time = time.time() - start_time

                print(f"   " + "=" * 60)

                if result_legacy.returncode != 0:
                    raise subprocess.CalledProcessError(result_legacy.returncode, cmd_legacy, "", "")

                print(f"✅ TensorRT conversion completed in {conversion_time:.1f}s")

                if not os.path.exists(engine_path):
                    raise RuntimeError("Engine file was not created")

                engine_size = os.path.getsize(engine_path) / (1024 * 1024)
                print(f"   Engine file size: {engine_size:.1f} MB")

                return engine_path
            else:
                print(f"❌ TensorRT conversion failed:")
                print(f"   Exit code: {result.returncode}")
                print(f"   Command: {' '.join(cmd)}")
                print(f"   Environment check:")
                print(f"     CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
                print(f"     PATH: {env['PATH'][:200]}...")
                print(f"     LD_LIBRARY_PATH: {env.get('LD_LIBRARY_PATH', 'not set')[:200]}...")
                raise RuntimeError(f"TensorRT conversion failed for {onnx_path}")

        except Exception as e:
            print(f"❌ TensorRT conversion error: {e}")
            raise

    def _convert_with_python_trt(self, onnx_path: str, engine_path: str, precision: str,
                                 max_workspace_size: int, original_pt_path: str = None) -> str:
        """Convert using YOLO ultralytics export instead of manual TensorRT (FP32/FP16 only)"""

        # Determine the original PT file path
        if original_pt_path and os.path.exists(original_pt_path):
            pt_path = original_pt_path
        else:
            # Try to derive the PT path from the ONNX path
            pt_path = onnx_path.replace('.onnx', '.pt')
            if not os.path.exists(pt_path):
                raise FileNotFoundError(f"Original PyTorch model not found. "
                                        f"Tried: {original_pt_path or 'None'} and {pt_path}. "
                                        f"Ultralytics export requires the original .pt file.")

        print(f"   Using YOLO ultralytics export...")
        print(f"   Loading model from: {pt_path}")

        try:
            # Load the original PyTorch model
            model = YOLO(pt_path)

            # Extract input size from the engine path or use default
            # Try to extract from filename patterns like 'img640'
            import re
            size_match = re.search(r'img(\d+)', os.path.basename(engine_path))
            input_size = int(size_match.group(1)) if size_match else 640

            print(f"   Input size: {input_size}")
            print(f"   Precision: {precision}")
            print(f"   Workspace: {max_workspace_size}MB")

            # Set up export parameters
            export_kwargs = {
                'format': 'engine',
                'imgsz': input_size,
                'device': self.device,
                'workspace': max_workspace_size / 1024,  # Convert MB to GB
                'verbose': self.verbose
            }

            # Set precision-specific parameters
            if precision == "fp16":
                export_kwargs['half'] = True
            elif precision == "fp32":
                export_kwargs['half'] = False
            else:
                print(f"   Warning: Precision '{precision}' not supported with ultralytics export")
                print(f"   Supported precisions: fp16, fp32")
                print(f"   Using fp16 instead")
                export_kwargs['half'] = True

            print(f"   Starting ultralytics TensorRT export...")
            start_time = time.time()

            # Export using ultralytics
            exported_path = model.export(**export_kwargs)

            conversion_time = time.time() - start_time

            # Move the exported file to the desired location if it's different
            if exported_path != engine_path:
                if os.path.exists(engine_path):
                    os.remove(engine_path)
                shutil.move(exported_path, engine_path)
                print(f"   Moved exported engine to: {engine_path}")

            # Clean up model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"✅ YOLO ultralytics export completed in {conversion_time:.1f}s")

            if not os.path.exists(engine_path):
                raise RuntimeError("Engine file was not created")

            engine_size = os.path.getsize(engine_path) / (1024 * 1024)
            print(f"   Engine file size: {engine_size:.1f} MB")

            return engine_path

        except Exception as e:
            print(f"❌ YOLO ultralytics export failed: {e}")
            print(f"   This might be due to:")
            print(f"   - Incompatible model format")
            print(f"   - TensorRT installation issues")
            print(f"   - CUDA/GPU memory issues")
            print(f"   - Unsupported model architecture")
            raise RuntimeError(f"YOLO ultralytics export failed for {pt_path}: {e}")


class ModelAnalyzer:
    """Analyzes YOLO models to extract architecture and configuration details"""

    @staticmethod
    def extract_model_info(model_path: str) -> Dict[str, Any]:
        """Extract comprehensive model information from YOLO model"""
        try:
            # Load model temporarily to extract info
            model = YOLO(model_path)

            model_info = {
                'task': model.task,
                'model_name': getattr(model.model, 'yaml', {}).get('model_name', 'unknown'),
                'nc': getattr(model.model, 'nc', 0),  # number of classes
                'names': getattr(model.model, 'names', {}),
                'yaml_file': getattr(model.model, 'yaml_file', None),
            }

            # Extract YOLO variant using folder path and filename
            filename = os.path.basename(model_path)
            folder_path = os.path.dirname(model_path)
            variant = ModelAnalyzer._detect_variant_comprehensive(model, filename, folder_path)
            model_info['variant'] = variant

            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return model_info

        except Exception as e:
            print(f"⚠️ Could not extract model info from {model_path}: {e}")
            return {
                'task': 'unknown',
                'model_name': 'unknown',
                'variant': 'unknown',
                'nc': 0,
                'names': {},
                'yaml_file': None
            }

    @staticmethod
    def _detect_variant_comprehensive(model, filename: str, folder_path: str) -> str:
        """Detect YOLO variant from folder path and filename"""

        # Method 1: Extract from folder path (most reliable for organized datasets)
        variant_from_folder = ModelAnalyzer._extract_variant_from_folder(folder_path)
        if variant_from_folder != 'unknown':
            print(f"   Variant detected from folder path: {variant_from_folder}")
            return variant_from_folder

        # Method 2: Parse filename
        variant_from_filename = ModelAnalyzer._extract_variant_from_filename(filename)
        if variant_from_filename != 'unknown':
            print(f"   Variant detected from filename: {variant_from_filename}")
            return variant_from_filename

        # Method 3: Use parameter count as fallback
        variant_from_params = ModelAnalyzer._infer_variant_from_parameters_precise(model)
        if variant_from_params != 'unknown':
            print(f"   Variant inferred from parameter count: {variant_from_params}")
            return variant_from_params

        print(f"   Warning: Could not determine variant, using 'unknown'")
        return 'unknown'

    @staticmethod
    def _extract_variant_from_folder(folder_path: str) -> str:
        """Extract YOLO variant from the entire absolute path"""
        # Use the full absolute path to search for variant indicators
        full_path_lower = os.path.abspath(folder_path).lower()

        # Look for patterns anywhere in the full path like:
        # /home/user/yolov8n/models/best.pt
        # /data/experiments/yolo_x/weights/model.pt
        # /models/yolov11s_pose/runs/best.pt
        patterns = [
            r'yolo(?:v)?(\d+)([nslmx])',  # yolov8n, yolo11s, etc.
            r'yolo([nslmx])(?:[_\-]|$)',  # yolon, yolos_, yolox-
            r'yolo[_\-]([nslmx])(?:[_\-]|$)',  # yolo_n, yolo-s
            r'(?:^|[_\-/])([nslmx])(?:[_\-/]|$)',  # _n_, /s/, etc.
            r'(?:nano|small|medium|large|xlarge)',  # full names
        ]

        print(f"   Searching in full path: {full_path_lower}")

        for pattern in patterns:
            match = re.search(pattern, full_path_lower)
            if match:
                print(f"   Found pattern '{match.group(0)}' in path")

                if 'nano' in match.group(0):
                    return 'n'
                elif 'small' in match.group(0):
                    return 's'
                elif 'medium' in match.group(0):
                    return 'm'
                elif 'large' in match.group(0) and 'xlarge' not in match.group(0):
                    return 'l'
                elif 'xlarge' in match.group(0):
                    return 'x'
                else:
                    # Extract the captured variant letter
                    groups = match.groups()
                    for group in groups:
                        if group and group in ['n', 's', 'm', 'l', 'x']:
                            return group

        return 'unknown'

    @staticmethod
    def _extract_variant_from_filename(filename: str) -> str:
        """Extract YOLO variant from filename patterns"""
        filename_lower = filename.lower()

        # Common patterns: yolov8n, yolo11s, yolov5m, etc.
        patterns = [
            r'yolo(?:v)?(\d+)([nslmx])',  # yolov8n, yolo11s, etc.
            r'yolo([nslmx])(?:v\d+)?',  # yolon, yolos, etc.
            r'(?:^|_)([nslmx])(?:_|$|\d)',  # _n_, _s_, etc.
            r'(?:nano|small|medium|large|xlarge)',  # full names
        ]

        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                if len(match.groups()) >= 2:
                    # Pattern with version and variant
                    variant = match.group(2)
                elif len(match.groups()) >= 1:
                    variant = match.group(1)
                else:
                    continue

                # Handle full names
                variant_map = {
                    'nano': 'n', 'small': 's', 'medium': 'm',
                    'large': 'l', 'xlarge': 'x'
                }
                variant = variant_map.get(variant, variant)

                if variant in ['n', 's', 'm', 'l', 'x']:
                    return variant

        return 'unknown'

    @staticmethod
    def _extract_variant_from_yaml(model) -> str:
        """Extract variant from model YAML configuration"""
        try:
            if hasattr(model.model, 'yaml') and isinstance(model.model.yaml, dict):
                yaml_config = model.model.yaml

                # Check for explicit variant in YAML
                if 'variant' in yaml_config:
                    variant = str(yaml_config['variant']).lower()
                    if variant in ['n', 's', 'm', 'l', 'x']:
                        return variant

                # Check model name in YAML
                if 'model_name' in yaml_config:
                    name = str(yaml_config['model_name']).lower()
                    for v in ['n', 's', 'm', 'l', 'x']:
                        if v in name:
                            return v

                # Check backbone configuration for known patterns
                if 'backbone' in yaml_config:
                    backbone = yaml_config['backbone']
                    if isinstance(backbone, list) and len(backbone) > 0:
                        # Look for channel configurations that indicate variant
                        return ModelAnalyzer._infer_from_backbone_channels(backbone)

        except Exception as e:
            print(f"   Warning: Error reading YAML config: {e}")

        return 'unknown'

    @staticmethod
    def _infer_from_backbone_channels(backbone) -> str:
        """Infer variant from backbone channel configurations"""
        try:
            # Look for Conv layers with specific channel patterns
            max_channels = 0
            conv_count = 0

            for layer in backbone:
                if isinstance(layer, list) and len(layer) >= 3:
                    layer_type = layer[2] if len(layer) > 2 else None
                    if 'Conv' in str(layer_type):
                        conv_count += 1
                        # Extract channel information if available
                        if len(layer) > 3 and isinstance(layer[3], (list, tuple)):
                            channels = layer[3]
                            if isinstance(channels, (list, tuple)) and len(channels) > 0:
                                if isinstance(channels[0], (int, float)):
                                    max_channels = max(max_channels, int(channels[0]))

            # Map channel patterns to variants (approximate)
            if max_channels > 0:
                if max_channels <= 320:
                    return 'n'
                elif max_channels <= 512:
                    return 's'
                elif max_channels <= 768:
                    return 'm'
                elif max_channels <= 1024:
                    return 'l'
                else:
                    return 'x'

        except Exception:
            pass

        return 'unknown'

    @staticmethod
    def _infer_variant_from_parameters_precise(model) -> str:
        """Infer variant from precise parameter count with updated thresholds"""
        try:
            param_count = sum(p.numel() for p in model.parameters())

            # Updated parameter count thresholds based on actual YOLO models
            # These are more accurate than the previous rough estimates

            # YOLOv8 approximate parameter counts:
            # v8n: ~3.2M, v8s: ~11.2M, v8m: ~25.9M, v8l: ~43.7M, v8x: ~68.2M
            # YOLOv11 approximate parameter counts:
            # v11n: ~2.6M, v11s: ~9.4M, v11m: ~20.1M, v11l: ~25.3M, v11x: ~56.9M

            if param_count < 4e6:  # < 4M params
                return 'n'
            elif param_count < 13e6:  # < 13M params
                return 's'
            elif param_count < 27e6:  # < 27M params
                return 'm'
            elif param_count < 50e6:  # < 50M params
                return 'l'
            else:  # >= 50M params
                return 'x'

        except Exception as e:
            print(f"   Warning: Error counting parameters: {e}")

        return 'unknown'

    @staticmethod
    def _analyze_model_architecture(model) -> str:
        """Analyze model architecture for variant clues"""
        try:
            # Count different types of layers
            conv_layers = 0
            bottleneck_layers = 0
            total_layers = 0

            def count_layers(module, prefix=""):
                nonlocal conv_layers, bottleneck_layers, total_layers

                for name, child in module.named_children():
                    total_layers += 1
                    module_name = type(child).__name__.lower()

                    if 'conv' in module_name:
                        conv_layers += 1
                    elif 'bottleneck' in module_name or 'c2f' in module_name or 'c3' in module_name:
                        bottleneck_layers += 1

                    # Recursively count in child modules
                    if len(list(child.children())) > 0:
                        count_layers(child, f"{prefix}.{name}" if prefix else name)

            count_layers(model.model)

            # Infer variant based on layer counts
            # These thresholds are approximate and based on typical YOLO architectures
            if total_layers < 150:
                return 'n'
            elif total_layers < 200:
                return 's'
            elif total_layers < 250:
                return 'm'
            elif total_layers < 300:
                return 'l'
            else:
                return 'x'

        except Exception as e:
            print(f"   Warning: Error analyzing architecture: {e}")

        return 'unknown'

    @staticmethod
    def extract_resolution_from_filename(filename: str) -> Optional[int]:
        """Extract image resolution from filename"""
        # Look for patterns like img640, 640p, size640, etc.
        patterns = [
            r'img(\d+)',
            r'(\d+)p',
            r'size(\d+)',
            r'res(\d+)',
            r'_(\d+)_',
            r'-(\d+)-'
        ]

        for pattern in patterns:
            match = re.search(pattern, filename.lower())
            if match:
                resolution = int(match.group(1))
                # Validate reasonable resolution range
                if 320 <= resolution <= 2048:
                    return resolution

        return None

    @staticmethod
    def generate_comprehensive_name(original_name: str, model_info: Dict, resolution: int,
                                    format_ext: str, precision: str = None) -> str:
        """Generate comprehensive model name based on all available info"""
        parts = []

        # Add task if available
        if model_info.get('task') and model_info['task'] != 'unknown':
            parts.append(f"yolo{model_info['task'][0]}")  # yolov8 -> yolov8, pose -> yolop
        else:
            parts.append("yolo")

        # Add variant
        variant = model_info.get('variant', 'unknown')
        if variant != 'unknown':
            parts.append(variant)

        # Add resolution
        parts.append(f"img{resolution}")

        # Add precision for TensorRT
        if precision:
            parts.append(precision)

        # Add original identifier if meaningful
        clean_original = re.sub(r'img\d+', '', original_name.lower())
        clean_original = re.sub(r'yolo[vnslmx]*\d*', '', clean_original)
        clean_original = re.sub(r'[_-]+', '_', clean_original).strip('_')
        if clean_original and clean_original not in ['best', 'last', 'model']:
            parts.append(clean_original)

        name = '_'.join(parts) + format_ext
        return name


class ModelBenchmarkAnalyzer:
    """Main analyzer class with improved model processing"""

    def __init__(self, model_folders: List[str], output_dir: str,
                 task: str = 'pose', device: int = 0, timing_runs: int = 100,
                 warmup_runs: int = 10, recursive: bool = False,
                 # TensorRT options (FP32/FP16 only)
                 use_tensorrt: bool = False, use_python_trt: bool = False,
                 trtexec_path: str = "/usr/src/tensorrt/bin/trtexec",
                 tensorrt_precision: str = "fp16", tensorrt_workspace: int = 1024,
                 convert_only: bool = False):

        self.model_folders = model_folders
        self.output_dir = output_dir
        self.task = task.lower()
        self.device = device
        self.timing_runs = timing_runs
        self.warmup_runs = warmup_runs
        self.recursive = recursive

        # TensorRT options
        self.use_tensorrt = use_tensorrt
        self.use_python_trt = use_python_trt
        self.tensorrt_precision = tensorrt_precision
        self.tensorrt_workspace = tensorrt_workspace
        self.convert_only = convert_only

        # Initialize precision list (FP32/FP16 only)
        self.target_precisions = []
        if self.use_tensorrt:
            if tensorrt_precision == "all":
                self.target_precisions = ["fp32", "fp16"]
                print(f"🔧 Multi-precision conversion: {', '.join(self.target_precisions)}")
            else:
                self.target_precisions = [tensorrt_precision]

        self.results = []
        self.timing_images = []
        self.tensorrt_converter = None

        # Create output directory structure
        self.models_output_dir = os.path.join(self.output_dir, "models")
        self.reports_output_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(self.models_output_dir, exist_ok=True)
        os.makedirs(self.reports_output_dir, exist_ok=True)

        # Initialize TensorRT converter if needed
        if self.use_tensorrt:
            try:
                self.tensorrt_converter = TensorRTConverter(
                    trtexec_path=trtexec_path,
                    use_python_trt=use_python_trt,
                    device=device,  # Pass device to converter
                    verbose=True
                )
                print(f"✅ TensorRT converter initialized")
            except Exception as e:
                print(f"⚠️ TensorRT initialization failed: {e}")
                self.use_tensorrt = False

        self._validate_inputs()

        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        sns.set_palette("husl")

    def _validate_inputs(self):
        """Validate all input paths"""
        for i, folder in enumerate(self.model_folders):
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Model folder {i + 1} not found: {folder}")
            print(f"✓ Model folder {i + 1}: {folder}")

        print("✓ All inputs validated")
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Task: {self.task}")
        print(f"✓ Device: {self.device}")

    def find_and_process_models(self) -> List[Dict[str, Any]]:
        """Find all .pt models and process them through the conversion pipeline"""
        model_files = []

        # Find all .pt files
        for folder_idx, folder in enumerate(self.model_folders, 1):
            print(f"\n🔍 Searching folder {folder_idx}: {folder}")
            search_path = Path(folder)

            pattern = "**/*.pt" if self.recursive else "*.pt"
            folder_models = 0

            for file_path in search_path.glob(pattern):
                if file_path.is_file():
                    # Extract resolution from filename
                    resolution = ModelAnalyzer.extract_resolution_from_filename(file_path.stem)
                    if resolution is None:
                        print(f"⚠️ Skipping {file_path.name}: cannot extract resolution")
                        continue

                    # Extract detailed model info
                    print(f"📋 Analyzing model: {file_path.name}")
                    model_info = ModelAnalyzer.extract_model_info(str(file_path))

                    # Create model record
                    model_record = {
                        'original_name': file_path.stem,
                        'original_path': str(file_path),
                        'extension': '.pt',
                        'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                        'resolution': resolution,
                        'source_folder': folder,
                        'model_info': model_info,
                        'generated_files': []
                    }

                    print(f"   Model info: task={model_info['task']}, variant={model_info['variant']}")
                    print(f"   Resolution: {resolution}px, Size: {model_record['size_mb']} MB")

                    model_files.append(model_record)
                    folder_models += 1

            print(f"  Found {folder_models} models in folder {folder_idx}")

        if not model_files:
            print("No .pt models found!")
            return []

        print(f"\nTotal found: {len(model_files)} PyTorch models")

        # Process each model through the conversion pipeline
        all_generated_files = []

        for i, model_record in enumerate(model_files, 1):
            print(f"\n[{i}/{len(model_files)}] Processing {model_record['original_name']}")
            generated = self._process_model_pipeline(model_record)
            all_generated_files.extend(generated)

        return all_generated_files

    def _process_model_pipeline(self, model_record: Dict) -> List[Dict[str, Any]]:
        """Process a single model through the complete pipeline"""
        generated_files = []

        # Step 1: Copy original .pt file with new naming
        pt_file = self._copy_with_comprehensive_naming(model_record, '.pt')
        if pt_file:
            generated_files.append(pt_file)

        # Step 2: Export to ONNX
        onnx_file = self._export_to_onnx(model_record)
        if onnx_file:
            generated_files.append(onnx_file)

            # Step 3: Convert ONNX to TensorRT (if enabled)
            if self.use_tensorrt and self.tensorrt_converter:
                trt_files = self._convert_to_tensorrt(onnx_file)
                generated_files.extend(trt_files)

        return generated_files

    def _copy_with_comprehensive_naming(self, model_record: Dict, extension: str) -> Optional[Dict]:
        """Copy file with comprehensive naming scheme"""
        try:
            new_name = ModelAnalyzer.generate_comprehensive_name(
                model_record['original_name'],
                model_record['model_info'],
                model_record['resolution'],
                extension
            )

            source_path = model_record['original_path']
            target_path = os.path.join(self.models_output_dir, new_name)

            # Avoid overwriting
            counter = 1
            while os.path.exists(target_path):
                base_name = new_name.replace(extension, f"_v{counter}{extension}")
                target_path = os.path.join(self.models_output_dir, base_name)
                counter += 1

            shutil.copy2(source_path, target_path)

            file_record = {
                'name': Path(target_path).stem,
                'path': target_path,
                'extension': extension,
                'size_mb': round(os.path.getsize(target_path) / (1024 * 1024), 2),
                'resolution': model_record['resolution'],
                'model_info': model_record['model_info'],
                'source_folder': model_record['source_folder'],
                'original_path': source_path,
                'comprehensive_name': new_name
            }

            print(f"  📁 Copied to: {new_name}")
            return file_record

        except Exception as e:
            print(f"  ❌ Failed to copy: {e}")
            return None

    def _export_to_onnx(self, model_record: Dict) -> Optional[Dict]:
        """Export PyTorch model to ONNX"""
        try:
            print(f"  📦 Exporting to ONNX...")

            # Generate ONNX name
            onnx_name = ModelAnalyzer.generate_comprehensive_name(
                model_record['original_name'],
                model_record['model_info'],
                model_record['resolution'],
                '.onnx'
            )

            onnx_path = os.path.join(self.models_output_dir, onnx_name)

            if os.path.exists(onnx_path):
                print(f"     ONNX already exists: {onnx_name}")
            else:
                # Load and export
                model = YOLO(model_record['original_path'], task=self.task)

                export_result = model.export(
                    format='onnx',
                    imgsz=model_record['resolution'],
                    dynamic=False,
                    simplify=True,
                    opset=11,
                )

                # Move to correct location if needed
                if export_result != onnx_path and os.path.exists(export_result):
                    shutil.move(export_result, onnx_path)

                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Create ONNX file record
            onnx_record = {
                'name': Path(onnx_path).stem,
                'path': onnx_path,
                'extension': '.onnx',
                'size_mb': round(os.path.getsize(onnx_path) / (1024 * 1024), 2),
                'resolution': model_record['resolution'],
                'model_info': model_record['model_info'],
                'source_folder': model_record['source_folder'],
                'original_pt_path': model_record['original_path'],
                'exported_from_pt': True,
                'comprehensive_name': onnx_name
            }

            print(f"     ✅ ONNX exported: {onnx_name} ({onnx_record['size_mb']} MB)")
            return onnx_record

        except Exception as e:
            print(f"     ❌ ONNX export failed: {e}")
            return None

    def _convert_to_tensorrt(self, onnx_record: Dict) -> List[Dict]:
        """Convert ONNX to TensorRT engines for all target precisions"""
        trt_files = []

        for precision in self.target_precisions:
            try:
                print(f"  🔧 Converting to TensorRT {precision.upper()}...")

                # Generate TensorRT engine name
                trt_name = ModelAnalyzer.generate_comprehensive_name(
                    onnx_record['name'],
                    onnx_record['model_info'],
                    onnx_record['resolution'],
                    '.engine',
                    precision
                )

                trt_path = os.path.join(self.models_output_dir, trt_name)

                if os.path.exists(trt_path):
                    print(f"     Engine already exists: {trt_name}")
                else:
                    # Convert using TensorRT converter (FP32/FP16 only)
                    self.tensorrt_converter.convert_onnx_to_engine(
                        onnx_path=onnx_record['path'],
                        engine_path=trt_path,
                        precision=precision,
                        max_workspace_size=self.tensorrt_workspace
                    )

                # Create TensorRT file record
                trt_record = {
                    'name': Path(trt_path).stem,
                    'path': trt_path,
                    'extension': '.engine',
                    'size_mb': round(os.path.getsize(trt_path) / (1024 * 1024), 2),
                    'resolution': onnx_record['resolution'],
                    'model_info': onnx_record['model_info'],
                    'source_folder': onnx_record['source_folder'],
                    'original_onnx_path': onnx_record['path'],
                    'converted_from_onnx': True,
                    'tensorrt_precision': precision,
                    'comprehensive_name': trt_name
                }

                trt_files.append(trt_record)
                print(f"     ✅ TensorRT {precision.upper()} created: {trt_name} ({trt_record['size_mb']} MB)")

            except Exception as e:
                print(f"     ❌ TensorRT {precision.upper()} conversion failed: {e}")
                continue

        return trt_files

    def analyze_models(self) -> List[Dict[str, Any]]:
        """Main analysis pipeline"""
        # Find and process all models
        model_files = self.find_and_process_models()

        if not model_files:
            print("No models to analyze!")
            return []

        if self.convert_only:
            print(f"\n✅ Conversion completed. Generated {len(model_files)} files.")
            return model_files

        print(f"\n{'=' * 80}")
        print(f"BENCHMARKING {len(model_files)} MODELS")
        print(f"{'=' * 80}")

        # Continue with benchmarking...
        # (The rest of the benchmarking code would follow here)

        return model_files

    # Additional methods for timing, validation, reporting, etc. would go here
    # (I'm keeping this focused on the core improvements you requested)


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Model Benchmark Analyzer with TensorRT Support (FP32/FP16)'
    )

    parser.add_argument('--model-folders', required=True, nargs='+',
                        help='Folders containing .pt model files')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for processed models and reports')
    parser.add_argument('--task', choices=['pose', 'detect', 'segment'], default='pose',
                        help='Model task type')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--recursive', action='store_true',
                        help='Search recursively in model folders')

    # TensorRT options (FP32/FP16 only)
    parser.add_argument('--use-tensorrt', action='store_true',
                        help='Convert models to TensorRT engines')
    parser.add_argument('--use-python-trt', action='store_true',
                        help='Use Python TensorRT instead of trtexec')
    parser.add_argument('--trtexec-path', default='/usr/src/tensorrt/bin/trtexec',
                        help='Path to trtexec executable')
    parser.add_argument('--tensorrt-precision', choices=['fp32', 'fp16', 'all'],
                        default='fp16', help='TensorRT precision mode (FP32/FP16 only)')
    parser.add_argument('--tensorrt-workspace', type=int, default=1024,
                        help='TensorRT workspace size in MB')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert models, skip benchmarking')

    args = parser.parse_args()

    print("🚀 Enhanced Model Benchmark Analyzer Starting")
    print(f"Model folders: {', '.join(args.model_folders)}")
    print(f"Task: {args.task}")
    print(f"Output: {args.output_dir}")

    if args.use_tensorrt:
        method = "Python TensorRT" if args.use_python_trt else "trtexec"
        print(f"🔧 TensorRT enabled ({method})")
        print(f"   Precision: {args.tensorrt_precision}")

    try:
        analyzer = ModelBenchmarkAnalyzer(
            model_folders=args.model_folders,
            output_dir=args.output_dir,
            task=args.task,
            device=args.device,
            recursive=args.recursive,
            use_tensorrt=args.use_tensorrt,
            use_python_trt=args.use_python_trt,
            trtexec_path=args.trtexec_path,
            tensorrt_precision=args.tensorrt_precision,
            tensorrt_workspace=args.tensorrt_workspace,
            convert_only=args.convert_only
        )

        results = analyzer.analyze_models()

        print(f"\n✅ Processing completed!")
        print(f"📁 Generated files: {len(results)}")
        print(f"📁 Models directory: {analyzer.models_output_dir}")

        if args.use_tensorrt:
            onnx_count = len([r for r in results if r['extension'] == '.onnx'])
            trt_count = len([r for r in results if r['extension'] == '.engine'])
            print(f"📦 ONNX exports: {onnx_count}")
            print(f"🔧 TensorRT engines: {trt_count}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()