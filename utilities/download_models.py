#!/usr/bin/env python3
"""
Model Download Script for Dataset Generation Pipelines

Downloads all required models for:
1. Reference Object Pipeline (PerSAM, SAM2, Zero123++, Qwen Image Edit, Qwen Multi-Camera + LoRA)
2. Scene Generation Pipeline (FLUX, Grounding DINO, SAM2)

Models are downloaded to a unified cache directory that works with HuggingFace's
native caching mechanism - running this script twice or running the pipelines
after will NOT re-download models.

Usage:
    python download_models.py --cache_dir ./models --all
    python download_models.py --cache_dir ./models --sam --sam2 --grounding_dino
    python download_models.py --cache_dir ./models --list
"""

import argparse
import os
import sys
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
import urllib.request
import shutil
import json

# Try to import optional dependencies
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, snapshot_download, scan_cache_dir
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    # SAM (Segment Anything) Models - Direct downloads (not HuggingFace)
    "sam": {
        "sam_vit_h": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "filename": "sam_vit_h_4b8939.pth",
            "size_mb": 2564,
            # "md5": "a7bf3b02f3ebf1267aba913ff637d9a2",
            "description": "SAM ViT-H (largest, best quality)"
        },
        "sam_vit_l": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "filename": "sam_vit_l_0b3195.pth",
            "size_mb": 1249,
            "md5": "3adcc4315b642a4d2101128f611684e8",
            "description": "SAM ViT-L (large)"
        },
        "sam_vit_b": {
            "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "filename": "sam_vit_b_01ec64.pth",
            "size_mb": 375,
            "md5": "ec2df62732614e57d3b16bf6329a885e",
            "description": "SAM ViT-B (base, fastest)"
        }
    },

    # SAM2 Models - HuggingFace
    "sam2": {
        "sam2_hiera_large": {
            "repo_id": "facebook/sam2-hiera-large",
            "filename": "sam2_hiera_large.pt",
            "config": "sam2_hiera_l.yaml",
            "size_mb": 898,
            "description": "SAM2 Hiera Large"
        },
        "sam2.1_hiera_large": {
            "repo_id": "facebook/sam2.1-hiera-large",
            "filename": "sam2.1_hiera_large.pt",
            "config": "sam2.1_hiera_l.yaml",
            "size_mb": 898,
            "description": "SAM2.1 Hiera Large (recommended)"
        },
        "sam2_hiera_base_plus": {
            "repo_id": "facebook/sam2-hiera-base-plus",
            "filename": "sam2_hiera_base_plus.pt",
            "config": "sam2_hiera_b+.yaml",
            "size_mb": 324,
            "description": "SAM2 Hiera Base+ (smaller)"
        },
        "sam2_hiera_small": {
            "repo_id": "facebook/sam2-hiera-small",
            "filename": "sam2_hiera_small.pt",
            "config": "sam2_hiera_s.yaml",
            "size_mb": 185,
            "description": "SAM2 Hiera Small (fastest)"
        }
    },

    # Grounding DINO Models - Direct downloads
    "grounding_dino": {
        "groundingdino_swint_ogc": {
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            "filename": "groundingdino_swint_ogc.pth",
            "size_mb": 694,
            "description": "Grounding DINO Swin-T (recommended)"
        },
        "groundingdino_swinb_cogcoor": {
            "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
            "filename": "groundingdino_swinb_cogcoor.pth",
            "size_mb": 938,
            "description": "Grounding DINO Swin-B (larger)"
        }
    },

    # HuggingFace Models - Use native HF caching
    "huggingface": {
        "qwen_image_edit": {
            "repo_id": "ovedrive/qwen-image-edit-4bit",
            "description": "Qwen Image Edit 4-bit (reference pipeline steps 1, 4, 5)",
            "pipeline": "reference"
        },
        "qwen_multicamera": {
            "repo_id": "ovedrive/Qwen-Image-Edit-2511-4bit",
            "description": "Qwen Image Edit 2511 4-bit base model (reference pipeline step 2)",
            "pipeline": "reference"
        },
        "qwen_multicamera_lora": {
            "repo_id": "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
            "description": "Multi-angle LoRA for Qwen Image Edit 2511 (reference pipeline step 2)",
            "pipeline": "reference"
        },
        "zero123plus_v1.2": {
            "repo_id": "sudo-ai/zero123plus-v1.2",
            "description": "Zero123++ v1.2 (reference pipeline step 2, alternative perspective generator)",
            "pipeline": "reference"
        },
        "flux_schnell": {
            "repo_id": "black-forest-labs/FLUX.1-schnell",
            "description": "FLUX.1 Schnell (scene generation pipeline - fast)",
            "pipeline": "scene"
        },
        "flux_dev": {
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "description": "FLUX.1 Dev (scene generation pipeline - higher quality, optional)",
            "pipeline": "optional"
        }
    }
}

# Grounding DINO config content
GROUNDING_DINO_CONFIG = '''# Grounding DINO config - SwinT OGC
batch_size = 1
modelname = "groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True
'''


# ============================================================================
# DOWNLOAD UTILITIES
# ============================================================================

class DownloadProgressBar:
    """Progress bar for downloads."""

    def __init__(self, total_size: int, desc: str = "Downloading"):
        self.total_size = total_size
        self.desc = desc
        self.downloaded = 0

        if TQDM_AVAILABLE:
            self.pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc)
        else:
            self.pbar = None
            print(f"{desc}: ", end="", flush=True)

    def update(self, block_size: int):
        self.downloaded += block_size
        if self.pbar:
            self.pbar.update(block_size)
        else:
            progress = self.downloaded / self.total_size * 100
            print(f"\r{self.desc}: {progress:.1f}%", end="", flush=True)

    def close(self):
        if self.pbar:
            self.pbar.close()
        else:
            print()


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest() == expected_md5


def download_file(url: str, dest_path: Path, desc: str = "Downloading",
                  expected_md5: Optional[str] = None) -> bool:
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        dest_path: Destination path
        desc: Description for progress bar
        expected_md5: Expected MD5 hash for verification

    Returns:
        True if successful, False otherwise
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and verify
    if dest_path.exists():
        if expected_md5:
            print(f"Verifying existing file: {dest_path.name}")
            if verify_md5(dest_path, expected_md5):
                print(f"✓ Already cached (verified): {dest_path.name}")
                return True
            else:
                print(f"✗ Existing file corrupted, re-downloading: {dest_path.name}")
        else:
            print(f"✓ Already cached: {dest_path.name}")
            return True

    temp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')

    try:
        # Get file size
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))

        # Download with progress
        progress = DownloadProgressBar(total_size, desc)

        def reporthook(block_num, block_size, total_size):
            progress.update(block_size)

        urllib.request.urlretrieve(url, temp_path, reporthook)
        progress.close()

        # Verify MD5 if provided
        if expected_md5:
            print(f"Verifying download...")
            if not verify_md5(temp_path, expected_md5):
                print(f"✗ MD5 verification failed for {dest_path.name}")
                temp_path.unlink()
                return False
            print(f"✓ MD5 verified")

        # Move to final location
        shutil.move(temp_path, dest_path)
        print(f"✓ Downloaded: {dest_path.name}")
        return True

    except Exception as e:
        print(f"✗ Failed to download {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def check_hf_cache_for_model(cache_dir: Path, repo_id: str) -> bool:
    """
    Check if a model is already in HuggingFace cache.

    Args:
        cache_dir: HuggingFace cache directory
        repo_id: Repository ID (e.g., "facebook/sam2-hiera-large")

    Returns:
        True if model appears to be cached
    """
    if not HF_HUB_AVAILABLE:
        return False

    try:
        cache_info = scan_cache_dir(cache_dir)
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception:
        # Fallback: check directory structure manually
        model_dir_name = f"models--{repo_id.replace('/', '--')}"
        model_path = cache_dir / model_dir_name
        return model_path.exists() and (model_path / "snapshots").exists()


def download_from_huggingface(repo_id: str, cache_dir: Path,
                               filename: Optional[str] = None,
                               force: bool = False) -> Optional[str]:
    """
    Download model from HuggingFace Hub using native caching.

    This uses HuggingFace's cache structure so:
    1. Running this script twice won't re-download
    2. Pipelines using the same cache_dir will find the models
    3. from_pretrained() calls will use the cached version

    Args:
        repo_id: HuggingFace repository ID
        cache_dir: Cache directory (will use HF's structure inside)
        filename: Specific filename to download (None for entire repo)
        force: Force re-download even if cached

    Returns:
        Path to downloaded file/directory or None if failed
    """
    if not HF_HUB_AVAILABLE:
        print("✗ huggingface_hub not installed. Install with: pip install huggingface_hub")
        return None

    # Check if already cached
    if not force and check_hf_cache_for_model(cache_dir, repo_id):
        if filename:
            # Return the expected path for the specific file
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(cache_dir),
                    local_files_only=True  # Don't download, just get path
                )
                print(f"✓ Already cached: {repo_id}/{filename}")
                return path
            except Exception:
                print(f"File {filename} not in cache, will download...")
                pass  # Fall through to download below
        else:
            print(f"✓ Already cached: {repo_id}")
            return str(cache_dir)

    try:
        if filename:
            # Download specific file - ONLY use cache_dir, not local_dir
            print(f"Downloading {filename} from {repo_id}...")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(cache_dir)
                # NO local_dir - use native HF cache structure
            )
            print(f"✓ Downloaded: {filename}")
            return path
        else:
            # Download entire repository - ONLY use cache_dir
            print(f"Downloading repository: {repo_id}...")
            path = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir)
                # NO local_dir - use native HF cache structure
            )
            print(f"✓ Downloaded repository: {repo_id}")
            return path

    except Exception as e:
        print(f"✗ Failed to download from HuggingFace: {e}")
        return None


# ============================================================================
# MODEL DOWNLOAD FUNCTIONS
# ============================================================================

def download_sam_models(cache_dir: Path, variants: List[str] = ["sam_vit_h"]) -> Dict[str, Path]:
    """Download SAM models (direct download, not HuggingFace)."""
    print("\n" + "=" * 60)
    print("Downloading SAM Models (Direct Download)")
    print("=" * 60)

    # SAM models go in a dedicated subdirectory (not HF cache structure)
    sam_dir = cache_dir / "checkpoints" / "sam"
    sam_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    for variant in variants:
        if variant not in MODELS["sam"]:
            print(f"✗ Unknown SAM variant: {variant}")
            continue

        model_info = MODELS["sam"][variant]
        dest_path = sam_dir / model_info["filename"]

        print(f"\n{model_info['description']} ({model_info['size_mb']} MB)")

        success = download_file(
            url=model_info["url"],
            dest_path=dest_path,
            desc=f"Downloading {model_info['filename']}",
            expected_md5=model_info.get("md5")
        )

        if success:
            downloaded[variant] = dest_path

    return downloaded


def download_sam2_models(cache_dir: Path, variants: List[str] = ["sam2.1_hiera_large"]) -> Dict[str, Dict]:
    """Download SAM2 models from HuggingFace using native caching."""
    print("\n" + "=" * 60)
    print("Downloading SAM2 Models (HuggingFace)")
    print("=" * 60)

    if not HF_HUB_AVAILABLE:
        print("✗ huggingface_hub required for SAM2 downloads")
        print("  Install with: pip install huggingface_hub")
        return {}

    downloaded = {}

    for variant in variants:
        if variant not in MODELS["sam2"]:
            print(f"✗ Unknown SAM2 variant: {variant}")
            continue

        model_info = MODELS["sam2"][variant]
        print(f"\n{model_info['description']} ({model_info['size_mb']} MB)")

        try:
            # Download checkpoint using native HF caching
            ckpt_path = download_from_huggingface(
                repo_id=model_info["repo_id"],
                cache_dir=cache_dir,
                filename=model_info["filename"]
            )

            # Download config using native HF caching
            config_path = download_from_huggingface(
                repo_id=model_info["repo_id"],
                cache_dir=cache_dir,
                filename=model_info["config"]
            )

            if ckpt_path and config_path:
                downloaded[variant] = {
                    "checkpoint": Path(ckpt_path),
                    "config": Path(config_path),
                    "repo_id": model_info["repo_id"]
                }
                print(f"✓ Checkpoint: {model_info['filename']}")
                print(f"✓ Config: {model_info['config']}")

        except Exception as e:
            print(f"✗ Failed to download {variant}: {e}")

    return downloaded


def download_grounding_dino_models(cache_dir: Path,
                                    variants: List[str] = ["groundingdino_swint_ogc"]) -> Dict[str, Dict]:
    """Download Grounding DINO models and create configs."""
    print("\n" + "=" * 60)
    print("Downloading Grounding DINO Models (Direct Download)")
    print("=" * 60)

    # Grounding DINO goes in dedicated subdirectory
    gdino_dir = cache_dir / "checkpoints" / "grounding_dino"
    gdino_dir.mkdir(parents=True, exist_ok=True)

    configs_dir = gdino_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    downloaded = {}

    for variant in variants:
        if variant not in MODELS["grounding_dino"]:
            print(f"✗ Unknown Grounding DINO variant: {variant}")
            continue

        model_info = MODELS["grounding_dino"][variant]
        ckpt_path = gdino_dir / model_info["filename"]

        print(f"\n{model_info['description']} ({model_info['size_mb']} MB)")

        success = download_file(
            url=model_info["url"],
            dest_path=ckpt_path,
            desc=f"Downloading {model_info['filename']}"
        )

        if success:
            # Create config file
            config_filename = model_info["filename"].replace(".pth", ".py")
            config_path = configs_dir / config_filename

            if not config_path.exists():
                with open(config_path, 'w') as f:
                    f.write(GROUNDING_DINO_CONFIG)
                print(f"✓ Created config: {config_filename}")
            else:
                print(f"✓ Config already exists: {config_filename}")

            downloaded[variant] = {
                "checkpoint": ckpt_path,
                "config": config_path
            }

    return downloaded


def download_huggingface_models(cache_dir: Path,
                                 models: List[str] = None) -> Dict[str, str]:
    """
    Pre-download HuggingFace models using native caching.

    These models will be cached in HuggingFace's standard structure,
    so pipelines using from_pretrained() with the same cache_dir
    will use the cached versions automatically.
    """
    print("\n" + "=" * 60)
    print("Downloading HuggingFace Models (Native Caching)")
    print("=" * 60)

    if not HF_HUB_AVAILABLE:
        print("✗ huggingface_hub required for HuggingFace downloads")
        print("  Install with: pip install huggingface_hub")
        return {}

    if models is None:
        models = DEFAULT_HF_MODELS

    downloaded = {}

    for model_name in models:
        if model_name not in MODELS["huggingface"]:
            print(f"✗ Unknown HuggingFace model: {model_name}")
            continue

        model_info = MODELS["huggingface"][model_name]
        print(f"\n{model_info['description']}")
        print(f"Repository: {model_info['repo_id']}")

        # Download entire repo using native HF caching
        path = download_from_huggingface(
            repo_id=model_info["repo_id"],
            cache_dir=cache_dir,
            filename=None  # Download entire repo
        )

        if path:
            downloaded[model_name] = model_info["repo_id"]

    return downloaded


# ============================================================================
# DEFAULT MODEL SELECTIONS
# ============================================================================

# Models required by both pipelines (--all downloads these)
# Reference pipeline: qwen_image_edit, qwen_multicamera, qwen_multicamera_lora, zero123plus
# Scene pipeline:     flux_schnell
DEFAULT_HF_MODELS = [
    "qwen_image_edit",
    "qwen_multicamera",
    "qwen_multicamera_lora",
    "zero123plus_v1.2",
    "flux_schnell",
]

# All models including optional ones (--include_optional adds these)
ALL_HF_MODELS = list(MODELS["huggingface"].keys())


# ============================================================================
# CONFIGURATION AND SUMMARY
# ============================================================================

def create_paths_config(cache_dir: Path, results: Dict) -> Path:
    """Create a configuration file with model paths for easy reference."""
    config = {
        "cache_dir": str(cache_dir.absolute()),
        "hf_cache_dir": str(cache_dir.absolute()),  # Same dir for HF models
        "models": {
            "sam": {},
            "sam2": {},
            "grounding_dino": {},
            "huggingface": {}
        },
        "usage_notes": {
            "huggingface_models": "Use cache_dir parameter in from_pretrained() calls",
            "direct_downloads": "Use the absolute paths listed below",
            "environment_variable": f"export HF_HOME={cache_dir.absolute()}"
        }
    }

    # SAM paths (direct downloads)
    if results.get("sam"):
        for name, path in results["sam"].items():
            config["models"]["sam"][name] = str(path.absolute())

    # SAM2 paths (HuggingFace)
    if results.get("sam2"):
        for name, info in results["sam2"].items():
            config["models"]["sam2"][name] = {
                "checkpoint": str(info["checkpoint"].absolute()),
                "config": str(info["config"].absolute()),
                "repo_id": info.get("repo_id", "")
            }

    # Grounding DINO paths (direct downloads)
    if results.get("grounding_dino"):
        for name, info in results["grounding_dino"].items():
            config["models"]["grounding_dino"][name] = {
                "checkpoint": str(info["checkpoint"].absolute()),
                "config": str(info["config"].absolute())
            }

    # HuggingFace models (repo IDs - they use native caching)
    if results.get("huggingface"):
        for name, repo_id in results["huggingface"].items():
            config["models"]["huggingface"][name] = {
                "repo_id": repo_id,
                "note": "Use from_pretrained(repo_id, cache_dir=cache_dir)"
            }

    config_path = cache_dir / "model_paths.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Model paths config saved to: {config_path}")
    return config_path


def create_env_script(cache_dir: Path) -> Path:
    """Create a shell script to set environment variables."""
    script_content = f'''#!/bin/bash
# Set HuggingFace cache directory
export HF_HOME="{cache_dir.absolute()}"
export HF_HUB_CACHE="{cache_dir.absolute()}"
export TRANSFORMERS_CACHE="{cache_dir.absolute()}"

echo "HuggingFace cache set to: $HF_HOME"
'''

    script_path = cache_dir / "set_cache_env.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)

    script_path.chmod(0o755)
    print(f"✓ Environment script saved to: {script_path}")
    print(f"  Source it with: source {script_path}")
    return script_path


def list_available_models():
    """List all available models."""
    print("\n" + "=" * 60)
    print("AVAILABLE MODELS")
    print("=" * 60)

    for category, models in MODELS.items():
        print(f"\n{category.upper()}:")
        for name, info in models.items():
            desc = info.get("description", "No description")
            size = info.get("size_mb", "varies")
            pipeline = info.get("pipeline", "")
            default_marker = ""

            # Mark which models are in the default --all set
            if category == "huggingface":
                if name in DEFAULT_HF_MODELS:
                    default_marker = " [default]"
                else:
                    default_marker = " [optional]"

            if isinstance(size, int):
                print(f"  • {name}: {desc} ({size} MB){default_marker}")
            else:
                print(f"  • {name}: {desc}{default_marker}")


def print_usage_instructions(cache_dir: Path, results: Dict):
    """Print instructions for using downloaded models."""
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)

    print(f"\nAll models downloaded to: {cache_dir.absolute()}")
    print(f"\nIMPORTANT: HuggingFace models use native caching.")
    print(f"Running this script again will NOT re-download cached models.")

    # Environment setup
    print("\n1. Set environment variable (recommended):")
    print(f"   export HF_HOME={cache_dir.absolute()}")
    print(f"   # Or source the generated script:")
    print(f"   source {cache_dir.absolute()}/set_cache_env.sh")

    # Scene generation pipeline usage
    print("\n2. For scene_generation_pipeline.py:")
    print("   python pipelines/scene_generation_pipeline.py \\")
    print(f"       --cache_dir {cache_dir.absolute()} \\")
    print("       --working_dir ./generated_scenes --session new \\")
    print("       --run_prompts --run_images --run_annotations \\")
    print("       --project_info_file examples/prompts/kitchen_objects.txt \\")
    print("       --predefined_classes \"cup,bottle,glass\" \\")
    print("       --num_prompts_per_execution 10 --num_random_imgs 2")

    # Reference pipeline usage
    print("\n3. For dataset_pipeline.py:")
    print("   python pipelines/dataset_pipeline.py \\")
    print(f"       --cache_dir {cache_dir.absolute()} \\")
    print("       --input_image <image> --input_mask <mask> --object_name <name>")

    print("\n" + "=" * 60)


def show_cache_status(cache_dir: Path):
    """Show what's currently cached."""
    print("\n" + "=" * 60)
    print("CACHE STATUS")
    print("=" * 60)

    cache_dir = Path(cache_dir)

    if not cache_dir.exists():
        print(f"Cache directory does not exist: {cache_dir}")
        return

    # Check direct download models
    checkpoints_dir = cache_dir / "checkpoints"
    if checkpoints_dir.exists():
        print("\nDirect downloads (checkpoints/):")
        for model_type in ["sam", "grounding_dino"]:
            model_dir = checkpoints_dir / model_type
            if model_dir.exists():
                files = list(model_dir.glob("*.pth")) + list(model_dir.glob("*.pt"))
                if files:
                    print(f"  {model_type}:")
                    for f in files:
                        size_mb = f.stat().st_size / (1024 * 1024)
                        print(f"    ✓ {f.name} ({size_mb:.1f} MB)")

    # Check HuggingFace cache
    if HF_HUB_AVAILABLE:
        try:
            cache_info = scan_cache_dir(cache_dir)
            if cache_info.repos:
                print("\nHuggingFace cache:")
                for repo in cache_info.repos:
                    size_mb = repo.size_on_disk / (1024 * 1024)
                    # Mark if it's a default or optional model
                    marker = ""
                    for name, info in MODELS["huggingface"].items():
                        if info["repo_id"] == repo.repo_id:
                            if name in DEFAULT_HF_MODELS:
                                marker = " [default]"
                            else:
                                marker = " [optional]"
                            break
                    print(f"  ✓ {repo.repo_id} ({size_mb:.1f} MB){marker}")
            else:
                print("\nHuggingFace cache: empty")
        except Exception as e:
            print(f"\nCould not scan HuggingFace cache: {e}")

    # Check for config file
    config_path = cache_dir / "model_paths.json"
    if config_path.exists():
        print(f"\n✓ Config file exists: {config_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download models for dataset generation pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all required models for both pipelines
    python download_models.py --cache_dir ./models --all

    # Download specific model types
    python download_models.py --cache_dir ./models --sam --sam2

    # Download with optional model variants (e.g. FLUX Dev)
    python download_models.py --cache_dir ./models --all --include_optional

    # List available models
    python download_models.py --list

    # Check what's cached
    python download_models.py --cache_dir ./models --status

Default --all downloads:
    Direct:       SAM ViT-H, SAM2.1 Hiera Large, Grounding DINO Swin-T
    HuggingFace:  Qwen Image Edit, Qwen Multi-Camera + LoRA,
                  Zero123++ v1.2, FLUX.1 Schnell

Optional (--include_optional):
    Additional SAM/SAM2 variants, Grounding DINO Swin-B, FLUX.1 Dev

Notes:
    - HuggingFace models use native caching (won't re-download if cached)
    - Direct downloads (SAM, Grounding DINO) check for existing files
    - Running this script multiple times is safe and efficient
        """
    )

    parser.add_argument('--cache_dir', type=str, default='./models',
                        help='Directory to download models to (default: ./models)')
    parser.add_argument('--list', action='store_true',
                        help='List available models and exit')
    parser.add_argument('--status', action='store_true',
                        help='Show cache status and exit')

    # Model selection
    parser.add_argument('--all', action='store_true',
                        help='Download all required models for both pipelines')
    parser.add_argument('--sam', action='store_true',
                        help='Download SAM models')
    parser.add_argument('--sam2', action='store_true',
                        help='Download SAM2 models')
    parser.add_argument('--grounding_dino', action='store_true',
                        help='Download Grounding DINO models')
    parser.add_argument('--huggingface', action='store_true',
                        help='Pre-download HuggingFace models (FLUX, Qwen, Zero123++, etc.)')

    # Options
    parser.add_argument('--include_optional', action='store_true',
                        help='Include optional model variants (e.g. FLUX Dev, extra SAM sizes)')
    parser.add_argument('--sam_variant', type=str, nargs='+',
                        default=['sam_vit_h'],
                        help='SAM variants to download (default: sam_vit_h)')
    parser.add_argument('--sam2_variant', type=str, nargs='+',
                        default=['sam2.1_hiera_large'],
                        help='SAM2 variants to download (default: sam2.1_hiera_large)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if cached')

    args = parser.parse_args()

    # List models and exit
    if args.list:
        list_available_models()
        return

    cache_dir = Path(args.cache_dir).absolute()

    # Show status and exit
    if args.status:
        show_cache_status(cache_dir)
        return

    # Create cache directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {cache_dir}")

    # Check dependencies
    if not HF_HUB_AVAILABLE:
        print("\n⚠️  Warning: huggingface_hub not installed")
        print("   SAM2 and HuggingFace model downloads will be skipped.")
        print("   Install with: pip install huggingface_hub")

    if not TQDM_AVAILABLE:
        print("\n⚠️  Warning: tqdm not installed (optional)")
        print("   Progress bars will be simplified. Install with: pip install tqdm")

    results = {
        "sam": {},
        "sam2": {},
        "grounding_dino": {},
        "huggingface": {}
    }

    # Determine what to download
    download_sam = args.all or args.sam
    download_sam2 = args.all or args.sam2
    download_gdino = args.all or args.grounding_dino
    download_hf = args.all or args.huggingface

    if not any([download_sam, download_sam2, download_gdino, download_hf]):
        print("\nNo models selected. Use --all or specify individual models.")
        print("Use --list to see available models.")
        print("Use --status to check what's cached.")
        return

    # Download selected models
    if download_sam:
        variants = args.sam_variant
        if args.include_optional:
            variants = list(MODELS["sam"].keys())
        results["sam"] = download_sam_models(cache_dir, variants)

    if download_sam2:
        variants = args.sam2_variant
        if args.include_optional:
            variants = list(MODELS["sam2"].keys())
        results["sam2"] = download_sam2_models(cache_dir, variants)

    if download_gdino:
        variants = ["groundingdino_swint_ogc"]
        if args.include_optional:
            variants = list(MODELS["grounding_dino"].keys())
        results["grounding_dino"] = download_grounding_dino_models(cache_dir, variants)

    if download_hf:
        if args.include_optional:
            models = ALL_HF_MODELS
        else:
            models = DEFAULT_HF_MODELS
        results["huggingface"] = download_huggingface_models(cache_dir, models)

    # Create config and env files
    create_paths_config(cache_dir, results)
    create_env_script(cache_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    total_downloaded = 0
    for category, items in results.items():
        if items:
            print(f"\n{category.upper()}: {len(items)} model(s)")
            total_downloaded += len(items)
            for name in items:
                print(f"  ✓ {name}")

    print(f"\nTotal models downloaded/verified: {total_downloaded}")

    # Print usage instructions
    print_usage_instructions(cache_dir, results)


if __name__ == "__main__":
    main()