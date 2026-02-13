#!/usr/bin/env python3
"""
Automatic Dataset Generation Pipeline with YOLO Annotation and Decoupled Augmentations

Pipeline steps:
1. Generate white background image using QwenImageEdit
2. Generate multiple perspectives using Zero123++ or Qwen Multi-Camera
3. Generate N prompts using prompt generator
4. Run each perspective through all prompts M times
5. Optional: Augment images with QwenImageEdit (generative augmentation) — NO labels
6. Annotate dataset using PerSAM → SAM2 (YOLO format) on the current image set
7. Optional: Apply Computer Vision augmentations (traditional augmentation)

Key design decision: PerSAM annotation runs AFTER generative augmentation so that
every image (original + Qwen-edited) gets its own accurate segmentation mask,
rather than blindly copying labels from originals whose geometry may have shifted.

Enhanced with:
- Selectable perspective generator (zero123 or qwen_multicamera)
- Decoupled Generative (Qwen) and CV augmentations
- CV augmentations can work on any YOLO dataset (step 6 output)
- Automatic retries on failures
- Robust error recovery
- YOLO dataset annotation
- Data augmentation capabilities
"""

import argparse
import os
import json
import time
import shutil
import random
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from PIL import Image
import torch

import sys

# Add the OpenFabrik directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import your existing classes
from modules.diffusion_models.image_edit_generator import QwenImageEditGenerator
from modules.diffusion_models.zero123_perspective_generator import Zero123Plus
from modules.diffusion_models.qwen_multicamera_generator import QwenMultiCameraGenerator
from modules.llm_models.prompt_generator import PromptGenerator
from modules.cv_processing.computer_vision_augmentation import ComputerVisionAugmentation

# Import the extended PerSAM processor (assuming it's in the same directory or properly installed)
from ref_segmentation.ref_segmentation_perseg import PerSAMProcessor


class ModelContextManager:
    """Helper to manage model loading/unloading and memory cleanup."""
    def __init__(self, model_instance):
        self.model = model_instance

    def __enter__(self):
        self.model.load_model()
        return self.model

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model.unload_model()

        # Aggressive cleanup: ensure all residual tensors from the previous model
        # are fully released before the next model loads. Without this, PyTorch's
        # caching allocator may hold fragmented blocks from the previous model,
        # causing OOM when the next model tries to allocate during VAE decode.
        gc.collect()
        gc.collect()  # Second pass catches ref cycles freed in the first

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # Reset allocator statistics so fragmentation info is fresh
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"[Memory cleanup] Allocated: {allocated:.2f} GiB, Reserved: {reserved:.2f} GiB")


class DatasetPipeline:
    """Main pipeline for automatic dataset generation with YOLO annotation and decoupled augmentations."""

    # Supported perspective generators
    PERSPECTIVE_GENERATORS = ["zero123", "qwen_multicamera"]

    def __init__(self, working_dir: str, cache_dir: str, model_path: str = "ovedrive/qwen-image-edit-4bit",
                 max_retries: int = 3,
                 # Perspective generator selection
                 perspective_generator: str = "qwen_multicamera",
                 # PerSAM/SAM2 arguments (unchanged)
                 sam_type: str = "vit_h", sam_ckpt: str = "sam_vit_h_4b8939.pth",
                 sam2_config: str = "sam2_hiera_l.yaml", sam2_ckpt: str = "sam2_hiera_large.pt",
                 enable_persam_training: bool = False, persam_lr: float = 1e-3,
                 persam_epochs: int = 1000, persam_log_epoch: int = 200,

                 # Computer Vision augmentation arguments
                 enable_cv_augmentation: bool = True,
                 bw_probability: float = 0.5,
                 saturation_probability: float = 0.3,
                 contrast_probability: float = 0.3,
                 brightness_probability: float = 0.3,
                 motion_blur_probability: float = 0.2,
                 compression_noise_probability: float = 0.2,
                 saturation_range: Tuple[float, float] = (0.5, 1.5),
                 contrast_range: Tuple[float, float] = (0.7, 1.3),
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 motion_blur_range: Tuple[int, int] = (3, 15),
                 compression_iterations_range: Tuple[int, int] = (5, 20),
                 compression_quality_range: Tuple[int, int] = (20, 80)):
        """
        Initialize the pipeline.
        """
        self.working_dir = Path(working_dir)
        self.cache_dir = cache_dir
        self.max_retries = max_retries

        # Perspective generator selection
        if perspective_generator not in self.PERSPECTIVE_GENERATORS:
            raise ValueError(
                f"Unknown perspective_generator '{perspective_generator}'. "
                f"Must be one of: {self.PERSPECTIVE_GENERATORS}"
            )
        self.perspective_generator_name = perspective_generator

        # PerSAM/SAM2 configuration
        self.sam_type = sam_type
        self.sam_ckpt = sam_ckpt
        self.sam2_config = sam2_config
        self.sam2_ckpt = sam2_ckpt
        self.enable_persam_training = enable_persam_training
        self.persam_lr = persam_lr
        self.persam_epochs = persam_epochs
        self.persam_log_epoch = persam_log_epoch

        # Store CV augmentation parameters
        self.enable_cv_augmentation = enable_cv_augmentation
        self.bw_probability = bw_probability
        self.saturation_probability = saturation_probability
        self.contrast_probability = contrast_probability
        self.brightness_probability = brightness_probability
        self.motion_blur_probability = motion_blur_probability
        self.compression_noise_probability = compression_noise_probability
        self.saturation_range = saturation_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.motion_blur_range = motion_blur_range
        self.compression_iterations_range = compression_iterations_range
        self.compression_quality_range = compression_quality_range

        # Initialize CV augmentation module
        if self.enable_cv_augmentation:
            self.cv_augmentation = ComputerVisionAugmentation(
                bw_probability=bw_probability,
                saturation_probability=saturation_probability,
                contrast_probability=contrast_probability,
                brightness_probability=brightness_probability,
                motion_blur_probability=motion_blur_probability,
                compression_noise_probability=compression_noise_probability,
                saturation_range=saturation_range,
                contrast_range=contrast_range,
                brightness_range=brightness_range,
                motion_blur_range=motion_blur_range,
                compression_iterations_range=compression_iterations_range,
                compression_quality_range=compression_quality_range
            )
            print("Computer Vision augmentation module initialized")
        else:
            self.cv_augmentation = None
            print("Computer Vision augmentation disabled")

        # Initialize components
        self.image_editor = QwenImageEditGenerator(model_path=model_path, cache_dir=cache_dir)
        self.prompt_generator = PromptGenerator()

        # Initialize perspective generator
        self._init_perspective_generator()

        # Initialize PerSAM processor with current configuration
        self.persam_processor = None

        # Pipeline state
        self.execution_dir = None
        self.state = {}

    def _init_perspective_generator(self):
        """Initialize the selected perspective generator."""
        if self.perspective_generator_name == "zero123":
            self.perspective_generator = Zero123Plus(version="v1.2", cache_dir=self.cache_dir)
            print("Perspective generator: Zero123Plus v1.2")
        elif self.perspective_generator_name == "qwen_multicamera":
            self.perspective_generator = QwenMultiCameraGenerator(cache_dir=self.cache_dir)
            print("Perspective generator: Qwen Multi-Camera")
        else:
            raise ValueError(f"Unknown perspective generator: {self.perspective_generator_name}")

    def _create_persam_processor(self):
        """Create PerSAM processor with current configuration"""

        class PerSAMArgs:
            def __init__(self, pipeline):
                self.workflow = 'persam_to_sam2'
                self.sam_type = pipeline.sam_type
                self.sam_ckpt = pipeline.sam_ckpt
                self.sam2_config = pipeline.sam2_config
                self.sam2_ckpt = pipeline.sam2_ckpt
                self.enable_training = pipeline.enable_persam_training
                self.lr = pipeline.persam_lr
                self.train_epoch = pipeline.persam_epochs
                self.log_epoch = pipeline.persam_log_epoch
                self.use_persam_points = True
                self.use_persam_box = True
                self.compare_results = False
                self.ref_image = None
                self.ref_mask = None
                self.test_image = None
                self.output_dir = None

        persam_args = PerSAMArgs(self)
        persam_processor = PerSAMProcessor(persam_args)
        print("PerSAM processor initialized")
        return persam_processor

    def setup_execution_dir(self, execution_mode: str, execution_name: str = None) -> Path:
        """
        Set up execution directory based on mode.

        Args:
            execution_mode: "new", "last", or "by_name"
            execution_name: Required if mode is "by_name"

        Returns:
            Path to execution directory
        """
        self.working_dir.mkdir(parents=True, exist_ok=True)

        if execution_mode == "new":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.execution_dir = self.working_dir / timestamp
            self.execution_dir.mkdir(parents=True, exist_ok=True)

        elif execution_mode == "last":
            execution_dirs = [d for d in self.working_dir.iterdir() if d.is_dir()]
            if not execution_dirs:
                raise ValueError("No previous executions found. Use 'new' mode.")
            self.execution_dir = max(execution_dirs, key=lambda x: x.stat().st_mtime)

        elif execution_mode == "by_name":
            if not execution_name:
                raise ValueError("execution_name is required when mode is 'by_name'")
            self.execution_dir = self.working_dir / execution_name
            if not self.execution_dir.exists():
                raise ValueError(f"Execution '{execution_name}' not found")

        else:
            raise ValueError("execution_mode must be 'new', 'last', or 'by_name'")

        print(f"Using execution directory: {self.execution_dir}")

        self.state_file = self.execution_dir / "pipeline_state.json"
        self.load_state()

        return self.execution_dir

    def load_state(self):
        """Load pipeline state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = {
                "completed_steps": [],
                "input_image": None,
                "object_name": None,
                "white_bg_image": None,
                "perspective_generator": None,
                "perspective_images": [],
                "perspective_view_info": [],
                "prompts": [],
                "classes": [],
                "generated_images": [],
                "failed_images": [],
                # Step 5: Qwen augmentation (images only, before annotation)
                "qwen_combined_images_path": None,
                "qwen_augmentation_summary": None,
                "failed_qwen_augmentations": [],
                # Step 6: PerSAM annotation
                "yolo_dataset_path": None,
                "annotation_summary": None,
                # Step 7: CV augmentation
                "cv_augmented_dataset_path": None,
                "cv_augmentation_summary": None,
                "failed_cv_augmentations": [],
            }

    def save_state(self):
        """Save pipeline state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def _generate_image(self, prompt: str, image_path: str, output_path: str) -> str:
        """Generate an edited image."""
        return self.image_editor.generate_image(prompt, image_path, output_path)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 1 — White Background
    # ──────────────────────────────────────────────────────────────────────

    def step1_generate_white_background(self, input_image_path: str, object_name: str) -> str:
        """Step 1: Generate white background image using QwenImageEdit with retry logic."""
        print("\n=== STEP 1: Generate White Background Image ===")

        if 1 in self.state["completed_steps"]:
            print("Step 1 already completed. Skipping.")
            return self.state["white_bg_image"]

        self.state["input_image"] = str(input_image_path)
        self.state["object_name"] = object_name

        prompt = f"{object_name} with white background, while maintaining object's features and shape"
        output_path = self.execution_dir / "01_white_background.png"

        print(f"Generating: {prompt}")
        print(f"Input image: {input_image_path}")

        try:
            with ModelContextManager(self.image_editor):
                last_exception = None
                for attempt in range(self.max_retries + 1):
                    try:
                        if attempt > 0:
                            print(f"Retry attempt {attempt}/{self.max_retries}")
                        self._generate_image(
                            prompt=prompt,
                            image_path=input_image_path,
                            output_path=str(output_path)
                        )
                        last_exception = None
                        break
                    except Exception as e:
                        last_exception = e
                        print(f"Attempt {attempt + 1} failed: {e}")
                if last_exception is not None:
                    raise last_exception
        except Exception as e:
            print(f"Critical error in Step 1: {e}")
            raise

        self.state["white_bg_image"] = str(output_path)
        self.state["completed_steps"].append(1)
        self.save_state()

        print(f"White background image saved to: {output_path}")
        return str(output_path)

    # ──────────────────────────────────────────────────────────────────────
    # STEP 2 — Perspectives
    # ──────────────────────────────────────────────────────────────────────

    def step2_generate_perspectives(self) -> List[str]:
        """Step 2: Generate multiple perspectives using the selected perspective generator."""
        print(f"\n=== STEP 2: Generate Multiple Perspectives ({self.perspective_generator_name}) ===")

        if 2 in self.state["completed_steps"]:
            print("Step 2 already completed. Skipping.")
            return self.state["perspective_images"]

        if not self.state["white_bg_image"]:
            raise ValueError("Step 1 must be completed first")

        perspectives_dir = self.execution_dir / "02_perspectives"
        perspectives_dir.mkdir(exist_ok=True)

        with ModelContextManager(self.perspective_generator):
            views, view_info = self.perspective_generator.generate_views(
                self.state["white_bg_image"]
            )

            perspective_paths = []
            for i, (view, info) in enumerate(zip(views, view_info)):
                if self.perspective_generator_name == "zero123":
                    filename = (
                        f"perspective_{i:02d}_az{info['azimuth']:03d}"
                        f"_el{info['elevation']:+03d}.png"
                    )
                elif self.perspective_generator_name == "qwen_multicamera":
                    view_clean = info["view"].replace(" ", "_").replace("-", "_")
                    elev_clean = info["elevation"].replace(" ", "_").replace("-", "_")
                    filename = f"perspective_{i:02d}_{view_clean}__{elev_clean}.png"
                else:
                    filename = f"perspective_{i:02d}.png"

                filepath = perspectives_dir / filename
                view.save(filepath)
                perspective_paths.append(str(filepath))
                print(f"Saved perspective {i + 1}/{len(views)}: {filename}")

            if self.perspective_generator_name == "zero123":
                combined = self.perspective_generator.get_last_combined_image()
                if combined:
                    combined.save(perspectives_dir / "combined_all_views.png")

        self.state["perspective_generator"] = self.perspective_generator_name
        self.state["perspective_images"] = perspective_paths
        self.state["perspective_view_info"] = view_info
        self.state["completed_steps"].append(2)
        self.save_state()

        print(f"✓ Generated {len(perspective_paths)} perspectives using {self.perspective_generator_name}")
        return perspective_paths

    # ──────────────────────────────────────────────────────────────────────
    # STEP 3 — Prompts
    # ──────────────────────────────────────────────────────────────────────

    def step3_generate_prompts(self, num_prompts: int, system_prompt_file: str, project_info_file: str) -> Tuple[
        List[str], List[str]]:
        """Step 3: Generate N prompts using prompt generator."""
        print(f"\n=== STEP 3: Generate {num_prompts} Prompts ===")

        if 3 in self.state["completed_steps"]:
            print("Step 3 already completed. Skipping.")
            return self.state["prompts"], self.state["classes"]

        prompts_dir = self.execution_dir / "03_prompts"
        prompts_dir.mkdir(exist_ok=True)

        prompts, classes = self.prompt_generator.generate_prompts_iteratively(
            system_prompt_file=system_prompt_file,
            project_info_file=project_info_file,
            num_prompts=num_prompts,
            execution_id=1
        )

        self.prompt_generator.save_results(output_dir=str(prompts_dir))

        self.state["prompts"] = prompts
        self.state["classes"] = classes
        self.state["completed_steps"].append(3)
        self.save_state()

        self.prompt_generator.unload_model()

        print(f"✓ Generated {len(prompts)} prompts and {len(classes)} classes")
        return prompts, classes

    # ──────────────────────────────────────────────────────────────────────
    # STEP 4 — Generate Dataset
    # ──────────────────────────────────────────────────────────────────────

    def _generate_single_image_with_retry(self, perspective_path: str, prompt: str,
                                          output_path: Path, p_idx: int, prompt_idx: int,
                                          iteration: int, seed: int) -> bool:
        """Generate a single image with retry logic."""
        filename = output_path.name

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print(f"    Retry {attempt}/{self.max_retries} for {filename}")

                self._generate_image(
                    prompt=prompt,
                    image_path=perspective_path,
                    output_path=str(output_path)
                )

                print(f"    Generated: {filename}")
                return True

            except Exception as e:
                print(f"    Attempt {attempt + 1} failed for {filename}: {e}")

        failed_info = {
            "perspective_idx": p_idx,
            "prompt_idx": prompt_idx,
            "iteration": iteration,
            "seed": seed,
            "filename": filename,
            "perspective_path": perspective_path,
            "prompt": prompt
        }

        if "failed_images" not in self.state:
            self.state["failed_images"] = []
        self.state["failed_images"].append(failed_info)

        print(f"    ✗ Failed to generate {filename} after {self.max_retries + 1} attempts")
        return False

    def step4_generate_dataset(self, num_iterations: int) -> List[str]:
        """Step 4: Generate dataset by running each perspective through all prompts."""
        print(f"\n=== STEP 4: Generate Dataset (M={num_iterations} iterations per prompt) ===")

        if 4 in self.state["completed_steps"]:
            print("Step 4 already completed. Skipping.")
            return self.state["generated_images"]

        if not all([self.state["perspective_images"], self.state["prompts"]]):
            raise ValueError("Steps 2 and 3 must be completed first")

        dataset_dir = self.execution_dir / "04_dataset"
        dataset_dir.mkdir(exist_ok=True)

        generated_images = self.state.get("generated_images", [])
        num_perspectives = len(self.state["perspective_images"])
        num_prompts = len(self.state["prompts"])
        total_images = num_perspectives * num_prompts * num_iterations
        current_image = len(generated_images)

        print(f"Perspectives: {num_perspectives}, Prompts: {num_prompts}, Iterations: {num_iterations}")
        print(f"Will generate {total_images} images total")
        print(f"Already generated: {current_image} images")

        successful_count = 0
        failed_count = 0

        with ModelContextManager(self.image_editor):
            for p_idx, perspective_path in enumerate(self.state["perspective_images"]):
                for prompt_idx, prompt in enumerate(self.state["prompts"]):
                    for iteration in range(num_iterations):
                        current_image += 1

                        seed = np.random.randint(0, 2 ** 32)
                        filename = f"perspective_{p_idx}_prompt_{prompt_idx}_seed_{seed}.png"
                        output_path = dataset_dir / filename

                        if str(output_path) in generated_images:
                            print(f"[{current_image}/{total_images}] Skipping existing: {filename}")
                            continue

                        print(f"[{current_image}/{total_images}] Generating: {filename}")
                        print(f"  Prompt: {prompt[:50]}...")

                        success = self._generate_single_image_with_retry(
                            perspective_path, prompt, output_path,
                            p_idx, prompt_idx, iteration, seed
                        )

                        if success:
                            generated_images.append(str(output_path))
                            successful_count += 1
                        else:
                            failed_count += 1

                        if (successful_count + failed_count) % 10 == 0:
                            self.state["generated_images"] = generated_images
                            self.save_state()

        self.state["generated_images"] = generated_images
        self.state["completed_steps"].append(4)
        self.save_state()

        metadata = {
            "object_name": self.state["object_name"],
            "input_image": self.state["input_image"],
            "perspective_generator": self.state.get("perspective_generator", "unknown"),
            "num_perspectives": len(self.state["perspective_images"]),
            "num_prompts": len(self.state["prompts"]),
            "num_iterations_per_prompt": num_iterations,
            "num_generated_images": len(self.state["generated_images"]),
            "num_successful_images": successful_count,
            "num_failed_images": failed_count,
            "prompts": self.state["prompts"],
            "classes": self.state["classes"],
            "perspective_view_info": self.state.get("perspective_view_info", []),
            "failed_images": self.state.get("failed_images", [])
        }

        with open(dataset_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        if failed_count > 0:
            self._generate_failure_report(dataset_dir)

        print(f"Generated {successful_count} images successfully")
        print(f"Failed to generate {failed_count} images")
        if (successful_count + failed_count) > 0:
            print(f"Success rate: {successful_count / (successful_count + failed_count) * 100:.1f}%")

        return generated_images

    # ──────────────────────────────────────────────────────────────────────
    # STEP 5 — Qwen Generative Augmentation (IMAGES ONLY, before annotation)
    # ──────────────────────────────────────────────────────────────────────

    def step5_qwen_augment_images(self, predefined_prompts_file: str, augmentation_count: int = 1,
                                  enable_qwen_augmentation: bool = True) -> Optional[str]:
        """
        Step 5: Augment raw images with QwenImageEdit (generative augmentation).

        Produces a combined images directory (original + Qwen-augmented) that will
        be annotated in step 6 by PerSAM. No labels are created here — every image
        gets its own accurate mask from PerSAM instead of copying potentially
        misaligned labels from the originals.

        Creates:
        - 05_qwen_augmented_images/  — only Qwen-augmented images
        - 05_qwen_combined_images/   — original + Qwen-augmented images
        """
        print(f"\n=== STEP 5: Qwen Generative Augmentation (images only, pre-annotation) ===")

        if not enable_qwen_augmentation:
            print("Generative augmentation disabled. Skipping step 5.")
            return None

        if 5 in self.state["completed_steps"]:
            print("Step 5 already completed. Skipping.")
            return self.state["qwen_combined_images_path"]

        if not self.state["generated_images"]:
            raise ValueError("Step 4 must be completed first")

        # Load predefined prompts
        prompts_file = Path(predefined_prompts_file)
        if not prompts_file.exists():
            raise ValueError(f"Predefined prompts file not found: {predefined_prompts_file}")

        print(f"Loading predefined prompts from: {predefined_prompts_file}")
        with open(prompts_file, 'r') as f:
            predefined_prompts = [line.strip() for line in f if line.strip()]

        if not predefined_prompts:
            raise ValueError("No prompts found in predefined prompts file")

        print(f"Loaded {len(predefined_prompts)} predefined prompts")
        print(f"Will generate {augmentation_count} Qwen-augmented version(s) per original image")

        # Source: raw images from step 4
        dataset_dir = self.execution_dir / "04_dataset"

        # Output directories (flat image directories, NOT YOLO structure)
        qwen_augmented_dir = self.execution_dir / "05_qwen_augmented_images"
        qwen_combined_dir = self.execution_dir / "05_qwen_combined_images"

        qwen_augmented_dir.mkdir(parents=True, exist_ok=True)
        qwen_combined_dir.mkdir(parents=True, exist_ok=True)

        # Copy all original images into the combined directory
        original_images = list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpg"))
        print(f"Found {len(original_images)} original images in step 4 output")

        for img_path in original_images:
            shutil.copy2(img_path, qwen_combined_dir / img_path.name)

        print(f"Copied originals into combined directory: {qwen_combined_dir}")

        # Initialize tracking
        if "failed_qwen_augmentations" not in self.state:
            self.state["failed_qwen_augmentations"] = []

        successful_augmentations = 0
        failed_augmentations = 0
        total_augmentations = 0

        with ModelContextManager(self.image_editor):
            for img_idx, image_file in enumerate(original_images, 1):
                print(f"\n[{img_idx}/{len(original_images)}] Processing {image_file.name}")
                print(f"  Generating {augmentation_count} Qwen-augmented version(s)...")

                for aug_idx in range(augmentation_count):
                    total_augmentations += 1

                    prompt = random.choice(predefined_prompts)

                    base_name = image_file.stem
                    aug_image_name = f"{base_name}_qwen{aug_idx:03d}.png"

                    aug_only_path = qwen_augmented_dir / aug_image_name
                    combined_path = qwen_combined_dir / aug_image_name

                    print(f"    [{aug_idx + 1}/{augmentation_count}] {aug_image_name}")
                    print(f"      Prompt: {prompt[:60]}...")

                    success = self._generate_qwen_image_with_retry(
                        str(image_file), prompt,
                        aug_only_path, combined_path,
                        image_file.name, aug_idx, augmentation_count
                    )

                    if success:
                        successful_augmentations += 1
                    else:
                        failed_augmentations += 1

                    # Save progress periodically
                    if total_augmentations % 20 == 0:
                        self.save_state()

        # Build summary
        qwen_augmentation_summary = {
            "source_dataset": str(dataset_dir),
            "qwen_augmented_images": str(qwen_augmented_dir),
            "qwen_combined_images": str(qwen_combined_dir),
            "predefined_prompts_file": predefined_prompts_file,
            "num_predefined_prompts": len(predefined_prompts),
            "augmentation_count_per_image": augmentation_count,
            "original_image_count": len(original_images),
            "total_augmentations_attempted": total_augmentations,
            "successful_augmentations": successful_augmentations,
            "failed_augmentations": failed_augmentations,
            "success_rate": successful_augmentations / total_augmentations * 100 if total_augmentations > 0 else 0,
            "total_combined_images": len(original_images) + successful_augmentations,
            "failed_augmentation_details": self.state.get("failed_qwen_augmentations", []),
        }

        # Save summary into both directories
        for d in [qwen_augmented_dir, qwen_combined_dir]:
            with open(d / "qwen_augmentation_summary.json", 'w') as f:
                json.dump(qwen_augmentation_summary, f, indent=2)

        if failed_augmentations > 0:
            self._generate_qwen_augmentation_failure_report(qwen_augmented_dir, qwen_combined_dir)

        # Update state
        self.state["qwen_combined_images_path"] = str(qwen_combined_dir)
        self.state["qwen_augmentation_summary"] = qwen_augmentation_summary
        self.state["completed_steps"].append(5)
        self.save_state()

        print(f"\n✓ Qwen image augmentation completed!")
        print(f"  Original images: {len(original_images)}")
        print(f"  Successful augmentations: {successful_augmentations}")
        print(f"  Failed augmentations: {failed_augmentations}")
        print(f"  Total combined images: {len(original_images) + successful_augmentations}")
        print(f"\n  📁 Qwen-augmented only: {qwen_augmented_dir}")
        print(f"  📁 Combined (for annotation): {qwen_combined_dir}")

        return str(qwen_combined_dir)

    def _generate_qwen_image_with_retry(self, original_image_path: str, prompt: str,
                                        aug_only_path: Path, combined_path: Path,
                                        img_name: str, aug_idx: int, aug_count: int) -> bool:
        """Generate a single Qwen-augmented image with retry logic (no labels involved)."""
        filename = aug_only_path.name

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    print(f"      Retry {attempt}/{self.max_retries} for {filename}")

                self._generate_image(
                    prompt=prompt,
                    image_path=original_image_path,
                    output_path=str(aug_only_path)
                )

                # Also copy into the combined directory
                shutil.copy2(aug_only_path, combined_path)

                print(f"      ✓ Generated: {filename}")
                return True

            except Exception as e:
                print(f"      Attempt {attempt + 1} failed for {filename}: {e}")

        # All attempts failed
        failed_info = {
            "original_image": original_image_path,
            "aug_filename": filename,
            "aug_index": aug_idx,
            "prompt": prompt,
            "timestamp": datetime.now().isoformat(),
            "error_type": "qwen_augmentation_failure"
        }

        if "failed_qwen_augmentations" not in self.state:
            self.state["failed_qwen_augmentations"] = []
        self.state["failed_qwen_augmentations"].append(failed_info)

        print(f"      ✗ Failed to generate {filename} after {self.max_retries + 1} attempts")
        return False

    # ──────────────────────────────────────────────────────────────────────
    # STEP 6 — PerSAM Annotation (runs on whichever image set is current)
    # ──────────────────────────────────────────────────────────────────────

    def step6_annotate_dataset(self, input_mask_path: str, split_ratio: float = 0.8) -> str:
        """
        Step 6: Annotate dataset using PerSAM → SAM2 in YOLO format.

        Automatically picks the right image source:
        - If step 5 ran (Qwen augmentation): annotates the combined images directory
        - Otherwise: annotates the raw step 4 dataset

        This ensures every image — including Qwen-augmented ones — gets its own
        accurate segmentation mask from PerSAM.
        """
        print(f"\n=== STEP 6: Annotate Dataset with PerSAM → SAM2 (YOLO Format) ===")

        if 6 in self.state["completed_steps"]:
            print("Step 6 already completed. Skipping.")
            return self.state["yolo_dataset_path"]

        # Determine which image directory to annotate
        if self.state.get("qwen_combined_images_path") and Path(self.state["qwen_combined_images_path"]).exists():
            images_dir = Path(self.state["qwen_combined_images_path"])
            source_desc = "step 5 combined (original + Qwen-augmented)"
        elif self.state.get("generated_images"):
            images_dir = self.execution_dir / "04_dataset"
            source_desc = "step 4 (raw generated images)"
        else:
            raise ValueError("No images to annotate. Complete step 4 (and optionally step 5) first.")

        if self.persam_processor is None:
            self.persam_processor = self._create_persam_processor()

        ref_image_path = self.state["input_image"]

        yolo_dir = self.execution_dir / "06_yolo_dataset"

        print(f"Annotating images from: {images_dir} ({source_desc})")
        print(f"Reference image (input): {ref_image_path}")
        print(f"Reference mask: {input_mask_path}")
        print(f"Output YOLO dataset: {yolo_dir}")

        try:
            yolo_dataset_path = self.persam_processor.annotate_dataset(
                dataset_dir=str(images_dir),
                ref_image_path=ref_image_path,
                ref_mask_path=input_mask_path,
                output_dir=str(yolo_dir),
                class_name=self.state["object_name"],
                split_ratio=split_ratio
            )

            summary_path = Path(yolo_dataset_path) / "annotation_summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    annotation_summary = json.load(f)
                self.state["annotation_summary"] = annotation_summary
            else:
                annotation_summary = {}

            self.state["yolo_dataset_path"] = yolo_dataset_path
            self.state["completed_steps"].append(6)
            self.save_state()

            print(f"✓ YOLO dataset annotation completed!")
            print(f"✓ Dataset path: {yolo_dataset_path}")
            print(f"✓ Image source: {source_desc}")
            if annotation_summary:
                print(f"✓ Successful annotations: {annotation_summary.get('successful_annotations', 0)}")
                print(f"✗ Failed annotations: {annotation_summary.get('failed_annotations', 0)}")

            return yolo_dataset_path

        except Exception as e:
            print(f"✗ Dataset annotation failed: {e}")
            raise

    # ──────────────────────────────────────────────────────────────────────
    # STEP 7 — Computer Vision Augmentation (on annotated YOLO dataset)
    # ──────────────────────────────────────────────────────────────────────

    def step7_cv_augment_dataset(self, augmentation_count: int = 1,
                                 enable_cv_augmentation: bool = True) -> Optional[str]:
        """
        Step 7: Apply Computer Vision augmentations to the annotated YOLO dataset.

        CV augmentations (BW, saturation, contrast, brightness, motion blur,
        compression noise) do not move/reshape the object, so copying labels
        verbatim is correct here.

        Creates:
        - 07_cv_augmented_dataset/  — only CV-augmented images
        - 07_cv_combined_dataset/   — annotated originals + CV-augmented images
        """
        print(f"\n=== STEP 7: Apply Computer Vision Augmentations ===")

        if not enable_cv_augmentation:
            print("CV augmentation disabled. Skipping step 7.")
            return None

        if 7 in self.state["completed_steps"]:
            print("Step 7 already completed. Skipping.")
            return self.state["cv_augmented_dataset_path"]

        if not self.enable_cv_augmentation or not self.cv_augmentation:
            print("CV augmentation module not initialized. Skipping step 7.")
            return None

        # Input is always the annotated YOLO dataset from step 6
        if not self.state.get("yolo_dataset_path") or not Path(self.state["yolo_dataset_path"]).exists():
            raise ValueError("No annotated dataset found. Complete step 6 first.")

        input_dataset_path = Path(self.state["yolo_dataset_path"])
        print(f"Input dataset: {input_dataset_path}")
        print(f"Will generate {augmentation_count} CV-augmented version(s) per image")

        cv_augmented_dir = self.execution_dir / "07_cv_augmented_dataset"
        cv_combined_dir = self.execution_dir / "07_cv_combined_dataset"

        self._create_empty_yolo_structure(cv_augmented_dir, input_dataset_path)
        shutil.copytree(input_dataset_path, cv_combined_dir, dirs_exist_ok=True)

        print(f"Created CV-augmented dataset structure: {cv_augmented_dir}")
        print(f"Created CV-combined dataset (with originals): {cv_combined_dir}")

        cv_augmentation_stats = {
            "input_dataset": str(input_dataset_path),
            "total_images_processed": 0,
            "images_with_cv_augmentations": 0,
            "cv_augmentation_types_applied": {
                "bw_conversion": 0,
                "saturation": 0,
                "contrast": 0,
                "brightness": 0,
                "motion_blur": 0,
                "compression_noise": 0
            },
            "cv_augmentations_per_image": [],
            "augmentation_probabilities_used": self.cv_augmentation.get_augmentation_summary()
        }

        if "failed_cv_augmentations" not in self.state:
            self.state["failed_cv_augmentations"] = []

        successful_augmentations = 0
        failed_augmentations = 0
        total_augmentations = 0

        for split in ['train', 'val']:
            input_images_dir = input_dataset_path / split / 'images'
            input_labels_dir = input_dataset_path / split / 'labels'

            cv_augmented_images_dir = cv_augmented_dir / split / 'images'
            cv_augmented_labels_dir = cv_augmented_dir / split / 'labels'

            cv_combined_images_dir = cv_combined_dir / split / 'images'
            cv_combined_labels_dir = cv_combined_dir / split / 'labels'

            if not input_images_dir.exists():
                print(f"No {split} split found, skipping...")
                continue

            image_files = list(input_images_dir.glob('*.jpg')) + list(input_images_dir.glob('*.png'))
            print(f"\nProcessing {split} split: {len(image_files)} images")

            if not image_files:
                continue

            for img_idx, image_file in enumerate(image_files, 1):
                label_file = input_labels_dir / f"{image_file.stem}.txt"

                if not label_file.exists():
                    print(f"Warning: No label file found for {image_file.name}, skipping...")
                    continue

                print(f"\n[{img_idx}/{len(image_files)}] Processing {image_file.name}")

                image_success_count = 0
                for aug_idx in range(augmentation_count):
                    total_augmentations += 1
                    cv_augmentation_stats["total_images_processed"] += 1

                    base_name = image_file.stem
                    aug_image_name = f"{base_name}_cv{aug_idx:03d}.jpg"
                    aug_label_name = f"{base_name}_cv{aug_idx:03d}.txt"

                    aug_image_path = cv_augmented_images_dir / aug_image_name
                    aug_label_path = cv_augmented_labels_dir / aug_label_name

                    combined_image_path = cv_combined_images_dir / aug_image_name
                    combined_label_path = cv_combined_labels_dir / aug_label_name

                    print(f"    [{aug_idx + 1}/{augmentation_count}] {aug_image_name}")

                    try:
                        input_image = Image.open(image_file)

                        cv_augmented_image, cv_augmentations_applied = self.cv_augmentation.apply_cascade_augmentations(
                            input_image)

                        cv_augmented_image.save(aug_image_path)
                        cv_augmented_image.save(combined_image_path)

                        # CV augmentations don't change geometry → labels are identical
                        shutil.copy2(label_file, aug_label_path)
                        shutil.copy2(label_file, combined_label_path)

                        successful_augmentations += 1
                        image_success_count += 1

                        if cv_augmentations_applied:
                            cv_augmentation_stats["images_with_cv_augmentations"] += 1
                            cv_augmentation_stats["cv_augmentations_per_image"].append({
                                "image": aug_image_name,
                                "augmentations": cv_augmentations_applied
                            })

                            for aug_desc in cv_augmentations_applied:
                                if "bw_conversion" in aug_desc:
                                    cv_augmentation_stats["cv_augmentation_types_applied"]["bw_conversion"] += 1
                                elif "saturation" in aug_desc:
                                    cv_augmentation_stats["cv_augmentation_types_applied"]["saturation"] += 1
                                elif "contrast" in aug_desc:
                                    cv_augmentation_stats["cv_augmentation_types_applied"]["contrast"] += 1
                                elif "brightness" in aug_desc:
                                    cv_augmentation_stats["cv_augmentation_types_applied"]["brightness"] += 1
                                elif "motion_blur" in aug_desc:
                                    cv_augmentation_stats["cv_augmentation_types_applied"]["motion_blur"] += 1
                                elif "compression_noise" in aug_desc:
                                    cv_augmentation_stats["cv_augmentation_types_applied"]["compression_noise"] += 1

                            print(f"      Applied: {', '.join(cv_augmentations_applied)}")
                        else:
                            print(f"      No CV augmentations applied (random selection)")

                    except Exception as e:
                        failed_augmentations += 1
                        print(f"      ✗ Failed: {e}")

                        self.state["failed_cv_augmentations"].append({
                            "input_image": str(image_file),
                            "aug_filename": aug_image_name,
                            "aug_index": aug_idx,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                            "error_type": "cv_augmentation_failure"
                        })

                    if total_augmentations % 50 == 0:
                        self.save_state()

                print(f"  Result: {image_success_count}/{augmentation_count} successful")

            # Per-split summary
            actual_aug = len(list(cv_augmented_images_dir.glob('*.jpg')) + list(cv_augmented_images_dir.glob('*.png')))
            actual_combined = len(list(cv_combined_images_dir.glob('*.jpg')) + list(cv_combined_images_dir.glob('*.png')))
            print(f"\nCompleted {split}: {actual_aug} augmented, {actual_combined} combined")

        # Update dataset.yaml files
        self._update_dataset_yaml_files(input_dataset_path, cv_augmented_dir, cv_combined_dir,
                                         "CV-augmented", "CV-combined")

        cv_augmentation_summary = {
            "input_dataset": str(input_dataset_path),
            "cv_augmented_dataset": str(cv_augmented_dir),
            "cv_combined_dataset": str(cv_combined_dir),
            "augmentation_count_per_image": augmentation_count,
            "total_augmentations_attempted": total_augmentations,
            "successful_augmentations": successful_augmentations,
            "failed_augmentations": failed_augmentations,
            "success_rate": successful_augmentations / total_augmentations * 100 if total_augmentations > 0 else 0,
            "failed_augmentation_details": self.state.get("failed_cv_augmentations", []),
            "cv_augmentation_stats": cv_augmentation_stats,
            "datasets_created": {
                "cv_augmented_only": str(cv_augmented_dir),
                "cv_combined": str(cv_combined_dir)
            }
        }

        for d in [cv_augmented_dir, cv_combined_dir]:
            with open(d / "cv_augmentation_summary.json", 'w') as f:
                json.dump(cv_augmentation_summary, f, indent=2)

        if failed_augmentations > 0:
            self._generate_cv_augmentation_failure_report(cv_augmented_dir, cv_combined_dir)

        self.state["cv_augmented_dataset_path"] = str(cv_combined_dir)
        self.state["cv_augmentation_summary"] = cv_augmentation_summary
        self.state["completed_steps"].append(7)
        self.save_state()

        print(f"\n✓ CV augmentation completed!")
        print(f"  Successful: {successful_augmentations}, Failed: {failed_augmentations}")
        if cv_augmentation_stats["total_images_processed"] > 0:
            print(f"  CV augmentation rate: "
                  f"{cv_augmentation_stats['images_with_cv_augmentations'] / cv_augmentation_stats['total_images_processed'] * 100:.1f}%")
        print(f"\n  📁 CV-augmented only: {cv_augmented_dir}")
        print(f"  📁 CV-combined: {cv_combined_dir}")

        return str(cv_combined_dir)

    # ──────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ──────────────────────────────────────────────────────────────────────

    def _generate_qwen_augmentation_failure_report(self, *dirs):
        """Generate a failure report for Qwen augmentation failures."""
        if not self.state.get("failed_qwen_augmentations"):
            return

        for d in dirs:
            report_path = d / "qwen_augmentation_failure_report.txt"
            with open(report_path, 'w') as f:
                f.write("QWEN AUGMENTATION FAILURE REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total failed: {len(self.state['failed_qwen_augmentations'])}\n\n")

                for i, failed_aug in enumerate(self.state["failed_qwen_augmentations"], 1):
                    f.write(f"Failed Augmentation {i}:\n")
                    f.write(f"  Original: {failed_aug['original_image']}\n")
                    f.write(f"  Filename: {failed_aug['aug_filename']}\n")
                    f.write(f"  Index: {failed_aug['aug_index']}\n")
                    f.write(f"  Timestamp: {failed_aug['timestamp']}\n")
                    f.write(f"  Prompt: {failed_aug['prompt'][:100]}...\n\n")

            print(f"✓ Qwen failure report saved to: {report_path}")

    def _generate_cv_augmentation_failure_report(self, *dirs):
        """Generate a failure report for CV augmentation failures."""
        if not self.state.get("failed_cv_augmentations"):
            return

        for d in dirs:
            report_path = d / "cv_augmentation_failure_report.txt"
            with open(report_path, 'w') as f:
                f.write("CV AUGMENTATION FAILURE REPORT\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total failed: {len(self.state['failed_cv_augmentations'])}\n\n")

                for i, failed_aug in enumerate(self.state["failed_cv_augmentations"], 1):
                    f.write(f"Failed Augmentation {i}:\n")
                    f.write(f"  Input: {failed_aug['input_image']}\n")
                    f.write(f"  Filename: {failed_aug['aug_filename']}\n")
                    f.write(f"  Index: {failed_aug['aug_index']}\n")
                    f.write(f"  Timestamp: {failed_aug['timestamp']}\n")
                    f.write(f"  Error: {failed_aug['error']}\n\n")

            print(f"✓ CV failure report saved to: {report_path}")

    def _update_dataset_yaml_files(self, source_dir, augmented_dir, combined_dir,
                                   aug_type, combined_type):
        """Update dataset.yaml files for augmented and combined datasets."""
        augmented_yaml = augmented_dir / "dataset.yaml"
        if augmented_yaml.exists():
            with open(augmented_yaml, 'r') as f:
                content = f.read()
            content = content.replace(str(source_dir), str(augmented_dir))
            content = f"# {aug_type} dataset\n{content}"
            with open(augmented_yaml, 'w') as f:
                f.write(content)

        combined_yaml = combined_dir / "dataset.yaml"
        if combined_yaml.exists():
            with open(combined_yaml, 'r') as f:
                content = f.read()
            content = content.replace(str(source_dir), str(combined_dir))
            content = f"# {combined_type} dataset\n{content}"
            with open(combined_yaml, 'w') as f:
                f.write(content)

    def _create_empty_yolo_structure(self, target_dir, source_dir):
        """Create empty YOLO dataset structure based on source directory."""
        target_dir.mkdir(parents=True, exist_ok=True)

        for config_file in ['dataset.yaml', 'classes.txt']:
            source_file = source_dir / config_file
            if source_file.exists():
                shutil.copy2(source_file, target_dir / config_file)

        for split in ['train', 'val', 'test']:
            split_dir = source_dir / split
            if split_dir.exists():
                target_split_dir = target_dir / split
                target_split_dir.mkdir(parents=True, exist_ok=True)
                (target_split_dir / 'images').mkdir(exist_ok=True)
                (target_split_dir / 'labels').mkdir(exist_ok=True)

    def _generate_failure_report(self, dataset_dir):
        """Generate a detailed failure report for step 4."""
        if not self.state.get("failed_images"):
            return

        report_path = dataset_dir / "failure_report.txt"
        with open(report_path, 'w') as f:
            f.write("FAILURE REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total failed images: {len(self.state['failed_images'])}\n\n")

            for i, failed_img in enumerate(self.state["failed_images"], 1):
                f.write(f"Failed Image {i}:\n")
                f.write(f"  Filename: {failed_img['filename']}\n")
                f.write(f"  Perspective: {failed_img['perspective_idx']}\n")
                f.write(f"  Prompt: {failed_img['prompt_idx']}\n")
                f.write(f"  Iteration: {failed_img['iteration']}\n")
                f.write(f"  Seed: {failed_img['seed']}\n")
                f.write(f"  Prompt text: {failed_img['prompt'][:100]}...\n\n")

        print(f"✓ Failure report saved to: {report_path}")

    # ──────────────────────────────────────────────────────────────────────
    # Retry Helpers
    # ──────────────────────────────────────────────────────────────────────

    def retry_failed_images(self) -> int:
        """Retry generating failed images from step 4."""
        if not self.state.get("failed_images"):
            print("No failed images to retry.")
            return 0

        print(f"\n=== RETRYING {len(self.state['failed_images'])} FAILED IMAGES ===")

        dataset_dir = self.execution_dir / "04_dataset"

        with ModelContextManager(self.image_editor):
            successful_retries = 0
            remaining_failures = []

            for i, failed_img in enumerate(self.state["failed_images"], 1):
                print(f"[{i}/{len(self.state['failed_images'])}] Retrying: {failed_img['filename']}")

                output_path = dataset_dir / failed_img['filename']

                success = self._generate_single_image_with_retry(
                    failed_img['perspective_path'],
                    failed_img['prompt'],
                    output_path,
                    failed_img['perspective_idx'],
                    failed_img['prompt_idx'],
                    failed_img['iteration'],
                    failed_img['seed']
                )

                if success:
                    self.state["generated_images"].append(str(output_path))
                    successful_retries += 1
                else:
                    remaining_failures.append(failed_img)

        self.state["failed_images"] = remaining_failures
        self.save_state()

        print(f"Successfully retried {successful_retries} images")
        print(f"✗ Still failed: {len(remaining_failures)} images")
        return successful_retries

    def retry_failed_qwen_augmentations(self) -> int:
        """Retry failed Qwen augmentations from step 5."""
        if not self.state.get("failed_qwen_augmentations"):
            print("No failed Qwen augmentations to retry.")
            return 0

        print(f"\n=== RETRYING {len(self.state['failed_qwen_augmentations'])} FAILED QWEN AUGMENTATIONS ===")

        qwen_augmented_dir = self.execution_dir / "05_qwen_augmented_images"
        qwen_combined_dir = self.execution_dir / "05_qwen_combined_images"

        if not qwen_augmented_dir.exists() or not qwen_combined_dir.exists():
            print("Qwen augmentation directories not found. Run step 5 first.")
            return 0

        successful_retries = 0
        remaining_failures = []

        try:
            with ModelContextManager(self.image_editor):
                for i, failed_aug in enumerate(self.state["failed_qwen_augmentations"], 1):
                    print(f"[{i}/{len(self.state['failed_qwen_augmentations'])}] Retrying: {failed_aug['aug_filename']}")

                    aug_only_path = qwen_augmented_dir / failed_aug['aug_filename']
                    combined_path = qwen_combined_dir / failed_aug['aug_filename']

                    success = self._generate_qwen_image_with_retry(
                        failed_aug['original_image'], failed_aug['prompt'],
                        aug_only_path, combined_path,
                        Path(failed_aug['original_image']).name,
                        failed_aug['aug_index'], 1
                    )

                    if success:
                        successful_retries += 1
                    else:
                        remaining_failures.append(failed_aug)

        finally:
            self.state["failed_qwen_augmentations"] = remaining_failures
            self.save_state()

        print(f"Successfully retried {successful_retries} Qwen augmentations")
        print(f"✗ Still failed: {len(remaining_failures)}")
        return successful_retries

    # ──────────────────────────────────────────────────────────────────────
    # Main Pipeline Runner
    # ──────────────────────────────────────────────────────────────────────

    def run_pipeline(self, input_image_path: str, object_name: str, num_prompts: int,
                     num_iterations: int, system_prompt_file: str, project_info_file: str,
                     input_mask_path: str = None,
                     enable_annotation: bool = True, split_ratio: float = 0.8,
                     enable_qwen_augmentation: bool = False, predefined_prompts_file: str = None,
                     qwen_augmentation_count: int = 1,
                     enable_cv_augmentation: bool = False, cv_augmentation_count: int = 1,
                     start_from_step: int = 1):
        """
        Run the pipeline starting from a specific step.

        Step order:
          1 → White background
          2 → Perspectives
          3 → Prompts
          4 → Generate dataset
          5 → Qwen augmentation (optional, images only)
          6 → PerSAM annotation (on combined or raw images)
          7 → CV augmentation (optional, on annotated YOLO dataset)
        """
        print(f"\n🚀 Starting pipeline from step {start_from_step}")
        print(f"Execution directory: {self.execution_dir}")
        print(f"Perspective generator: {self.perspective_generator_name}")
        print(f"Retry settings: max_retries={self.max_retries}")

        if enable_qwen_augmentation:
            print("✓ Qwen generative augmentations enabled (step 5, before annotation)")
        if enable_cv_augmentation:
            print("✓ Computer Vision augmentations enabled (step 7, after annotation)")

        # Clear completion status for steps >= start_from_step
        if start_from_step > 1:
            steps_to_clear = [step for step in self.state["completed_steps"] if step >= start_from_step]
            for step in steps_to_clear:
                self.state["completed_steps"].remove(step)

            if steps_to_clear:
                print(f"Clearing completion status for steps: {steps_to_clear}")
                self.save_state()

        try:
            if start_from_step <= 1:
                self.step1_generate_white_background(input_image_path, object_name)

            if start_from_step <= 2:
                self.step2_generate_perspectives()

            if start_from_step <= 3:
                self.step3_generate_prompts(num_prompts, system_prompt_file, project_info_file)

            if start_from_step <= 4:
                self.step4_generate_dataset(num_iterations)

            # Step 5: Qwen augmentation BEFORE annotation
            if start_from_step <= 5 and enable_qwen_augmentation:
                if not predefined_prompts_file:
                    raise ValueError("predefined_prompts_file is required for step 5 (Qwen augmentation)")
                self.step5_qwen_augment_images(predefined_prompts_file, qwen_augmentation_count,
                                               enable_qwen_augmentation)

            # Step 6: PerSAM annotation (on combined images if step 5 ran, else raw)
            if start_from_step <= 6 and enable_annotation:
                if not input_mask_path:
                    raise ValueError("input_mask_path is required for step 6 (annotation)")
                self.step6_annotate_dataset(input_mask_path, split_ratio)

            # Step 7: CV augmentation on annotated YOLO dataset
            if start_from_step <= 7 and enable_cv_augmentation:
                self.step7_cv_augment_dataset(cv_augmentation_count, enable_cv_augmentation)

            print("\n🎉 Pipeline completed successfully!")
            print(f"📁 All files are in: {self.execution_dir}")

            if self.state.get("qwen_combined_images_path"):
                print(f"🔄 Qwen-combined images: {self.state['qwen_combined_images_path']}")
            if self.state.get("yolo_dataset_path"):
                print(f"📊 Annotated YOLO dataset: {self.state['yolo_dataset_path']}")
            if self.state.get("cv_augmented_dataset_path"):
                print(f"🎨 CV-augmented dataset: {self.state['cv_augmented_dataset_path']}")

            if self.state.get("failed_images"):
                print(f"⚠️  {len(self.state['failed_images'])} images failed — use --retry_failed")
            if self.state.get("failed_qwen_augmentations"):
                print(f"⚠️  {len(self.state['failed_qwen_augmentations'])} Qwen augmentations failed — use --retry_failed_qwen_augmentations")
            if self.state.get("failed_cv_augmentations"):
                print(f"⚠️  {len(self.state['failed_cv_augmentations'])} CV augmentations failed")

        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            raise

        finally:
            self.image_editor.unload_model()
            gc.collect()
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def resolve_model_paths(args) -> None:
    """Resolve model checkpoint/config paths from cache_dir when not explicitly provided."""
    cache_dir = args.cache_dir
    resolved = []

    sam_filename_map = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    # Resolve SAM checkpoint
    if args.sam_ckpt is None:
        filename = sam_filename_map.get(args.sam_type)
        if filename is None:
            print("ERROR: No known checkpoint filename for SAM type '{}'.".format(args.sam_type))
            print("  Please provide --sam_ckpt explicitly.")
            sys.exit(1)
        sam_ckpt = os.path.join(cache_dir, "checkpoints", "sam", filename)
        if not os.path.exists(sam_ckpt):
            print("ERROR: Could not resolve SAM checkpoint from cache_dir.")
            print("  Expected: {}".format(sam_ckpt))
            print("  Run: python utilities/download_models.py --cache_dir {} --sam".format(cache_dir))
            sys.exit(1)
        args.sam_ckpt = sam_ckpt
        resolved.append(("sam_ckpt", args.sam_ckpt))

    # Resolve SAM2 checkpoint
    if args.sam2_ckpt is None:
        try:
            from huggingface_hub import hf_hub_download
            try:
                args.sam2_ckpt = hf_hub_download(
                    repo_id="facebook/sam2.1-hiera-large",
                    filename="sam2.1_hiera_large.pt",
                    cache_dir=cache_dir,
                    local_files_only=True
                )
            except Exception:
                print("SAM2 checkpoint not in local cache, downloading...")
                args.sam2_ckpt = hf_hub_download(
                    repo_id="facebook/sam2.1-hiera-large",
                    filename="sam2.1_hiera_large.pt",
                    cache_dir=cache_dir
                )
            resolved.append(("sam2_ckpt", args.sam2_ckpt))
        except Exception as e:
            print("ERROR: Could not resolve SAM2 checkpoint from cache_dir: {}".format(e))
            print("  Run: python utilities/download_models.py --cache_dir {} --sam2".format(cache_dir))
            sys.exit(1)

    # Resolve SAM2 config
    if args.sam2_config is None:
        try:
            from huggingface_hub import hf_hub_download
            try:
                args.sam2_config = hf_hub_download(
                    repo_id="facebook/sam2.1-hiera-large",
                    filename="sam2.1_hiera_l.yaml",
                    cache_dir=cache_dir,
                    local_files_only=True
                )
            except Exception:
                print("SAM2 config not in local cache, downloading...")
                args.sam2_config = hf_hub_download(
                    repo_id="facebook/sam2.1-hiera-large",
                    filename="sam2.1_hiera_l.yaml",
                    cache_dir=cache_dir
                )
            resolved.append(("sam2_config", args.sam2_config))
        except Exception as e:
            print("ERROR: Could not resolve SAM2 config from cache_dir: {}".format(e))
            print("  Run: python utilities/download_models.py --cache_dir {} --sam2".format(cache_dir))
            sys.exit(1)

    if resolved:
        print("\nResolved model paths from cache_dir:")
        for name, path in resolved:
            print(f"  {name}: {path}")
        print()


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Automatic Dataset Generation Pipeline with YOLO Annotation and Decoupled Augmentations")

    # Basic parameters
    parser.add_argument("--input_image", required=True, help="Path to input image")
    parser.add_argument("--input_mask", type=str, help="Path to mask for the input image (required for annotation)")
    parser.add_argument("--object_name", required=True, help="Name of the object")
    parser.add_argument("--working_dir", default="./executions", help="Working directory")
    parser.add_argument("--cache_dir", required=True, help="Cache directory for models")

    # Execution mode
    parser.add_argument("--execution_mode", choices=["new", "last", "by_name"],
                        default="new", help="Execution mode")
    parser.add_argument("--execution_name", help="Execution name (required for by_name mode)")

    # Perspective generator
    parser.add_argument("--perspective_generator", choices=["zero123", "qwen_multicamera"],
                        default="qwen_multicamera")

    # Pipeline parameters
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--system_prompt_file", default="examples/prompts/system_qwen_edit.txt")
    parser.add_argument("--project_info_file", default="project_info.txt")

    # Image editing model
    parser.add_argument("--model_path", default="ovedrive/qwen-image-edit-4bit")

    # SAM/SAM2 model paths
    parser.add_argument("--sam_type", type=str, default="vit_h", choices=["vit_h", "vit_l", "vit_b", "vit_t"])
    parser.add_argument("--sam_ckpt", type=str, default=None)
    parser.add_argument("--sam2_config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2_ckpt", type=str, default=None)

    # PerSAM training
    parser.add_argument("--enable_persam_training", action="store_true")
    parser.add_argument("--persam_lr", type=float, default=1e-3)
    parser.add_argument("--persam_epochs", type=int, default=1000)
    parser.add_argument("--persam_log_epoch", type=int, default=200)

    # Step 5: Qwen augmentation (now BEFORE annotation)
    parser.add_argument("--enable_qwen_augmentation", action="store_true",
                        help="Enable Qwen generative augmentation (step 5, before annotation)")
    parser.add_argument("--predefined_prompts_file", type=str,
                        default="examples/prompts/qwen_edit_augmentations.txt")
    parser.add_argument("--qwen_augmentation_count", type=int, default=1)

    # Step 6: Annotation
    parser.add_argument("--enable_annotation", action="store_true", default=True)
    parser.add_argument("--split_ratio", type=float, default=0.8)

    # Step 7: CV augmentation
    parser.add_argument("--enable_cv_augmentation", action="store_true", default=True)
    parser.add_argument("--cv_augmentation_count", type=int, default=1)

    # CV augmentation probabilities and ranges
    parser.add_argument("--bw_probability", type=float, default=0.5)
    parser.add_argument("--saturation_probability", type=float, default=0.7)
    parser.add_argument("--contrast_probability", type=float, default=0.5)
    parser.add_argument("--brightness_probability", type=float, default=0.3)
    parser.add_argument("--motion_blur_probability", type=float, default=0.8)
    parser.add_argument("--compression_noise_probability", type=float, default=1.0)

    parser.add_argument("--saturation_range", type=float, nargs=2, default=[0.1, 2.5])
    parser.add_argument("--contrast_range", type=float, nargs=2, default=[0.2, 0.8])
    parser.add_argument("--brightness_range", type=float, nargs=2, default=[0.4, 1.5])
    parser.add_argument("--motion_blur_range", type=int, nargs=2, default=[20, 55])
    parser.add_argument("--compression_iterations_range", type=int, nargs=2, default=[10, 30])
    parser.add_argument("--compression_quality_range", type=int, nargs=2, default=[10, 40])

    # Step selection
    parser.add_argument("--start_from_step", type=int, default=1,
                        help="Step to start from (1-7)")

    # Retry settings
    parser.add_argument("--max_retries", type=int, default=3)

    # Special actions
    parser.add_argument("--retry_failed", action="store_true",
                        help="Retry failed images from step 4")
    parser.add_argument("--retry_failed_qwen_augmentations", action="store_true",
                        help="Retry failed Qwen augmentations from step 5")

    args = parser.parse_args()

    # Resolve model paths
    resolve_model_paths(args)

    # Validation
    if args.start_from_step < 1 or args.start_from_step > 7:
        parser.error("--start_from_step must be between 1 and 7")

    if args.execution_mode == "by_name" and not args.execution_name:
        parser.error("--execution_name is required when --execution_mode is 'by_name'")

    if args.enable_annotation and args.start_from_step <= 6:
        if not args.input_mask:
            parser.error("--input_mask is required when --enable_annotation is True")

    if args.enable_qwen_augmentation and args.start_from_step <= 5:
        if not args.predefined_prompts_file:
            parser.error("--predefined_prompts_file is required when --enable_qwen_augmentation is True")

    for range_name, range_val in [
        ("saturation_range", args.saturation_range),
        ("contrast_range", args.contrast_range),
        ("brightness_range", args.brightness_range),
        ("motion_blur_range", args.motion_blur_range),
        ("compression_iterations_range", args.compression_iterations_range),
        ("compression_quality_range", args.compression_quality_range),
    ]:
        if range_val[0] >= range_val[1]:
            parser.error(f"Invalid {range_name}: min must be less than max")

    if args.compression_quality_range[0] < 1 or args.compression_quality_range[1] > 100:
        parser.error("compression_quality_range must be between 1 and 100")

    # Initialize pipeline
    pipeline = DatasetPipeline(
        working_dir=args.working_dir,
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        max_retries=args.max_retries,
        perspective_generator=args.perspective_generator,
        sam_type=args.sam_type,
        sam_ckpt=args.sam_ckpt,
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
        enable_persam_training=args.enable_persam_training,
        persam_lr=args.persam_lr,
        persam_epochs=args.persam_epochs,
        persam_log_epoch=args.persam_log_epoch,
        enable_cv_augmentation=args.enable_cv_augmentation,
        bw_probability=args.bw_probability,
        saturation_probability=args.saturation_probability,
        contrast_probability=args.contrast_probability,
        brightness_probability=args.brightness_probability,
        motion_blur_probability=args.motion_blur_probability,
        compression_noise_probability=args.compression_noise_probability,
        saturation_range=tuple(args.saturation_range),
        contrast_range=tuple(args.contrast_range),
        brightness_range=tuple(args.brightness_range),
        motion_blur_range=tuple(args.motion_blur_range),
        compression_iterations_range=tuple(args.compression_iterations_range),
        compression_quality_range=tuple(args.compression_quality_range)
    )

    pipeline.setup_execution_dir(args.execution_mode, args.execution_name)

    # Handle special actions
    if args.retry_failed:
        pipeline.retry_failed_images()
        return

    if args.retry_failed_qwen_augmentations:
        pipeline.retry_failed_qwen_augmentations()
        return

    # Run pipeline
    pipeline.run_pipeline(
        input_image_path=args.input_image,
        object_name=args.object_name,
        num_prompts=args.num_prompts,
        num_iterations=args.num_iterations,
        system_prompt_file=args.system_prompt_file,
        project_info_file=args.project_info_file,
        input_mask_path=args.input_mask,
        enable_annotation=args.enable_annotation,
        split_ratio=args.split_ratio,
        enable_qwen_augmentation=args.enable_qwen_augmentation,
        predefined_prompts_file=args.predefined_prompts_file,
        qwen_augmentation_count=args.qwen_augmentation_count,
        enable_cv_augmentation=args.enable_cv_augmentation,
        cv_augmentation_count=args.cv_augmentation_count,
        start_from_step=args.start_from_step
    )


if __name__ == "__main__":
    main()