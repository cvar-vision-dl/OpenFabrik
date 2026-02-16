#!/usr/bin/env python3
"""
Enhanced Dataset Generation Pipeline with Iterative Prompt Generation

A complete pipeline that supports two prompt generation strategies:
1. Batch: Traditional method asking for all prompts at once
2. Iterative: Ask for prompts one by one while maintaining conversation context

Features:
- Iterative prompt generation with conversation context
- Separate execution strategies for prompts vs images/annotations
- Automatic date-based directory management
- Modular pipeline execution
- Comprehensive logging and error handling

Usage:
    # Iterative prompts with multiple executions, then single image/annotation run
    python pipeline.py --working_dir ./datasets --session new --run_prompts --run_images --run_annotations \
        --iterative_prompts --num_prompts_per_execution 10 --num_executions 4 \
        --system_prompt_file system.txt --project_info_file project.txt --cache_dir ./models \
        --sam2_ckpt ./sam2.pt [other model files...]

    # Traditional batch mode
    python pipeline.py --working_dir ./datasets --session new --run_prompts --run_images \
        --system_prompt_file system.txt --project_info_file project.txt --cache_dir ./models

    # Continue with last session for annotations only
    python pipeline.py --working_dir ./datasets --session last --run_annotations \
        --sam2_ckpt ./sam2.pt [other model files...]

    # Use predefined classes instead of auto-generated ones
    python pipeline.py --working_dir ./datasets --session new --run_prompts --run_images --run_annotations \
        --predefined_classes "car,person,tree,building" \
        --system_prompt_file system.txt --project_info_file project.txt --cache_dir ./models
"""

import argparse
import os
import sys
import json
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import gc
import torch
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.llm_models.prompt_generator import PromptGenerator
from modules.diffusion_models.flux_image_generator import FluxImageGenerator
from modules.open_set_models.grounde_sam2_detector import GroundedSAM2Detector


def get_session_directory(base_working_dir: str, session_param: str) -> str:
    """
    Get or create the session directory based on the session parameter.

    Args:
        base_working_dir: Base directory where sessions are stored
        session_param: "new", "last", or specific directory name

    Returns:
        Path to the session directory

    Raises:
        ValueError: If session parameter is invalid or no previous sessions exist
    """
    base_path = Path(base_working_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    if session_param == "new":
        # Create new directory with today's date
        today = datetime.now().strftime("%Y%m%d")
        session_dir = base_path / today

        # If directory already exists, add a counter
        counter = 1
        original_session_dir = session_dir
        while session_dir.exists():
            session_dir = base_path / f"{today}_{counter:02d}"
            counter += 1

        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created new session directory: {session_dir}")
        return str(session_dir)

    elif session_param == "last":
        # Find the most recent date directory
        date_pattern = base_path / "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]*"
        existing_dirs = sorted(glob.glob(str(date_pattern)), reverse=True)

        if not existing_dirs:
            raise ValueError(
                f"No existing session directories found in {base_working_dir}. Use --session new to create one.")

        session_dir = existing_dirs[0]
        print(f"Using last session directory: {session_dir}")
        return session_dir

    else:
        # Use specific directory name
        session_dir = base_path / session_param

        # Create if it doesn't exist
        if not session_dir.exists():
            session_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created specified session directory: {session_dir}")
        else:
            print(f"Using existing session directory: {session_dir}")

        return str(session_dir)


def list_available_sessions(base_working_dir: str) -> None:
    """List all available session directories."""
    base_path = Path(base_working_dir)
    if not base_path.exists():
        print(f"Working directory {base_working_dir} does not exist.")
        return

    date_pattern = base_path / "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]*"
    existing_dirs = sorted(glob.glob(str(date_pattern)), reverse=True)

    if not existing_dirs:
        print(f"No session directories found in {base_working_dir}")
        return

    print(f"Available sessions in {base_working_dir}:")
    for session_dir in existing_dirs:
        session_name = Path(session_dir).name

        # Check what files exist in this session
        files_info = []
        if os.path.exists(os.path.join(session_dir, "prompts.json")):
            files_info.append("prompts")
        if os.path.exists(os.path.join(session_dir, "image_paths.json")):
            files_info.append("images")
        if os.path.exists(os.path.join(session_dir, "outputs")):
            files_info.append("annotations")

        # Check for multi-execution mode
        execution_summary_file = os.path.join(session_dir, "execution_summary.json")
        if os.path.exists(execution_summary_file):
            try:
                execution_summary = load_json(execution_summary_file)
                if execution_summary.get("multi_execution_mode", False):
                    num_executions = execution_summary.get("num_executions", 0)
                    total_images = execution_summary.get("total_images", 0)
                    files_info.append(f"multi-exec:{num_executions}×{total_images}imgs")

                    # Check for per-execution files
                    execution_files = []
                    for exec_id in range(1, num_executions + 1):
                        prompt_file = os.path.join(session_dir, f"prompts_execution_{exec_id:02d}.json")
                        class_file = os.path.join(session_dir, f"classes_execution_{exec_id:02d}.json")
                        image_file = os.path.join(session_dir, f"image_paths_execution_{exec_id:02d}.json")

                        exec_status = []
                        if os.path.exists(prompt_file):
                            exec_status.append("P")
                        if os.path.exists(class_file):
                            exec_status.append("C")
                        if os.path.exists(image_file):
                            exec_status.append("I")

                        if exec_status:
                            execution_files.append(f"{exec_id}:{''.join(exec_status)}")

                    if execution_files:
                        files_info.append(f"execs:[{','.join(execution_files)}]")
            except:
                pass

        status = f" [{', '.join(files_info)}]" if files_info else " [empty]"
        print(f"  {session_name}{status}")

    print("\nLegend: P=Prompts, C=Classes, I=Images per execution")


def check_multi_execution_session(working_dir: str) -> Dict:
    """Check if this is a multi-execution session and verify its state."""
    execution_summary_file = os.path.join(working_dir, "execution_summary.json")

    session_info = {
        "is_multi_execution": False,
        "num_executions": 0,
        "completed_executions": [],
        "missing_files": [],
        "merged_files_exist": {
            "prompts": False,
            "classes": False,
            "images": False
        }
    }

    if os.path.exists(execution_summary_file):
        try:
            execution_summary = load_json(execution_summary_file)
            if execution_summary.get("multi_execution_mode", False):
                session_info["is_multi_execution"] = True
                session_info["num_executions"] = execution_summary.get("num_executions", 0)

                # Check per-execution files
                for exec_id in range(1, session_info["num_executions"] + 1):
                    exec_files = {
                        "prompts": os.path.join(working_dir, f"prompts_execution_{exec_id:02d}.json"),
                        "classes": os.path.join(working_dir, f"classes_execution_{exec_id:02d}.json"),
                        "images": os.path.join(working_dir, f"image_paths_execution_{exec_id:02d}.json")
                    }

                    exec_status = {"execution_id": exec_id, "files": {}}
                    for file_type, file_path in exec_files.items():
                        if os.path.exists(file_path):
                            exec_status["files"][file_type] = True
                        else:
                            exec_status["files"][file_type] = False
                            session_info["missing_files"].append(f"{file_type}_execution_{exec_id:02d}.json")

                    session_info["completed_executions"].append(exec_status)

                # Check merged files
                merged_files = {
                    "prompts": os.path.join(working_dir, "prompts.json"),
                    "classes": os.path.join(working_dir, "classes.json"),
                    "images": os.path.join(working_dir, "image_paths.json")
                }

                for file_type, file_path in merged_files.items():
                    session_info["merged_files_exist"][file_type] = os.path.exists(file_path)
        except:
            pass

    return session_info


def print_session_status(working_dir: str) -> None:
    """Print detailed status of the current session."""
    session_info = check_multi_execution_session(working_dir)

    print(f"\n=== Session Status: {os.path.basename(working_dir)} ===")

    if session_info["is_multi_execution"]:
        print(f"Multi-execution session with {session_info['num_executions']} executions")

        print("\nPer-execution files:")
        for exec_info in session_info["completed_executions"]:
            exec_id = exec_info["execution_id"]
            files = exec_info["files"]
            status_symbols = {
                "prompts": "✓" if files["prompts"] else "✗",
                "classes": "✓" if files["classes"] else "✗",
                "images": "✓" if files["images"] else "✗"
            }
            print(
                f"  Execution {exec_id}: P:{status_symbols['prompts']} C:{status_symbols['classes']} I:{status_symbols['images']}")

        print("\nMerged files:")
        for file_type, exists in session_info["merged_files_exist"].items():
            symbol = "✓" if exists else "✗"
            print(f"  {file_type}.json: {symbol}")

        if session_info["missing_files"]:
            print(f"\nMissing files: {session_info['missing_files']}")
    else:
        print("Single execution session")
        # Check for standard files
        standard_files = ["prompts.json", "classes.json", "image_paths.json"]
        for filename in standard_files:
            filepath = os.path.join(working_dir, filename)
            symbol = "✓" if os.path.exists(filepath) else "✗"
            print(f"  {filename}: {symbol}")


def save_json(data, path):
    """Save data to JSON file with pretty formatting."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Successfully saved: {path}")
    except Exception as e:
        print(f"✗ Error saving {path}: {e}")


def load_json(path):
    """Load data from JSON file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        print(f"✓ Successfully loaded: {path}")
        return data
    except Exception as e:
        print(f"✗ Error loading {path}: {e}")
        return None


def save_session_info(working_dir: str, args) -> None:
    """Save information about the current session."""
    session_info = {
        "created_at": datetime.now().isoformat(),
        "args": vars(args),
        "pipeline_steps_completed": []
    }

    info_path = os.path.join(working_dir, "session_info.json")
    save_json(session_info, info_path)


def update_session_progress(working_dir: str, step: str) -> None:
    """Update the session progress with completed step."""
    info_path = os.path.join(working_dir, "session_info.json")

    if os.path.exists(info_path):
        session_info = load_json(info_path)
    else:
        session_info = {"pipeline_steps_completed": []}

    if step not in session_info["pipeline_steps_completed"]:
        session_info["pipeline_steps_completed"].append(step)
        session_info["last_updated"] = datetime.now().isoformat()
        save_json(session_info, info_path)


def merge_prompts_from_executions(working_dir: str, num_executions: int) -> List[str]:
    """Merge prompts from multiple executions."""
    all_prompts = []
    found_execution_files = []

    print(f"Looking for prompt files from {num_executions} executions in: {working_dir}")

    for execution_id in range(1, num_executions + 1):
        prompts_file = os.path.join(working_dir, f"prompts_execution_{execution_id:02d}.json")
        print(f"Checking for: {prompts_file}")

        if os.path.exists(prompts_file):
            try:
                execution_prompts = load_json(prompts_file)
                all_prompts.extend(execution_prompts)
                found_execution_files.append(execution_id)
                print(f"✓ Loaded {len(execution_prompts)} prompts from execution {execution_id}")
            except Exception as e:
                print(f"✗ Error loading prompts from execution {execution_id}: {e}")
        else:
            print(f"✗ Prompts file not found for execution {execution_id}")

    if not found_execution_files:
        print(f"WARNING: No execution prompt files found! Expected files like 'prompts_execution_01.json'")
        return []

    # Save merged prompts
    merged_prompts_file = os.path.join(working_dir, "prompts.json")
    save_json(all_prompts, merged_prompts_file)

    print(f"✓ Merged prompts from executions: {found_execution_files}")
    print(f"✓ Total prompts across all executions: {len(all_prompts)}")
    return all_prompts


def merge_classes_from_executions(working_dir: str, num_executions: int) -> List[str]:
    """Merge classes from multiple executions, removing duplicates."""
    all_classes = []
    found_execution_files = []

    print(f"Looking for class files from {num_executions} executions in: {working_dir}")

    for execution_id in range(1, num_executions + 1):
        classes_file = os.path.join(working_dir, f"classes_execution_{execution_id:02d}.json")
        print(f"Checking for: {classes_file}")

        if os.path.exists(classes_file):
            try:
                execution_classes = load_json(classes_file)
                all_classes.extend(execution_classes)
                found_execution_files.append(execution_id)
                print(f"✓ Loaded {len(execution_classes)} classes from execution {execution_id}")
                print(f"  Classes: {execution_classes}")
            except Exception as e:
                print(f"✗ Error loading classes from execution {execution_id}: {e}")
        else:
            print(f"✗ Classes file not found for execution {execution_id}")

    if not found_execution_files:
        print(f"WARNING: No execution class files found! Expected files like 'classes_execution_01.json'")
        return []

    # Remove duplicates while preserving order
    unique_classes = []
    seen = set()
    for cls in all_classes:
        if cls not in seen:
            unique_classes.append(cls)
            seen.add(cls)

    # Save merged classes
    merged_classes_file = os.path.join(working_dir, "classes.json")
    save_json(unique_classes, merged_classes_file)

    print(f"✓ Merged classes: {len(all_classes)} total -> {len(unique_classes)} unique")
    print(f"✓ Found executions: {found_execution_files}")
    print(f"✓ Final merged classes: {unique_classes}")
    return unique_classes


def collect_all_image_paths(working_dir: str, num_executions: int) -> List[str]:
    """Collect all image paths from multiple executions."""
    all_image_paths = []
    found_execution_files = []

    print(f"Looking for image path files from {num_executions} executions in: {working_dir}")

    for execution_id in range(1, num_executions + 1):
        image_paths_file = os.path.join(working_dir, f"image_paths_execution_{execution_id:02d}.json")
        print(f"Checking for: {image_paths_file}")

        if os.path.exists(image_paths_file):
            try:
                execution_images = load_json(image_paths_file)
                all_image_paths.extend(execution_images)
                found_execution_files.append(execution_id)
                print(f"✓ Loaded {len(execution_images)} images from execution {execution_id}")
            except Exception as e:
                print(f"✗ Error loading images from execution {execution_id}: {e}")
        else:
            print(f"✗ Image paths file not found for execution {execution_id}")

    if not found_execution_files:
        print(f"WARNING: No execution image files found! Expected files like 'image_paths_execution_01.json'")
        return []

    # Save consolidated image paths
    consolidated_file = os.path.join(working_dir, "image_paths.json")
    save_json(all_image_paths, consolidated_file)

    print(f"✓ Collected images from executions: {found_execution_files}")
    print(f"✓ Total images across all executions: {len(all_image_paths)}")
    return all_image_paths


def run_prompt_generation(model_name, system_prompt_file, project_info_file, working_dir,
                          execution_id=1, multi_execution_mode=False, iterative_mode=False,
                          num_prompts_per_execution=None):
    """Step 1: Generate prompts and classes using Ollama."""
    print(f"\n=== STEP 1: Generating Prompts and Classes (Execution {execution_id}) ===")

    mode_str = "Iterative" if iterative_mode else "Batch"
    print(f"Mode: {mode_str}")
    if iterative_mode and num_prompts_per_execution:
        print(f"Target: {num_prompts_per_execution} prompts")

    generator = PromptGenerator(model_name=model_name)

    if iterative_mode and num_prompts_per_execution:
        # Use iterative generation
        prompts, classes = generator.generate_prompts_iteratively(
            system_prompt_file,
            project_info_file,
            num_prompts_per_execution,
            execution_id
        )
    else:
        # Use traditional batch generation
        prompts, classes = generator.generate_prompts_and_classes(
            system_prompt_file,
            project_info_file
        )

    generator.print_results()
    generator.save_results(output_dir=working_dir)

    # Save to disk in session directory
    if multi_execution_mode:
        # Save execution-specific files for multi-execution mode
        prompts_file = os.path.join(working_dir, f"prompts_execution_{execution_id:02d}.json")
        classes_file = os.path.join(working_dir, f"classes_execution_{execution_id:02d}.json")
        save_json(prompts, prompts_file)
        save_json(classes, classes_file)
        print(f"Saved execution {execution_id} prompts and classes to {working_dir}")
    else:
        # Save standard files for single execution mode
        save_json(prompts, os.path.join(working_dir, "prompts.json"))
        save_json(classes, os.path.join(working_dir, "classes.json"))
        print(f"Saved prompts and classes to {working_dir}")

    update_session_progress(working_dir, "prompts")
    return prompts, classes


def print_memory_usage():
    """Print current memory usage statistics."""
    # System memory
    memory = psutil.virtual_memory()
    print(
        f"System Memory: {memory.percent:.1f}% used ({memory.used / (1024 ** 3):.1f}GB / {memory.total / (1024 ** 3):.1f}GB)")

    # GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory
            gpu_allocated = torch.cuda.memory_allocated(i)
            gpu_cached = torch.cuda.memory_reserved(i)
            print(f"GPU {i} Memory: Allocated {gpu_allocated / (1024 ** 3):.1f}GB, "
                  f"Cached {gpu_cached / (1024 ** 3):.1f}GB, Total {gpu_memory / (1024 ** 3):.1f}GB")


def force_cleanup():
    """Force comprehensive memory cleanup."""
    print("Performing comprehensive memory cleanup...")

    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("Memory cleanup completed")


def run_image_generation(prompts, cache_dir, working_dir, execution_id=1, num_random_imgs=1, **flux_params):
    """Step 2: Generate images using FLUX model with proper memory management."""
    print(f"\n=== STEP 2: Generating Images (Execution {execution_id}) ===")

    # Print memory before starting
    print("Memory usage before image generation:")
    print_memory_usage()

    # Use context manager for automatic cleanup
    with FluxImageGenerator(cache_dir=cache_dir) as generator:
        generator.load_model()

        # Enable VAE optimizations for better memory efficiency
        generator.enable_vae_optimizations()

        # Create images subdirectory for this execution
        base_images_dir = os.path.join(working_dir, "generated_images")
        execution_images_dir = os.path.join(base_images_dir, f"execution_{execution_id:02d}")
        os.makedirs(execution_images_dir, exist_ok=True)

        all_image_paths = []

        # Generate multiple sets of images with different random seeds
        for random_gen in range(1, num_random_imgs + 1):
            print(f"\n--- Random Generation {random_gen}/{num_random_imgs} ---")

            # Print memory usage before each generation batch
            print(f"Memory usage before batch {random_gen}:")
            print_memory_usage()

            # Create subdirectory for this random generation
            random_gen_dir = os.path.join(execution_images_dir, f"random_{random_gen:02d}")
            os.makedirs(random_gen_dir, exist_ok=True)

            # Generate images with random seed (use -1 for random)
            results = generator.batch_generate_and_save(
                prompts=prompts,
                guidance_scale=flux_params.get('guidance_scale', 7.5),
                num_inference_steps=flux_params.get('inference_steps', 4),
                height=flux_params.get('height', None),
                width=flux_params.get('width', None),
                seed=-1,  # Always use random seed for variety
                output_dir=random_gen_dir,
                show_images=False
            )

            batch_image_paths = [r[1] for r in results]
            all_image_paths.extend(batch_image_paths)
            print(f"Generated {len(batch_image_paths)} images in {random_gen_dir}")

            # Clean up the results list to free memory
            del results

            # Force cleanup between random generations
            if random_gen < num_random_imgs:  # Don't cleanup on last iteration (will be done by context manager)
                print("Cleaning up between random generations...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Print memory after generation is complete
    print("Memory usage after image generation (after cleanup):")
    print_memory_usage()

    # Save image paths for this execution
    execution_image_paths_file = os.path.join(working_dir, f"image_paths_execution_{execution_id:02d}.json")
    save_json(all_image_paths, execution_image_paths_file)

    # Save generation parameters for this execution
    generation_info = {
        "execution_id": execution_id,
        "generation_time": datetime.now().isoformat(),
        "num_prompts": len(prompts),
        "num_random_imgs": num_random_imgs,
        "total_images": len(all_image_paths),
        "flux_parameters": flux_params,
        "model_info": "FluxImageGenerator with memory optimization"
    }
    generation_info_file = os.path.join(working_dir, f"generation_info_execution_{execution_id:02d}.json")
    save_json(generation_info, generation_info_file)

    print(f"\nExecution {execution_id} complete:")
    print(f"  Generated {len(all_image_paths)} total images ({num_random_imgs} random generations)")
    print(f"  Saved to: {execution_images_dir}")

    return all_image_paths


def load_classes_for_annotation(working_dir: str) -> List[str]:
    """
    Load classes for annotation from the working directory.

    Priority:
    1. classes.json (preferred - structured format)
    2. classes.txt (fallback - parse period-separated format)

    Args:
        working_dir: Session working directory

    Returns:
        List of classes for annotation

    Raises:
        FileNotFoundError: If no class files are found
    """
    import json
    import os

    # Try to load classes.json first (preferred)
    classes_json_path = os.path.join(working_dir, "classes.json")
    if os.path.exists(classes_json_path):
        try:
            with open(classes_json_path, 'r') as f:
                classes = json.load(f)
            if isinstance(classes, list) and classes:
                print(f"✓ Loaded {len(classes)} classes from classes.json")
                print(f"  Classes: {classes}")
                return classes
            else:
                print(f"⚠ classes.json exists but contains invalid data: {type(classes)}")
        except Exception as e:
            print(f"⚠ Error loading classes.json: {e}")

    # Fallback to classes.txt (period-separated format)
    classes_txt_path = os.path.join(working_dir, "classes.txt")
    if os.path.exists(classes_txt_path):
        try:
            with open(classes_txt_path, 'r') as f:
                content = f.read().strip()
            # Parse period-separated format
            classes = [cls.strip() for cls in content.split('.') if cls.strip()]
            if classes:
                print(f"✓ Loaded {len(classes)} classes from classes.txt (period-separated)")
                print(f"  Classes: {classes}")
                return classes
            else:
                print(f"⚠ classes.txt exists but is empty or invalid")
        except Exception as e:
            print(f"⚠ Error loading classes.txt: {e}")

    # No valid class files found
    available_files = []
    for filename in ["classes.json", "classes.txt"]:
        filepath = os.path.join(working_dir, filename)
        if os.path.exists(filepath):
            available_files.append(filename)

    if available_files:
        raise FileNotFoundError(
            f"Found class files {available_files} in {working_dir} but couldn't load valid classes from any of them"
        )
    else:
        raise FileNotFoundError(
            f"No class files (classes.json or classes.txt) found in {working_dir}. "
            f"Run prompt generation first or check the working directory."
        )


def run_annotation(image_paths, classes, working_dir, dataset_name, split,
                         sam2_ckpt, sam2_config, gdino_config, gdino_ckpt,
                         box_threshold=0.20, text_threshold=0.20):
    """
    FIXED: Step 3: Annotate images using Grounding DINO + SAM2 with proper class management.
    """
    print("\n=== STEP 3: Annotating Images (FIXED VERSION) ===")

    # If classes not provided directly, load from working directory
    if not classes:
        print("No classes provided, loading from working directory...")
        classes = load_classes_for_annotation(working_dir)

    print(f"Using {len(classes)} global classes for consistent annotation:")
    print(f"Global classes: {classes}")

    # Create detector with global classes
    from modules.open_set_models.grounde_sam2_detector import GroundedSAM2Detector

    detector = GroundedSAM2Detector(
        sam2_checkpoint=sam2_ckpt,
        sam2_model_config=sam2_config,
        grounding_dino_config=gdino_config,
        grounding_dino_checkpoint=gdino_ckpt,
        global_classes=classes  # FIXED: Use global classes from prompt generation
    )

    detector.load_models()

    # Verify class mapping
    print(f"\nClass ID mapping:")
    for class_name, class_id in sorted(detector.class_to_id.items(), key=lambda x: x[1]):
        print(f"  {class_id}: {class_name}")

    # Create text prompt from classes
    text_prompt = ".".join(classes).strip(".")
    print(f"\nUsing detection prompt: {text_prompt}")

    # Create outputs directory
    outputs_dir = os.path.join(working_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    # Process each image
    successful_annotations = 0
    failed_annotations = []
    detected_classes_summary = {}

    for i, img_path in enumerate(image_paths):
        print(f"\nProcessing image {i + 1}/{len(image_paths)}: {os.path.basename(img_path)}")

        result = detector.process_and_save_all(
            image_path=img_path,
            text_prompt=text_prompt,
            output_dir=outputs_dir,
            dataset_name=dataset_name,
            split=split,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        if result['results']['boxes'].size > 0:
            successful_annotations += 1
            detected_classes = result['results']['class_names']
            detected_ids = result['results']['class_ids']

            print(f"  ✓ Detected {len(result['results']['boxes'])} objects")
            print(f"    Classes: {detected_classes}")
            print(f"    Class IDs: {detected_ids.tolist()}")

            # Track detected classes
            for class_name in detected_classes:
                detected_classes_summary[class_name] = detected_classes_summary.get(class_name, 0) + 1
        else:
            print(f"  ⚠ No objects detected")

    # Save annotation summary with class information
    annotation_summary = {
        "annotation_time": datetime.now().isoformat(),
        "total_images": len(image_paths),
        "successful_annotations": successful_annotations,
        "failed_annotations": failed_annotations,
        "global_classes": classes,
        "global_class_mapping": detector.class_to_id,
        "detected_classes_summary": detected_classes_summary,
        "detection_parameters": {
            "text_prompt": text_prompt,
            "box_threshold": box_threshold,
            "text_threshold": text_threshold
        },
        "dataset_info": {
            "name": dataset_name,
            "split": split
        },
        "class_files_info": {
            "working_dir_classes_json": os.path.join(working_dir, "classes.json"),
            "working_dir_classes_txt": os.path.join(working_dir, "classes.txt"),
            "yolo_dataset_classes_txt": os.path.join(outputs_dir, dataset_name, "classes.txt")
        }
    }

    summary_path = os.path.join(working_dir, "annotation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(annotation_summary, f, indent=2)

    # Print comprehensive summary
    summary = detector.get_summary()
    print(f"\n" + "=" * 60)
    print(f"ANNOTATION SUMMARY")
    print(f"=" * 60)
    print(f"Successfully annotated: {successful_annotations}/{len(image_paths)} images")
    if failed_annotations:
        print(f"Failed annotations: {len(failed_annotations)}")

    print(f"\nGLOBAL CLASSES ({len(classes)}):")
    for i, cls in enumerate(classes):
        count = detected_classes_summary.get(cls, 0)
        print(f"  {i}: {cls} (detected in {count} images)")

    print(f"\nCLASS FILES CREATED:")
    print(f"  Source: {os.path.join(working_dir, 'classes.json')} ({len(classes)} classes)")
    print(f"  YOLO:   {os.path.join(outputs_dir, dataset_name, 'classes.txt')} (same {len(classes)} classes)")
    print(f"  Index range: 0 to {len(classes) - 1}")

    if detected_classes_summary:
        print(f"\nDETECTED CLASSES IN DATASET:")
        for class_name, count in sorted(detected_classes_summary.items()):
            class_id = detector.class_to_id[class_name]
            print(f"  ID {class_id}: {class_name} ({count} detections)")

    print(f"\nAll annotations use consistent class IDs from the global mapping!")
    print(f"=" * 60)

    update_session_progress(working_dir, "annotations")


def resolve_model_paths(args) -> None:
    """
    Resolve model checkpoint/config paths from cache_dir when not explicitly provided.

    Uses the same directory structure created by utilities/download_models.py:
    - SAM2: HuggingFace native cache (resolved via hf_hub_download)
    - Grounding DINO: cache_dir/checkpoints/grounding_dino/

    Args:
        args: Parsed argparse namespace. Modified in-place.

    Raises:
        SystemExit: If a required model path cannot be resolved.
    """
    cache_dir = args.cache_dir
    resolved = []

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

    # Resolve Grounding DINO checkpoint
    if args.gdino_ckpt is None:
        gdino_ckpt = os.path.join(cache_dir, "checkpoints", "grounding_dino", "groundingdino_swint_ogc.pth")
        if not os.path.exists(gdino_ckpt):
            print("ERROR: Could not resolve Grounding DINO checkpoint from cache_dir.")
            print("  Expected: {}".format(gdino_ckpt))
            print("  Run: python utilities/download_models.py --cache_dir {} --grounding_dino".format(cache_dir))
            sys.exit(1)
        args.gdino_ckpt = gdino_ckpt
        resolved.append(("gdino_ckpt", args.gdino_ckpt))

    # Resolve Grounding DINO config
    if args.gdino_config is None:
        gdino_config = os.path.join(cache_dir, "checkpoints", "grounding_dino", "configs", "groundingdino_swint_ogc.py")
        if not os.path.exists(gdino_config):
            print("ERROR: Could not resolve Grounding DINO config from cache_dir.")
            print("  Expected: {}".format(gdino_config))
            print("  Run: python utilities/download_models.py --cache_dir {} --grounding_dino".format(cache_dir))
            sys.exit(1)
        args.gdino_config = gdino_config
        resolved.append(("gdino_config", args.gdino_config))

    if resolved:
        print("\nResolved model paths from cache_dir:")
        for name, path in resolved:
            print(f"  {name}: {path}")
        print()


def validate_step_requirements(args, steps_to_run: List[str]) -> bool:
    """Validate that required arguments are provided for each step."""
    if "prompts" in steps_to_run:
        if not args.system_prompt_file or not args.project_info_file:
            print("ERROR: --system_prompt_file and --project_info_file are required for prompt generation")
            return False

        if args.iterative_prompts and not args.num_prompts_per_execution:
            print("ERROR: --num_prompts_per_execution is required when using --iterative_prompts")
            return False

    if "images" in steps_to_run:
        if not args.cache_dir:
            print("ERROR: --cache_dir is required for image generation")
            return False

    if "annotations" in steps_to_run:
        required_files = [args.sam2_ckpt, args.sam2_config, args.gdino_config, args.gdino_ckpt]
        if not all(required_files):
            print(
                "ERROR: All model files (--sam2_ckpt, --sam2_config, --gdino_config, --gdino_ckpt) are required for annotation")
            return False

    # Validate execution parameters
    if args.num_executions > 1:
        if args.iterative_prompts and args.run_prompts:
            print(f"INFO: Iterative prompt mode - will run {args.num_executions} separate prompt conversations")
            if args.num_prompts_per_execution:
                total_prompts = args.num_executions * args.num_prompts_per_execution
                print(
                    f"      Expected total prompts: {total_prompts} ({args.num_executions} executions × {args.num_prompts_per_execution} prompts)")
        elif args.run_prompts and args.run_images:
            print(f"INFO: Traditional multi-execution mode - {args.num_executions} complete cycles")
        else:
            print("WARNING: --num_executions > 1 only applies when using iterative prompts or both prompts+images")

    if args.num_random_imgs > 1:
        print(f"INFO: Each prompt set will generate {args.num_random_imgs} random variations")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Modular Auto Dataset Pipeline with Iterative Prompt Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Session Management:
  --session new     : Create new session with today's date
  --session last    : Use the most recent existing session  
  --session NAME    : Use specific session directory
  --list_sessions   : Show available sessions and exit
  --session_status  : Show detailed status of specified session and exit

Prompt Generation Strategies:
  1. Batch (default): Ask for all prompts at once
  2. Iterative: Ask for prompts one by one with conversation context

Execution Strategies:
  1. Single execution: Run all steps in sequence
  2. Multi-execution (iterative prompts): Multiple prompt conversations → single image/annotation run
  3. Multi-execution (traditional): Multiple complete cycles of prompts+images

Examples:
  # Iterative prompts with 4 conversations, then single image/annotation run
  python pipeline.py --working_dir ./datasets --session new \\
    --run_prompts --run_images --run_annotations \\
    --iterative_prompts --num_prompts_per_execution 10 --num_executions 4 \\
    --system_prompt_file system.txt --project_info_file project.txt \\
    --cache_dir ./models --sam2_ckpt ./sam2.pt [other model files...]

  # Traditional batch mode
  python pipeline.py --working_dir ./datasets --session new \\
    --run_prompts --run_images --system_prompt_file system.txt \\
    --project_info_file project.txt --cache_dir ./models

  # Continue last session with annotations only
  python pipeline.py --working_dir ./datasets --session last --run_annotations \\
    --sam2_ckpt ./sam2.pt [other model files...]

  # Use predefined classes (skips auto-generated classes)
  python pipeline.py --working_dir ./datasets --session new \\
    --run_prompts --run_images --run_annotations \\
    --predefined_classes "car,person,tree,building" \\
    --system_prompt_file system.txt --project_info_file project.txt \\
    --cache_dir ./models
        """
    )

    # Session management
    parser.add_argument('--working_dir', required=True,
                        help="Base directory for all sessions")
    parser.add_argument('--session', default='new',
                        help="Session to use: 'new', 'last', or specific name (default: new)")
    parser.add_argument('--list_sessions', action='store_true',
                        help="List available sessions and exit")
    parser.add_argument('--session_status', action='store_true',
                        help="Show detailed status of the specified session and exit")

    # Pipeline control
    parser.add_argument('--run_prompts', action='store_true',
                        help="Run prompt generation step")
    parser.add_argument('--run_images', action='store_true',
                        help="Run image generation step")
    parser.add_argument('--run_annotations', action='store_true',
                        help="Run dataset annotation step")

    # Prompt generation parameters
    parser.add_argument('--model_name', default='cogito:latest',
                        help="Ollama model name for prompt generation")
    parser.add_argument('--system_prompt_file', default='examples/prompts/system.txt',
                        help="System prompt file (required for --run_prompts)")
    parser.add_argument('--project_info_file',
                        help="Project info file (required for --run_prompts)")

    # New iterative prompt parameters
    parser.add_argument('--iterative_prompts', action='store_true', default=True,
                        help="Use iterative prompt generation (ask for one prompt at a time, enabled by default)")
    parser.add_argument('--no_iterative_prompts', action='store_false', dest='iterative_prompts',
                        help="Disable iterative prompt generation and use batch mode instead")
    parser.add_argument('--num_prompts_per_execution', type=int,
                        help="Number of prompts to generate per execution (required with --iterative_prompts)")

    # Image generation parameters
    parser.add_argument('--cache_dir', required=True,
                        help="Model cache directory (used for FLUX and to resolve model paths)")
    parser.add_argument('--flux_guidance_scale', type=float, default=7.5,
                        help="FLUX guidance scale (default: 7.5)")
    parser.add_argument('--flux_inference_steps', type=int, default=4,
                        help="FLUX inference steps (default: 4)")
    parser.add_argument('--flux_height', type=int,
                        help="Generated image height (default: model default)")
    parser.add_argument('--flux_width', type=int,
                        help="Generated image width (default: model default)")
    parser.add_argument('--num_random_imgs', type=int, default=1,
                        help="Number of random image generations per prompt set (default: 1)")
    parser.add_argument('--num_executions', type=int, default=1,
                        help="Number of executions (default: 1)")

    # Annotation parameters
    parser.add_argument('--sam2_ckpt', default=None, help="SAM2 checkpoint file (resolved from cache_dir if not provided)")
    parser.add_argument('--sam2_config', default="./configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config file (resolved from cache_dir if not provided)")
    parser.add_argument('--gdino_config', default="./Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py", help="Grounding DINO config file (resolved from cache_dir if not provided)")
    parser.add_argument('--gdino_ckpt', default=None, help="Grounding DINO checkpoint file (resolved from cache_dir if not provided)")
    parser.add_argument('--predefined_classes', type=str, default=None,
                        help="Comma-separated list of classes to use instead of auto-generated ones (e.g. 'car,person,tree')")
    parser.add_argument('--dataset_name', default="generated_dataset",
                        help="Name for the final dataset (default: generated_dataset)")
    parser.add_argument('--split', default="train",
                        help="Dataset split name (default: train)")
    parser.add_argument('--box_threshold', type=float, default=0.20,
                        help="Box detection threshold (default: 0.20)")
    parser.add_argument('--text_threshold', type=float, default=0.20,
                        help="Text detection threshold (default: 0.20)")

    args = parser.parse_args()

    # Resolve model paths from cache_dir when not explicitly provided
    resolve_model_paths(args)

    # Parse predefined classes if provided
    predefined_classes = None
    if args.predefined_classes:
        predefined_classes = [c.strip() for c in args.predefined_classes.split(',') if c.strip()]
        if predefined_classes:
            print(f"\n✓ Using {len(predefined_classes)} predefined classes: {predefined_classes}")
            print("  (auto-generated classes will be ignored)")
        else:
            print("WARNING: --predefined_classes provided but no valid classes parsed. Falling back to auto-generation.")
            predefined_classes = None

    # Handle session listing
    if args.list_sessions:
        list_available_sessions(args.working_dir)
        return

    # Handle session status
    if args.session_status:
        if args.session == "new":
            print("Cannot show status for 'new' session. Specify an existing session.")
            return
        try:
            session_working_dir = get_session_directory(args.working_dir, args.session)
            print_session_status(session_working_dir)
        except ValueError as e:
            print(f"ERROR: {e}")
        return

    # Get the actual working directory for this session
    try:
        session_working_dir = get_session_directory(args.working_dir, args.session)
    except ValueError as e:
        print(f"ERROR: {e}")
        return

    # Save session info
    save_session_info(session_working_dir, args)

    print(f"Working in session directory: {session_working_dir}")

    # If predefined classes provided, save them to the session directory immediately
    if predefined_classes:
        save_json(predefined_classes, os.path.join(session_working_dir, "classes.json"))
        print(f"Saved predefined classes to session directory")

    # Validate and run pipeline steps
    steps_to_run = []
    if args.run_prompts:
        steps_to_run.append("prompts")
    if args.run_images:
        steps_to_run.append("images")
    if args.run_annotations:
        steps_to_run.append("annotations")

    if not steps_to_run:
        print("No pipeline steps specified. Use --run_prompts, --run_images, or --run_annotations")
        print("Use --list_sessions to see available sessions")
        return

    # Validate requirements for all steps
    if not validate_step_requirements(args, steps_to_run):
        return

    # Initialize variables
    prompts = None
    classes = predefined_classes  # Start with predefined classes if available
    image_paths = None

    # Determine execution strategy
    iterative_multi_execution = (args.iterative_prompts and args.run_prompts and args.num_executions > 1)
    traditional_multi_execution = (
                not args.iterative_prompts and args.run_prompts and args.run_images and args.num_executions > 1)

    if iterative_multi_execution:
        print(f"\n=== ITERATIVE MULTI-EXECUTION MODE: {args.num_executions} prompt conversations ===")
        print(f"Each conversation will generate {args.num_prompts_per_execution} prompts")
        print("After all conversations, images and annotations will run once with all collected prompts")

        # Print initial memory state
        print("\nInitial memory usage:")
        print_memory_usage()

        # Run multiple prompt-only executions
        for execution_id in range(1, args.num_executions + 1):
            print(f"\n{'=' * 60}")
            print(f"PROMPT CONVERSATION {execution_id}/{args.num_executions}")
            print(f"{'=' * 60}")

            # Generate prompts for this execution
            execution_prompts, execution_classes = run_prompt_generation(
                args.model_name,
                args.system_prompt_file,
                args.project_info_file,
                session_working_dir,
                execution_id,
                multi_execution_mode=True,
                iterative_mode=True,
                num_prompts_per_execution=args.num_prompts_per_execution
            )

            # Clear variables and cleanup between executions
            del execution_prompts, execution_classes
            force_cleanup()

        # After all prompt executions, merge results
        print(f"\n{'=' * 60}")
        print("MERGING RESULTS FROM ALL PROMPT CONVERSATIONS")
        print(f"{'=' * 60}")

        # Merge prompts from all executions
        prompts = merge_prompts_from_executions(session_working_dir, args.num_executions)

        # Use predefined classes or merge from executions
        if predefined_classes:
            classes = predefined_classes
            save_json(classes, os.path.join(session_working_dir, "classes.json"))
            print(f"✓ Using predefined classes (ignoring auto-generated): {classes}")
        else:
            classes = merge_classes_from_executions(session_working_dir, args.num_executions)

        if not prompts or not classes:
            print("ERROR: Failed to merge prompts/classes from executions.")
            return

        # Update session progress
        update_session_progress(session_working_dir, "prompts")

        # Now run image generation with all collected prompts (single execution)
        if "images" in steps_to_run:
            print(f"\n{'=' * 60}")
            print("GENERATING IMAGES WITH ALL COLLECTED PROMPTS")
            print(f"{'=' * 60}")

            flux_params = {
                'guidance_scale': args.flux_guidance_scale,
                'inference_steps': args.flux_inference_steps,
                'height': args.flux_height,
                'width': args.flux_width
            }

            image_paths = run_image_generation(
                prompts=prompts,
                cache_dir=args.cache_dir,
                working_dir=session_working_dir,
                execution_id=1,  # Single execution for images
                num_random_imgs=args.num_random_imgs,
                **flux_params
            )

            # Save in standard format for compatibility
            save_json(image_paths, os.path.join(session_working_dir, "image_paths.json"))
            update_session_progress(session_working_dir, "images")

            # Cleanup after image generation
            print("Post-image generation cleanup:")
            force_cleanup()
            print_memory_usage()

        # Save execution summary
        execution_summary = {
            "iterative_multi_execution_mode": True,
            "num_prompt_executions": args.num_executions,
            "num_prompts_per_execution": args.num_prompts_per_execution,
            "total_prompts": len(prompts) if prompts else 0,
            "unique_classes": len(classes) if classes else 0,
            "predefined_classes": predefined_classes is not None,
            "total_images": len(image_paths) if image_paths else 0,
            "completion_time": datetime.now().isoformat()
        }
        save_json(execution_summary, os.path.join(session_working_dir, "execution_summary.json"))

    elif traditional_multi_execution:
        print(f"\n=== TRADITIONAL MULTI-EXECUTION MODE: {args.num_executions} complete cycles ===")
        print(f"Each cycle will generate prompts and {args.num_random_imgs} random image sets")

        # Print initial memory state
        print("\nInitial memory usage:")
        print_memory_usage()

        # Run multiple executions of prompt + image generation
        for execution_id in range(1, args.num_executions + 1):
            print(f"\n{'=' * 60}")
            print(f"EXECUTION {execution_id}/{args.num_executions}")
            print(f"{'=' * 60}")

            # Step 1: Generate prompts for this execution
            execution_prompts, execution_classes = run_prompt_generation(
                args.model_name,
                args.system_prompt_file,
                args.project_info_file,
                session_working_dir,
                execution_id,
                multi_execution_mode=True,
                iterative_mode=args.iterative_prompts,
                num_prompts_per_execution=args.num_prompts_per_execution
            )

            # Step 2: Generate images for this execution
            flux_params = {
                'guidance_scale': args.flux_guidance_scale,
                'inference_steps': args.flux_inference_steps,
                'height': args.flux_height,
                'width': args.flux_width
            }

            execution_image_paths = run_image_generation(
                prompts=execution_prompts,
                cache_dir=args.cache_dir,
                working_dir=session_working_dir,
                execution_id=execution_id,
                num_random_imgs=args.num_random_imgs,
                **flux_params
            )

            # CRITICAL: Force comprehensive cleanup between executions
            del execution_prompts, execution_classes, execution_image_paths
            force_cleanup()

        # After all executions, merge results
        print(f"\n{'=' * 60}")
        print("MERGING RESULTS FROM ALL EXECUTIONS")
        print(f"{'=' * 60}")

        prompts = merge_prompts_from_executions(session_working_dir, args.num_executions)

        # Use predefined classes or merge from executions
        if predefined_classes:
            classes = predefined_classes
            save_json(classes, os.path.join(session_working_dir, "classes.json"))
            print(f"✓ Using predefined classes (ignoring auto-generated): {classes}")
        else:
            classes = merge_classes_from_executions(session_working_dir, args.num_executions)

        image_paths = collect_all_image_paths(session_working_dir, args.num_executions)

        update_session_progress(session_working_dir, "prompts")
        update_session_progress(session_working_dir, "images")

        # Save execution summary
        execution_summary = {
            "multi_execution_mode": True,
            "num_executions": args.num_executions,
            "num_random_imgs_per_execution": args.num_random_imgs,
            "total_images": len(image_paths) if image_paths else 0,
            "unique_classes": len(classes) if classes else 0,
            "predefined_classes": predefined_classes is not None,
            "completion_time": datetime.now().isoformat()
        }
        save_json(execution_summary, os.path.join(session_working_dir, "execution_summary.json"))

    else:
        # Single execution mode
        print(f"\n=== SINGLE EXECUTION MODE ===")
        if args.iterative_prompts:
            print(f"Using iterative prompts ({args.num_prompts_per_execution} prompts)")
        if args.num_random_imgs > 1:
            print(f"Will generate {args.num_random_imgs} random image sets")

        print("\nInitial memory usage:")
        print_memory_usage()

        # Step 1: Generate prompts
        if "prompts" in steps_to_run:
            prompts, auto_classes = run_prompt_generation(
                args.model_name,
                args.system_prompt_file,
                args.project_info_file,
                session_working_dir,
                execution_id=1,
                multi_execution_mode=False,
                iterative_mode=args.iterative_prompts,
                num_prompts_per_execution=args.num_prompts_per_execution
            )

            # Override classes with predefined if provided
            if predefined_classes:
                classes = predefined_classes
                save_json(classes, os.path.join(session_working_dir, "classes.json"))
                print(f"✓ Overriding auto-generated classes with predefined: {classes}")
            else:
                classes = auto_classes

        # Load existing data if not generating prompts
        if not prompts:
            prompts_file = os.path.join(session_working_dir, "prompts.json")
            if os.path.exists(prompts_file):
                prompts = load_json(prompts_file)

        if not classes:
            classes_file = os.path.join(session_working_dir, "classes.json")
            if os.path.exists(classes_file):
                classes = load_json(classes_file)

        # Step 2: Generate images
        if "images" in steps_to_run and prompts:
            flux_params = {
                'guidance_scale': args.flux_guidance_scale,
                'inference_steps': args.flux_inference_steps,
                'height': args.flux_height,
                'width': args.flux_width
            }

            image_paths = run_image_generation(
                prompts=prompts,
                cache_dir=args.cache_dir,
                working_dir=session_working_dir,
                execution_id=1,
                num_random_imgs=args.num_random_imgs,
                **flux_params
            )
            save_json(image_paths, os.path.join(session_working_dir, "image_paths.json"))
            update_session_progress(session_working_dir, "images")

            print("Post-image generation cleanup:")
            force_cleanup()
            print_memory_usage()

        # Load existing image paths if not generating images
        if not image_paths:
            image_paths_file = os.path.join(session_working_dir, "image_paths.json")
            if os.path.exists(image_paths_file):
                image_paths = load_json(image_paths_file)

    # Step 3: Run annotations (always single execution)
    if "annotations" in steps_to_run and image_paths and classes:
        print(f"\n{'=' * 60}")
        print("ANNOTATING ALL IMAGES")
        print(f"{'=' * 60}")

        run_annotation(
            image_paths=image_paths,
            classes=classes,
            working_dir=session_working_dir,
            dataset_name=args.dataset_name,
            split=args.split,
            sam2_ckpt=args.sam2_ckpt,
            sam2_config=args.sam2_config,
            gdino_config=args.gdino_config,
            gdino_ckpt=args.gdino_ckpt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )

    print(f"\n=== Pipeline Complete ===")
    print(f"Session directory: {session_working_dir}")

    if predefined_classes:
        print(f"Used predefined classes: {predefined_classes}")

    if iterative_multi_execution:
        print(f"Completed {args.num_executions} iterative prompt conversations")
        print(f"Total prompts generated: {len(prompts) if prompts else 0}")
        print(f"Total images generated: {len(image_paths) if image_paths else 0}")
        print(f"Unique classes found: {len(classes) if classes else 0}")
    elif traditional_multi_execution:
        print(f"Completed {args.num_executions} executions with {args.num_random_imgs} random generations each")
        print(f"Total images generated: {len(image_paths) if image_paths else 0}")
        print(f"Unique classes found: {len(classes) if classes else 0}")
    else:
        print(f"Completed steps: {steps_to_run}")
        if args.iterative_prompts:
            print(f"Used iterative prompt generation")
        if args.num_random_imgs > 1:
            print(f"Generated {args.num_random_imgs} random image sets")

    # Final memory report
    print("\nFinal memory usage:")
    print_memory_usage()


if __name__ == "__main__":
    main()