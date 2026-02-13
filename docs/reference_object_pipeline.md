# Reference Object Pipeline

Generate multi-perspective datasets from a **single reference image** using Qwen Image Edit → Multi-Camera / Zero123++ → FLUX → PerSAM + SAM2 annotation → augmentations.

## Overview

The pipeline runs up to 7 sequential steps:

| Step | Name | What it does |
|------|------|---|
| 1 | White background | Isolates the object onto a clean white background using Qwen Image Edit |
| 2 | Perspectives | Generates multiple viewing angles (Qwen Multi-Camera or Zero123++) |
| 3 | Prompts | LLM generates diverse scene descriptions |
| 4 | Dataset generation | Places each perspective into each scene via Qwen Image Edit |
| 5 | Qwen augmentation *(optional)* | Applies generative augmentations (lighting, weather, context) — images only, no labels |
| 6 | Annotation | PerSAM + SAM2 segments the object across all images → YOLO dataset |
| 7 | CV augmentation *(optional)* | Traditional augmentations (B/W, blur, compression) on the annotated dataset |

Steps 5 and 7 are **decoupled**: generative augmentations run before annotation so every image gets its own accurate mask; CV augmentations run after annotation because they don't change geometry, so labels are copied verbatim.

## Quick Start

First, generate a mask for your reference image (skip if you already have one):

```bash
python utilities/sam_mask_labeler.py \
  --image ./examples/pikachu_bag.jpg \
  --cache_dir ./my_cache_dir \
  --output_dir ./my_output_folder
```

Then run the pipeline:

```bash
python pipelines/reference_object_pipeline.py \
  --input_image ./examples/pikachu_bag.jpg \
  --input_mask ./examples/pikachu_bag_mask.png \
  --project_info_file examples/prompts/project_pikachu.txt \
  --object_name "pikachu bag" \
  --num_prompts 10 \
  --num_iterations 1 \
  --working_dir ./datasets/my_product \
  --enable_annotation \
  --enable_qwen_augmentation \
  --qwen_augmentation_count 2 \
  --enable_cv_augmentation \
  --cv_augmentation_count 2 \
  --cache_dir ./my_cache_dir
```

## Parameter Reference

### Basic

| Parameter | Default | Description |
|---|---|---|
| `--input_image` | *(required)* | Path to reference image of the object |
| `--input_mask` | — | Mask for the reference image (required when annotation is enabled) |
| `--object_name` | *(required)* | Name of the object (used for labeling and prompts) |
| `--working_dir` | `./executions` | Base directory for execution outputs |
| `--cache_dir` | *(required)* | Model cache directory |

### Execution Mode

| Parameter | Default | Description |
|---|---|---|
| `--execution_mode` | `new` | `new`, `last`, or `by_name` |
| `--execution_name` | — | Required when `--execution_mode by_name` |
| `--start_from_step` | `1` | Start pipeline from this step (1–7) |

### Perspective Generation

| Parameter | Default | Description |
|---|---|---|
| `--perspective_generator` | `qwen_multicamera` | `qwen_multicamera` or `zero123` |

**Qwen Multi-Camera** generates configurable view/elevation/distance combinations using a LoRA adapter. **Zero123++** generates 6 fixed viewing angles with 3D-aware transformations.

### Scene Generation (Steps 3–4)

| Parameter | Default | Description |
|---|---|---|
| `--num_prompts` | `5` | Number of scene prompts to generate |
| `--num_iterations` | `3` | Times each perspective × prompt combination is rendered |
| `--system_prompt_file` | `examples/prompts/system_qwen_edit.txt` | System prompt for LLM |
| `--project_info_file` | `project_info.txt` | Project-specific context |
| `--model_path` | `ovedrive/qwen-image-edit-4bit` | Qwen Image Edit model |

### Qwen Generative Augmentation (Step 5)

| Parameter | Default | Description |
|---|---|---|
| `--enable_qwen_augmentation` | off | Enable generative augmentation |
| `--predefined_prompts_file` | `examples/prompts/qwen_edit_augmentations.txt` | One augmentation prompt per line |
| `--qwen_augmentation_count` | `1` | Augmented versions per original image |

### Annotation (Step 6)

| Parameter | Default | Description |
|---|---|---|
| `--enable_annotation` | on | Enable PerSAM + SAM2 annotation |
| `--split_ratio` | `0.8` | Train/val split ratio |
| `--sam_type` | `vit_h` | SAM backbone (`vit_h`, `vit_l`, `vit_b`) |
| `--sam_ckpt` | *(from cache_dir)* | SAM checkpoint |
| `--sam2_config` | *(from cache_dir)* | SAM2 config |
| `--sam2_ckpt` | *(from cache_dir)* | SAM2 checkpoint |
| `--enable_persam_training` | off | Fine-tune PerSAM on the reference |
| `--persam_lr` | `1e-3` | PerSAM learning rate |
| `--persam_epochs` | `1000` | PerSAM training epochs |
| `--persam_log_epoch` | `200` | PerSAM log interval |

### CV Augmentation (Step 7)

| Parameter | Default | Description |
|---|---|---|
| `--enable_cv_augmentation` | on | Enable traditional CV augmentations |
| `--cv_augmentation_count` | `1` | Augmented versions per annotated image |

**Probability parameters** — each augmentation is applied independently with these probabilities:

| Parameter | Default | Augmentation |
|---|---|---|
| `--bw_probability` | `0.5` | Black & white conversion |
| `--saturation_probability` | `0.7` | Saturation shift |
| `--contrast_probability` | `0.5` | Contrast adjustment |
| `--brightness_probability` | `0.3` | Brightness adjustment |
| `--motion_blur_probability` | `0.8` | Motion blur kernel |
| `--compression_noise_probability` | `1.0` | JPEG compression noise |

**Range parameters** — control the intensity of each augmentation:

| Parameter | Default | Unit |
|---|---|---|
| `--saturation_range` | `0.1 2.5` | Multiplier (1.0 = unchanged) |
| `--contrast_range` | `0.2 0.8` | Multiplier |
| `--brightness_range` | `0.4 1.5` | Multiplier |
| `--motion_blur_range` | `20 55` | Kernel size in pixels |
| `--compression_iterations_range` | `10 30` | Number of re-compression passes |
| `--compression_quality_range` | `10 40` | JPEG quality (1–100) |

### Retry & Recovery

| Parameter | Default | Description |
|---|---|---|
| `--max_retries` | `3` | Retries per image generation attempt |
| `--retry_failed` | — | Re-run failed step 4 image generations and exit |
| `--retry_failed_qwen_augmentations` | — | Re-run failed step 5 augmentations and exit |

## Step-by-Step Breakdown

### Step 1 — White Background

Sends the reference image through Qwen Image Edit with the prompt: *"`<object_name>` with white background, while maintaining object's features and shape"*.

**Input:** `--input_image`
**Output:** `01_white_background.png`

### Step 2 — Perspectives

Generates multiple viewing angles from the white-background image.

- **Qwen Multi-Camera** (default): produces views with configurable azimuth, elevation, and distance via a LoRA adapter.
- **Zero123++**: produces 6 canonical views with 3D-aware transformations.

**Output:** `02_perspectives/perspective_*.png`

### Step 3 — Prompts

The Ollama LLM generates `--num_prompts` scene descriptions using iterative conversation mode.

**Output:** `03_prompts/`

### Step 4 — Dataset Generation

Each perspective is combined with each prompt, rendered `--num_iterations` times with random seeds.

Total images = perspectives × prompts × iterations.

**Output:** `04_dataset/*.png`

### Step 5 — Qwen Augmentation (optional)

Each image from step 4 is edited with a randomly chosen prompt from `--predefined_prompts_file`. Produces `--qwen_augmentation_count` variants per image.

Creates two directories:
- `05_qwen_augmented_images/` — augmented images only
- `05_qwen_combined_images/` — originals + augmented (used as input for annotation)

### Step 6 — Annotation

PerSAM + SAM2 segments the object in every image using the original reference image and mask as guidance. Outputs a YOLO segmentation dataset with train/val split.

If step 5 ran, annotates the combined directory; otherwise annotates step 4 output.

**Output:** `06_yolo_dataset/`

### Step 7 — CV Augmentation (optional)

Applies cascaded traditional augmentations to the annotated YOLO dataset. Labels are copied verbatim since these augmentations don't change object geometry.

Creates two directories:
- `07_cv_augmented_dataset/` — augmented images + labels only
- `07_cv_combined_dataset/` — originals + augmented (final dataset)

## Using `--start_from_step`

Resume or re-run from any step. The pipeline loads state from the previous run and clears completion status for the requested step and all later steps.

```bash
# Re-run from annotation (steps 1–5 complete)
python pipelines/reference_object_pipeline.py \
  --input_image ./my_object.png \
  --input_mask ./my_object_mask.png \
  --object_name "my_object" \
  --working_dir ./datasets/my_product \
  --execution_mode last \
  --start_from_step 6 \
  --enable_annotation \
  --cache_dir ./my_cache_dir

# Re-run from CV augmentation only (steps 1–6 complete)
python pipelines/reference_object_pipeline.py \
  --input_image ./my_object.png \
  --object_name "my_object" \
  --working_dir ./datasets/my_product \
  --execution_mode last \
  --start_from_step 7 \
  --enable_cv_augmentation \
  --cv_augmentation_count 3 \
  --cache_dir ./my_cache_dir
```

### Step Dependencies

```
Step 1 (white bg) ──► Step 2 (perspectives) ──► Step 3 (prompts) ──► Step 4 (dataset)
                                                                          │
                                                            ┌─────────────┤
                                                            ▼             ▼
                                                    Step 5 (Qwen aug)  Step 6 (annotation) ◄─ Step 5 output (if ran)
                                                                          │
                                                                          ▼
                                                                    Step 7 (CV aug)
```

## Session / Execution Modes

| Mode | Flag | Behavior |
|---|---|---|
| New | `--execution_mode new` | Creates a timestamped directory (`YYYYMMDD_HHMMSS`) |
| Last | `--execution_mode last` | Resumes the most recently modified execution |
| By name | `--execution_mode by_name --execution_name <name>` | Resumes a specific named execution |

State is persisted to `pipeline_state.json` after every step, including lists of generated images, failed operations, and completed step numbers.

## Retry Mechanisms

### Retry Failed Images (Step 4)

```bash
python pipelines/reference_object_pipeline.py \
  --input_image ./my_object.png \
  --object_name "my_object" \
  --working_dir ./datasets/my_product \
  --execution_mode last \
  --retry_failed \
  --cache_dir ./my_cache_dir
```

### Retry Failed Qwen Augmentations (Step 5)

```bash
python pipelines/reference_object_pipeline.py \
  --input_image ./my_object.png \
  --object_name "my_object" \
  --working_dir ./datasets/my_product \
  --execution_mode last \
  --retry_failed_qwen_augmentations \
  --cache_dir ./my_cache_dir
```

Both retry modes load the failed operation list from `pipeline_state.json`, re-attempt each, and update the state file.

## Output Directory Structure

```
working_dir/
└── 20250610_143022/
    ├── pipeline_state.json
    ├── 01_white_background.png
    ├── 02_perspectives/
    │   ├── perspective_00_front__eye_level.png
    │   ├── perspective_01_side_left__eye_level.png
    │   └── ...
    ├── 03_prompts/
    ├── 04_dataset/
    │   ├── perspective_0_prompt_0_seed_12345.png
    │   ├── metadata.json
    │   └── ...
    ├── 05_qwen_augmented_images/     # if Qwen augmentation enabled
    ├── 05_qwen_combined_images/      # originals + Qwen augmented
    ├── 06_yolo_dataset/
    │   ├── dataset.yaml
    │   ├── classes.txt
    │   ├── train/
    │   │   ├── images/
    │   │   └── labels/
    │   └── val/
    │       ├── images/
    │       └── labels/
    ├── 07_cv_augmented_dataset/      # if CV augmentation enabled
    └── 07_cv_combined_dataset/       # originals + CV augmented
        ├── dataset.yaml
        ├── train/
        └── val/
```

## Tips

**Memory:** The pipeline uses `ModelContextManager` to load/unload models between steps with aggressive CUDA cleanup. If you still hit OOM, reduce `--num_iterations` or switch from `vit_h` to `vit_l` for SAM.

**Augmentation tuning:** Start with default probabilities and review the augmentation summary JSONs generated in each output directory. Adjust probabilities and ranges based on your domain — for outdoor robotics you may want higher motion blur probability; for well-lit environments, lower brightness range.

**Perspective generator:** Qwen Multi-Camera is the default and generally produces higher quality results. Zero123++ is faster and may work better for objects with strong symmetry.

**Predefined prompts:** The file at `examples/prompts/qwen_edit_augmentations.txt` contains lighting and weather variations. Create your own domain-specific prompts file for better augmentation diversity.
