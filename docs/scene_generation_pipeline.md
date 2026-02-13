# Scene Generation Pipeline

Generate multi-object synthetic datasets from text descriptions using **LLM → FLUX → Grounded-SAM2**.

## Overview

```
Text descriptions ──► LLM prompts ──► FLUX images ──► Grounded-SAM2 annotations ──► YOLO dataset
     (Step 1)           (Step 1)        (Step 2)            (Step 3)
```

1. **Prompt generation** — An Ollama LLM generates diverse, photorealistic scene descriptions from a system prompt and project info file.
2. **Image synthesis** — FLUX diffusion model renders each prompt into one or more images.
3. **Auto-annotation** — Grounding-DINO detects objects and SAM2 produces precise segmentation masks, output in YOLO format.

## Quick Start

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./my_dataset \
  --session new \
  --run_prompts --run_images --run_annotations \
  --project_info_file examples/prompts/kitchen_objects.txt \
  --predefined_classes "cup,bottle,glass,plate,spoon,knife,fork,bowl" \
  --num_prompts_per_execution 10 \
  --num_random_imgs 2 \
  --cache_dir ./my_cache_dir
```

Output: `./my_dataset/YYYYMMDD/outputs/` — a ready-to-train YOLO segmentation dataset.

## Parameter Reference

### Session Management

| Parameter | Default | Description |
|---|---|---|
| `--working_dir` | *(required)* | Base directory for all sessions |
| `--session` | `new` | `new` creates a dated folder, `last` resumes most recent, or pass a custom name |
| `--list_sessions` | — | List available sessions and exit |
| `--session_status` | — | Print detailed status of the specified session and exit |

### Pipeline Control

| Parameter | Default | Description |
|---|---|---|
| `--run_prompts` | off | Run prompt generation (step 1) |
| `--run_images` | off | Run image generation (step 2) |
| `--run_annotations` | off | Run annotation (step 3) |

### Prompt Generation

| Parameter | Default | Description |
|---|---|---|
| `--model_name` | `cogito:latest` | Ollama model for prompt generation |
| `--system_prompt_file` | `examples/prompts/system.txt` | System prompt defining LLM behavior |
| `--project_info_file` | *(required for prompts)* | Project-specific context (objects, scenes, lighting) |
| `--iterative_prompts` | off | Generate prompts one at a time with conversation context (recommended for diversity) |
| `--num_prompts_per_execution` | — | Prompts per execution (required with `--iterative_prompts`) |
| `--num_executions` | `1` | Number of independent prompt conversations |
| `--predefined_classes` | — | Comma-separated class list (skips auto-generated classes) |

### Image Generation

| Parameter | Default | Description |
|---|---|---|
| `--cache_dir` | *(required)* | Model cache directory (FLUX weights + checkpoint resolution) |
| `--num_random_imgs` | `1` | Random image variations per prompt set |
| `--flux_guidance_scale` | `7.5` | FLUX classifier-free guidance scale |
| `--flux_inference_steps` | `4` | FLUX denoising steps |
| `--flux_height` | model default | Generated image height in pixels |
| `--flux_width` | model default | Generated image width in pixels |

### Annotation

| Parameter | Default | Description |
|---|---|---|
| `--sam2_ckpt` | *(from cache_dir)* | SAM2 checkpoint path |
| `--sam2_config` | *(from cache_dir)* | SAM2 config path |
| `--gdino_config` | *(from cache_dir)* | Grounding-DINO config path |
| `--gdino_ckpt` | *(from cache_dir)* | Grounding-DINO checkpoint path |
| `--box_threshold` | `0.20` | Detection confidence threshold |
| `--text_threshold` | `0.20` | Text-match confidence threshold |
| `--dataset_name` | `generated_dataset` | Name for the output YOLO dataset folder |
| `--split` | `train` | Dataset split (`train` or `val`) |

> All four model paths are resolved automatically from `--cache_dir` if not provided. Run `python utilities/download_models.py --cache_dir <dir> --all` first.

## Execution Modes

### Single Execution (default)

One pass: prompts → images → annotations.

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets --session new \
  --run_prompts --run_images --run_annotations \
  --project_info_file examples/prompts/kitchen_objects.txt \
  --cache_dir ./my_cache_dir
```

### Iterative Multi-Execution (recommended for diversity)

Runs **N** independent prompt conversations, merges all prompts, then generates images and annotations once.

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets --session new \
  --run_prompts --run_images --run_annotations \
  --iterative_prompts \
  --num_prompts_per_execution 50 \
  --num_executions 10 \
  --num_random_imgs 4 \
  --project_info_file examples/prompts/warehouse_automation.txt \
  --cache_dir ./my_cache_dir
```

This produces `10 × 50 = 500` prompts × `4` random images = **2000 images**.

### Traditional Multi-Execution

Runs **N** complete cycles of prompt generation + image generation, then merges for annotation.

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets --session new \
  --run_prompts --run_images --run_annotations \
  --num_executions 5 \
  --num_random_imgs 3 \
  --project_info_file examples/prompts/kitchen_objects.txt \
  --cache_dir ./my_cache_dir
```

## Session Management

### Directory Structure

```
working_dir/
├── 20250610/              # session "new" → today's date
│   ├── session_info.json
│   ├── prompts.json       # merged prompts
│   ├── classes.json       # merged classes
│   ├── image_paths.json   # all generated image paths
│   ├── generated_images/
│   │   └── execution_01/
│   │       ├── random_01/
│   │       └── random_02/
│   ├── outputs/           # YOLO dataset
│   │   └── generated_dataset/
│   │       ├── dataset.yaml
│   │       ├── classes.txt
│   │       └── train/
│   │           ├── images/
│   │           └── labels/
│   ├── execution_summary.json   # multi-exec mode only
│   ├── prompts_execution_01.json
│   ├── prompts_execution_02.json
│   └── ...
└── 20250610_01/           # second session on same date
```

### Resuming Sessions

```bash
# Resume the most recent session (annotations only)
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets --session last \
  --run_annotations --cache_dir ./my_cache_dir

# Resume a specific session by name
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets --session 20250610 \
  --run_images --run_annotations --cache_dir ./my_cache_dir
```

### Listing Sessions

```bash
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets --list_sessions
```

Output shows each session with status flags: `prompts`, `images`, `annotations`, and per-execution file status for multi-exec sessions.

## Partial Execution

Run any combination of steps independently:

```bash
# Prompts only
--run_prompts

# Images only (requires prompts.json in session)
--run_images

# Annotations only (requires image_paths.json and classes.json in session)
--run_annotations

# Prompts + images, annotate later
--run_prompts --run_images
```

## Class Management

### Auto-Generated Classes

By default, the LLM extracts class names from the generated prompts. These are saved to `classes.json` and used for annotation.

### Predefined Classes

Override auto-generated classes with a fixed list:

```bash
--predefined_classes "car,person,tree,building"
```

This saves the classes to `classes.json` immediately and ignores any LLM-extracted classes. Useful when you need exact control over the class vocabulary.

## Tips

**Memory:** Reduce `--num_random_imgs` or lower FLUX resolution (`--flux_height`, `--flux_width`) if running out of VRAM. The pipeline performs aggressive CUDA cache clearing between steps.

**Detection rate:** If too few objects are detected, lower `--box_threshold` and `--text_threshold` (try `0.15`). If too many false positives appear, raise them (try `0.25`–`0.30`).

**Prompt diversity:** Use `--iterative_prompts` with multiple `--num_executions` rather than a single large batch. Each conversation maintains context, producing more varied results.

**Class consistency:** The Grounded-SAM2 detector maintains global class IDs across all images. Fuzzy matching handles compound phrases (e.g., "equipment radio" → "equipment").
