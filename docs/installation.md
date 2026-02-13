# Installation Guide

## Prerequisites

| Requirement | Minimum |
|---|---|
| GPU | NVIDIA with CUDA support |
| VRAM | 24 GB |
| Python | 3.10+ |
| OS | Linux (tested on Ubuntu 22.04/24.04) |

## 1. Install Ollama

Ollama provides the LLM backend used for prompt generation in both pipelines.

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull cogito:latest
```

You can use any Ollama-compatible model by passing `--model_name` to the pipelines.

## 2. Create Conda Environment (optional)

```bash
conda create -n openfabrik python=3.10 -y
conda activate openfabrik
```

## 3. Clone and Install OpenFabrik

```bash
git clone https://github.com/yourusername/OpenFabrik.git
cd OpenFabrik
pip install -r requirements.txt
```

## 4. Install External Dependencies

### Grounded-SAM-2 (Scene Generation Pipeline)

```bash
git clone https://github.com/alejodosr/Grounded-SAM-2
cd Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ..
```

### Personalize-SAM (Reference Object Pipeline)

```bash
git clone https://github.com/alejodosr/Personalize-SAM
cd Personalize-SAM
pip install -e .
cd ..
```

## 5. Download Model Checkpoints

The `download_models.py` utility downloads all required model weights into a single cache directory. Running it twice will **not** re-download existing files.

```bash
# Install huggingface_hub if not already present
pip install huggingface_hub

# Download everything (~45 GB)
python utilities/download_models.py --cache_dir ./my_cache_dir --all

# Or download selectively
python utilities/download_models.py --cache_dir ./my_cache_dir --sam --sam2 --grounding_dino
python utilities/download_models.py --cache_dir ./my_cache_dir --flux
python utilities/download_models.py --cache_dir ./my_cache_dir --qwen_multicamera

# List available models and their status
python utilities/download_models.py --cache_dir ./my_cache_dir --list
```

Both pipelines resolve checkpoint paths automatically from `--cache_dir`, so you typically don't need to pass explicit paths like `--sam2_ckpt`.

## Verify Installation

```bash
# Quick smoke test — should print help without import errors
python pipelines/scene_generation_pipeline.py --help
python pipelines/reference_object_pipeline.py --help
```

## Next Steps

- [Scene Generation Pipeline](scene_generation_pipeline.md) — generate multi-object datasets from text
- [Reference Object Pipeline](reference_object_pipeline.md) — generate datasets from a single reference image
- [Configuration Files](configuration.md) — prompt files and dataset format
