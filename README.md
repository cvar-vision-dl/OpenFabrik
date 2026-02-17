<div align="center">

# 🏭 OpenFabrik

**Bootstrap Your Computer Vision Models Without Data or Annotations**

*Open-source synthetic data generation for object detection and segmentation*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick Start](#quick-start) • [Pipelines](#pipelines) • [Examples](#examples) • [Documentation](#documentation) • [Community](#community)

</div>

---

## 🎯 The Problem

Training computer vision models requires **thousands of labeled images**. Data gathering and manual annotation is expensive, time-consuming, and becomes a bottleneck for rapid prototyping and experimentation.

## ✨ The Solution

**OpenFabrik** automatically generates unlimited synthetic training data with perfect annotations. No dataset collection, no manual labeling, no waiting weeks for annotators.

Just describe what you want to detect, and OpenFabrik generates fully annotated datasets ready for training YOLOv8, YOLOv10, or any modern detection/segmentation model.


https://github.com/user-attachments/assets/66e1eea5-9428-415f-ae29-b195a38f72e5


---

## 📋 Use Cases

<table>
<tr>
<td width="33%" valign="top">

### 🔬 ML Research
Rapidly prototype new architectures without waiting for dataset collection and annotation

</td>
<td width="33%" valign="top">

### 🏭 Industrial Vision
Bootstrap models for manufacturing QA, defect detection, and inventory management

</td>
<td width="33%" valign="top">

### 🤖 Robotics
Generate multi-perspective datasets for robot manipulation and navigation tasks

</td>
</tr>
</table>

### How Does OpenFabrik Compare?

<table>
<tr>
<th></th>
<th align="center">OpenFabrik</th>
<th align="center">3D Rendering<br><sub>(Omniverse, BlenderProc, Kubric, Gazebo)</sub></th>
<th align="center">Diffusion Academic<br><sub>(GeoDiffusion, InstaGen, DatasetDM)</sub></th>
<th align="center">Commercial<br><sub>(Synthesis AI, Anyverse, Datagen)</sub></th>
</tr>
<tr>
<td><b>Input</b></td>
<td align="center">Text or 1 photo</td>
<td align="center">3D assets + scenes</td>
<td align="center">Text + layouts</td>
<td align="center">3D assets (provided)</td>
</tr>
<tr>
<td><b>3D Assets Required</b></td>
<td align="center">No</td>
<td align="center">Yes</td>
<td align="center">No</td>
<td align="center">Yes</td>
</tr>
<tr>
<td><b>Auto-Annotation</b></td>
<td align="center">Open-set (any class)</td>
<td align="center">From 3D scene</td>
<td align="center">Partial / custom</td>
<td align="center">Built-in</td>
</tr>
<tr>
<td><b>Output Format</b></td>
<td align="center">YOLO (bbox, seg)</td>
<td align="center">COCO, KITTI, custom</td>
<td align="center">Custom</td>
<td align="center">COCO, custom</td>
</tr>
<tr>
<td><b>Multi-View</b></td>
<td align="center">Yes (6 views)</td>
<td align="center">Yes</td>
<td align="center">No</td>
<td align="center">Yes</td>
</tr>
<tr>
<td><b>Runs Locally</b></td>
<td align="center">Yes</td>
<td align="center">Yes</td>
<td align="center">Yes</td>
<td align="center">No (cloud)</td>
</tr>
<tr>
<td><b>End-to-End</b></td>
<td align="center">Yes</td>
<td align="center">Yes</td>
<td align="center">Partial</td>
<td align="center">Yes</td>
</tr>
<tr>
<td><b>Open Source</b></td>
<td align="center">Yes</td>
<td align="center">Mixed</td>
<td align="center">Yes</td>
<td align="center">No</td>
</tr>
</table>

**Why OpenFabrik?** No 3D assets needed, open-set detection for any object class, production-ready YOLO output, and fully local and open-source — all in one end-to-end pipeline.

<details>
<summary><b>See detailed tool-by-tool comparison</b></summary>
<br>

**3D Rendering Frameworks** — Require pre-built 3D assets and scenes:
[Omniverse Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html) |
[BlenderProc](https://github.com/DLR-RM/BlenderProc) |
[Infinigen](https://github.com/princeton-vl/infinigen) |
[Kubric](https://github.com/google-research/kubric) |
[Unreal](https://www.unrealengine.com/) / [Unity](https://unity.com/) / [Gazebo](https://gazebosim.org/)

**Diffusion-Based Methods** — Academic research, typically generation-only or partial pipelines:
[GeoDiffusion](https://github.com/KaiChen1998/GeoDiffusion) (ICLR 2024) |
[DiffusionEngine](https://www.sciencedirect.com/science/article/abs/pii/S0031320325008015) (2024) |
[InstaGen](https://arxiv.org/abs/2402.05937) (CVPR 2024) |
[DatasetDM](https://github.com/showlab/DatasetDM) (NeurIPS 2023) |
[X-Paste](https://arxiv.org/abs/2212.03863) (ICML 2023)

**Commercial Platforms** — Cloud-based, require enterprise pricing:
[Synthesis AI](https://synthesis.ai/) |
[Anyverse](https://anyverse.ai/) |
[Rendered.ai](https://rendered.ai/) |
[Datagen](https://datagen.tech/)

</details>

---

## 🚀 Key Features

- **🎨 Zero Data Required** - Generate datasets from text descriptions or reference images
- **🏷️ Zero Manual Labeling** - Perfect annotations generated automatically
- **🔄 Unlimited Diversity** - Generate infinite variations with different backgrounds, lighting, and perspectives
- **🎯 Open-Set Detection** - Detect any object class via text prompts (powered by Grounding-DINO + SAM2)
- **📐 Multi-View Generation** - Create datasets from multiple viewing angles automatically
- **⚡ Production Ready** - YOLO-compatible output, ROS2 integration, TensorRT export
- **🌍 Fully Open Source** - Built on state-of-the-art open models (FLUX, SAM2, Qwen Multicamera, Zero123++)

---

<a id="quick-start"></a>

## ⚡ Quick Start

### Prerequisites

**Note:** A GPU with CUDA capabilities and <ins>>=24GB VRAM required</ins>.

```bash
# Install Ollama (for LLM-based prompt generation)
curl -fsSL https://ollama.com/install.sh | sh

# Pull an LLM model
ollama pull cogito:latest

# Conda env creation [optional]
conda create -n openfabrik python=3.10 -y
conda activate openfabrik

# Download all pipeline models into your cache_dir (up to 45GB)
pip install huggingface_hub
git clone https://github.com/cvar-vision-dl/OpenFabrik
cd OpenFabrik
python utilities/download_models.py --cache_dir ./my_cache_dir --all
```

### Installation

```bash
# Conda env creation [optional]
conda activate openfabrik
cd OpenFabrik

# Install dependencies
pip install -r requirements.txt

# Clone Grounded Sam 2 repository
git clone https://github.com/alejodosr/Grounded-SAM-2
cd Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ..

# Clone PerSam repository
git clone https://github.com/alejodosr/Personalize-SAM
cd Personalize-SAM
pip install -e .
cd ..
```

### Generate Your First Dataset (3 minutes)

```bash
# Scene Generation Pipeline - multi-object detection dataset
python pipelines/scene_generation_pipeline.py \
  --working_dir ./my_dataset \
  --session new \
  --run_prompts --run_images --run_annotations \
  --project_info_file examples/prompts/kitchen_objects.txt \
  --predefined_classes "cup,bottle,glass,plate,spoon,knife,fork,bowl" \
  --num_prompts_per_execution 10 \
  --num_random_imgs 2 \
  --cache_dir ./my_cache_dir

# Output: YOLO-format dataset at ./my_dataset/YYYYMMDD/outputs/
```

✅ **Done!** Your dataset is ready to train with any YOLO from ultalytics:

```bash
yolo train data=./my_dataset/YYYYMMDD/outputs/dataset.yaml model=yolov8n.pt epochs=100
```

---

<a id="pipelines"></a>

## 🏗️ Pipelines

OpenFabrik provides two specialized pipelines for different use cases:

### 1️⃣ Scene Generation Pipeline

**Best for:** Multi-object detection, general-purpose datasets, rapid prototyping

**How it works:**
1. 📝 **LLM generates prompts** - Describe your target objects and scenes
2. 🎨 **FLUX creates synthetic images** - State-of-the-art diffusion model generates diverse scenes
3. 🏷️ **Auto-annotation** - Grounding-DINO detects objects, SAM2 generates precise masks

**Output:** YOLO segmentation dataset with bounding boxes and masks


https://github.com/user-attachments/assets/3ec126b9-f90e-490b-8087-f24dddf15bf2


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

**Key Features:**
- Supports batch or iterative prompt generation
- Multiple executions with automatic result merging
- Configurable image variations per prompt
- Session persistence - resume from any step

---

### 2️⃣ Reference Object Pipeline

**Best for:** Custom objects, multi-view datasets, robotics manipulation

**How it works:**
1. 📸 **Start with one reference image** - Upload a photo of your target object
2. 🔄 **Generate multiple perspectives** - Qwen Multicamera (default) or Zero123++ creates multi-view representations
3. 🌍 **Augment contexts** - Place object in diverse environments and lighting
4. 🏷️ **Auto-annotation** - PerSAM + SAM2 for reference-based segmentation

**Output:** YOLO segmentation dataset with multi-perspective annotations

https://github.com/user-attachments/assets/0b40a002-055f-4909-999a-2d0fbe5fd97f


For this pipeline, you need a **reference image** and a **reference mask for that image**. If you don't have a reference mask, you can generate it with this utility:

```bash
# Generate mask for image
python utilities/sam_mask_labeler.py \
--image ./examples/pikachu_bag.jpg \
--cache_dir ./my_cache_dir \
--outputu_dir ./my_output_folder
```

In case you already have the mask for the reference image, just proceed to the pipeline:

```bash
python pipelines/reference_object_pipeline.py \
  --input_image ./examples/pikachu_bag.jpg \
  --input_mask ./examples/pikachu_bag_mask.png \
  --project_info_file ./examples/prompts/project_pikachu.txt \
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

**Key Features:**
- Multi-perspective 3D-aware generation
- Reference-based segmentation (no class labels needed)
- Decoupled augmentation strategies:
  - **Generative augmentation**: Change lighting, context, occlusions
  - **CV augmentation**: Motion blur, compression, B/W, contrast
- Robust retry mechanisms with automatic server restart

---

## 📊 Pipeline Comparison

| Feature | Scene Generation | Reference Object |
|---------|-----------------|------------------|
| **Input** | Text descriptions | Single reference image |
| **Best For** | Multi-object detection | Custom single-object |
| **Perspectives** | Single view | Multi-view (configurable) |
| **Generation** | FLUX only | FLUX + Qwen Multicamera / Zero123++ |
| **Annotation** | Grounded-SAM2 | PerSAM + SAM2 |
| **Augmentation** | None (built into generation) | Generative + CV |
| **Use Case** | General datasets | Robotics, specialized objects |

---

<a id="examples"></a>

## 🎬 Examples

### Example 1: Industrial Parts Detection

```bash
# Generate dataset for factory automation
python pipelines/scene_generation_pipeline.py \
  --predefined_classes bolt nut washer screw gear \
  --num_prompts_per_execution 100 \
  --num_random_imgs 5 \
  --working_dir ./datasets/industrial_parts \
  --cache_dir ./my_cache_dir \
  --session new --run_prompts --run_images --run_annotations
```

### Example 2: Custom Product Recognition

```bash
# Train model to recognize your specific product from all angles
python pipelines/reference_object_pipeline.py \
  --input_image ./my_product_photo.jpg \
  --input_mask ./my_product_mask.png \
  --object_name "my_product" \
  --working_dir ./datasets/product_detection \
  --enable_annotation \
  --enable_qwen_augmentation \
  --enable_cv_augmentation \
  --cache_dir ./my_cache_dir
```

### Example 3: Resume from Last Session

```bash
# Resume previous run (annotations only)
python pipelines/scene_generation_pipeline.py \
  --working_dir ./datasets/office_objects \
  --session last \
  --run_annotations \
  --cache_dir ./my_cache_dir
```

---

## 🛠️ Utilities

OpenFabrik includes comprehensive utilities for the entire ML pipeline:

### YOLO Training & Export

```bash
# Train model
python utilities/yolo_scripts/yolo_training.py \
  --dataset ./datasets/my_dataset/dataset.yaml \
  --model yolov8n-seg.pt \
  --epochs 100

# Export to ONNX
python utilities/yolo_scripts/yolo_export_onnx.py \
  --model ./runs/train/weights/best.pt

# Export to TensorRT (for production deployment)
python utilities/yolo_scripts/yolo_export_tensorrt.py \
  --model ./runs/train/weights/best.pt
```

### Dataset Management

```bash
# Get dataset statistics
python utilities/yolo_scripts/statistics_yolo_dataset.py \
  --dataset ./datasets/my_dataset/dataset.yaml

# Split dataset into train/val
python utilities/yolo_scripts/yolo_split_dataset.py \
  --dataset ./raw_dataset \
  --output ./split_dataset \
  --split 0.8
```

### ROS2 Real-Time Inference

```bash
# Publish detections to ROS2 topics
python utilities/ros2_scripts/yolo_segmentation_publisher.py \
  --model ./weights/best.pt \
  --input-topic /camera/image_raw \
  --output-topic /detections/segmentation \
  --tensorrt  # Use TensorRT for faster inference
```

For high-performance C++ ROS2 YOLO inference, see [yolo-ros2-inference](https://github.com/cvar-vision-dl/yolo-ros2-inference).

---

## 🧠 Architecture

OpenFabrik follows a modular, pipeline-based architecture:

```
┌─────────────────────────────────────────────────────────┐
│                  PIPELINES                              │
├─────────────────────────────────────────────────────────┤
│  Scene Generation  │  Reference Object                  │
│  (multi-object)    │  (single object, multi-view)       │
└─────────────────────────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
┌──────────────┐ ┌────────────┐ ┌──────────────┐
│  Generation  │ │ Annotation │ │ Augmentation │
│   Modules    │ │  Modules   │ │   Modules    │
├──────────────┤ ├────────────┤ ├──────────────┤
│ • LLM       │ │ • Grounding│ │ • Generative │
│   Prompts   │ │   DINO     │ │   (Qwen Edit)│
│ • FLUX      │ │ • SAM2     │ │ • CV Augment │
│ • Qwen      │ │ • PerSAM   │ │              │
│   Multicam  │ │            │ │              │
│ • Zero123++ │ │            │ │              │
└──────────────┘ └────────────┘ └──────────────┘
                     │
        ┌────────────▼────────────┐
        │   YOLO Dataset Output   │
        │  (ready for training)   │
        └─────────────────────────┘
```

---

<a id="documentation"></a>

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Scene Generation Pipeline](docs/scene_generation_pipeline.md)
- [Reference Object Pipeline](docs/reference_object_pipeline.md)
- [Configuration Files](docs/configuration.md)

---

## 🎓 How It Works

OpenFabrik combines state-of-the-art models into automated pipelines:

### Scene Generation Pipeline

1. **LLM Prompt Generation** (Ollama)
   - Generates diverse scene descriptions
   - Supports iterative or batch mode
   - Maintains conversation context for diversity

2. **Image Synthesis** (FLUX Diffusion)
   - State-of-the-art text-to-image generation
   - Configurable resolution and quality
   - Memory-optimized for batch processing

3. **Auto-Annotation** (Grounding-DINO + SAM2)
   - Open-set object detection (any class via text)
   - Precise segmentation masks
   - Global class consistency across dataset

### Reference Object Pipeline

1. **White Background Generation** (Qwen Edit)
   - Clean object isolation
   - Optimized for multi-view generation

2. **Multi-Perspective Generation** (Qwen Multicamera / Zero123++)
   - Qwen Multicamera (default): generates configurable view/elevation/distance combinations using a LoRA adapter
   - Zero123++ (alternative): 6 viewing angles with 3D-aware transformations
   - Consistent object appearance across perspectives

3. **Context Generation** (LLM + FLUX)
   - Places object in diverse environments
   - Multiple iterations per perspective
   - Configurable scene complexity

4. **Reference-Based Annotation** (PerSAM + SAM2)
   - Segments object across all generated images
   - No class labels needed
   - High precision with reference guidance

5. **Augmentation** (Generative + CV)
   - **Generative**: Lighting, occlusions, weather
   - **CV**: Motion blur, compression, B/W, contrast

---

## 📝 Citation

If you use OpenFabrik in your research, please cite:
```bibtex
@misc{openfabrik2025,
    author = {Rodriguez-Ramos, Alejandro and Campoy, Pascual},
    title = {OpenFabrik: Bootstrap Your Computer Vision Models Without Data or Annotations},
    howpublished = "\url{https://github.com/cvar-vision-dl/OpenFabrik}",
    doi = {10.5281/zenodo.18669083},
    year = {2025}
}
```

---

<a id="community"></a>

## 🤝 Community

### Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Related Work

OpenFabrik builds on these excellent projects:
- [FLUX](https://github.com/black-forest-labs/flux) - Text-to-image diffusion
- [Qwen Multicamera](https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA) - Multi-view generation (default)
- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus) - Multi-view generation (alternative)
- [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) - Open-set detection
- [SAM2](https://github.com/facebookresearch/sam2) - Segmentation
- [PerSAM](https://github.com/ZrrSkywalker/Personalize-SAM) - Reference segmentation
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - Object detection framework
- [YOLO ROS2 Inference](https://github.com/cvar-vision-dl/yolo-ros2-inference) - Real-time ROS2 C++ YOLO inference

### Acknowledgments

Special thanks to:
- The open-source computer vision community
- All contributors and users providing feedback

---

## 📄 License

OpenFabrik is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🌟 Star History

If OpenFabrik helps your project, consider giving it a star! ⭐

---

<div align="center">

**Built with ❤️ by Computer Vision and Aerial Robotics (CVAR) for the open-source community**

[Report Bug](https://github.com/yourusername/OpenFabrik/issues) • [Request Feature](https://github.com/yourusername/OpenFabrik/issues) • [Discussions](https://github.com/yourusername/OpenFabrik/discussions)

</div>
