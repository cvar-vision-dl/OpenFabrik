# Configuration Files

OpenFabrik pipelines are configured through prompt text files and YAML dataset files. This guide covers each type.

## System Prompt File

Defines the LLM's role and behavior when generating image prompts. Passed via `--system_prompt_file`.

**Two variants are provided:**

| File | Used by | Purpose |
|---|---|---|
| `examples/prompts/system.txt` | Scene Generation Pipeline | Guides FLUX prompt generation for multi-object scenes |
| `examples/prompts/system_qwen_edit.txt` | Reference Object Pipeline | Guides Qwen Image Edit prompt generation for object-in-context scenes |

### Writing a System Prompt

A good system prompt should:
- Define the LLM's role (e.g., "expert at generating photorealistic image prompts")
- Set realism expectations (not artistic or dreamy)
- List required prompt elements (camera settings, lighting, angles, textures)
- Emphasize diversity across the generated set

**Example** (`examples/prompts/system.txt`):

```text
You are an expert at generating diverse, realistic image prompts for synthetic
data generation using FLUX diffusion models.

Key Guidelines:
1. Be extremely detailed and specific about objects, environments, and camera settings
2. Focus on realistic, practical scenarios (not artistic or dreamy scenes)
3. Include technical camera details (focal length, aperture, ISO, angle)
4. Vary lighting conditions (natural, artificial, mixed, harsh, soft)
5. Include different perspectives and viewing angles
...
```

## Project Info File

Provides domain-specific context about the objects and scenes you want to generate. Passed via `--project_info_file`.

### Writing a Project Info File

Structure your file with these sections:

1. **Project name** — one-line summary
2. **Target environment** — where the objects appear
3. **Objects to detect** — categorized list of object classes
4. **Scene characteristics** — surface types, states, lighting conditions
5. **Camera specifications** — angles, focal lengths, depth of field
6. **Important considerations** — realism details, variability, domain-specific needs

**Example** (`examples/prompts/kitchen_objects.txt`):

```text
Project: Kitchen Tabletop Object Detection and Segmentation

Target Environment:
Kitchen table/counter surfaces, dining areas, food preparation zones

Objects to Detect:
Dishware & Utensils:
- Plates (dinner plates, side plates, bowls)
- Cups and mugs (coffee mugs, tea cups, glasses)
- Cutlery (forks, knives, spoons)
...

Scene Characteristics:
Surface Types:
- Wooden tables (light oak, dark walnut)
- Laminate countertops
...

Camera Specifications:
Angles & Perspectives:
- Overhead/bird's eye (90° downward)
- 45° angle (typical robot arm camera)
...
```

The LLM uses this context to generate varied, domain-appropriate prompts. More detail = better prompt diversity.

## Predefined Prompts File (Qwen Augmentation)

Used by the Reference Object Pipeline's step 5 (generative augmentation). Each line is a short augmentation directive applied randomly to images. Passed via `--predefined_prompts_file`.

**Example** (`examples/prompts/qwen_edit_augmentations.txt`):

```text
in dim lighting with soft shadows
under bright fluorescent overhead lights
with harsh natural sunlight from the side
in overcast diffused lighting
backlit by strong light source
with dramatic side lighting
in foggy conditions
in rainy conditions with rain streaks
with dust particles in the air
```

### Writing Your Own

- One prompt per line, no blank lines needed
- Keep prompts short and focused on a single variation
- Cover: lighting, weather, atmosphere, camera effects, context changes
- Avoid prompts that significantly alter object shape/position (the mask must still apply)

## YOLO Dataset Format

All pipeline outputs follow the standard YOLO directory layout:

```
dataset_name/
├── dataset.yaml
├── classes.txt
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

### dataset.yaml

```yaml
path: /absolute/path/to/dataset_name
train: train/images
val: val/images
names:
  0: cup
  1: bottle
  2: plate
```

### classes.txt

One class name per line, order matches the ID in `dataset.yaml`:

```text
cup
bottle
plate
```

### Label Format

Each `.txt` label file in `labels/` contains one line per detected object:

**Segmentation (polygon):**
```
class_id x1 y1 x2 y2 ... xn yn
```
All coordinates are normalized to [0, 1].

**Bounding box:**
```
class_id x_center y_center width height
```

**Keypoints:**
```
class_id x y w h kp1_x kp1_y kp1_v ... kpn_x kpn_y kpn_v
```
Visibility: `0` = not labeled, `1` = labeled but invisible, `2` = labeled and visible.

## Example Prompt Files

The `examples/prompts/` directory contains ready-to-use templates:

| File | Pipeline | Purpose |
|---|---|---|
| `system.txt` | Scene Generation | General-purpose system prompt for FLUX |
| `system_qwen_edit.txt` | Reference Object | System prompt for Qwen Image Edit |
| `kitchen_objects.txt` | Scene Generation | Kitchen tabletop detection project |
| `warehouse_automation.txt` | Scene Generation | Warehouse/logistics project |
| `project_pikachu.txt` | Reference Object | Example reference object project |
| `qwen_edit_augmentations.txt` | Reference Object | Lighting/weather augmentation prompts |
