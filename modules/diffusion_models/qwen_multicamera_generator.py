import torch
from PIL import Image
from diffusers import DiffusionPipeline
import gc
from itertools import product
from typing import List, Optional, Tuple, Union


class QwenMultiCameraGenerator:
    """
    Qwen Multi-Camera perspective generator using the Qwen-Image-Edit model
    with a multi-angle LoRA adapter.

    Generates individual perspective images from a single input image using
    configurable view, elevation, and distance combinations.
    """

    # Default view/elevation/distance presets
    DEFAULT_VIEWS = [
        "front view",
        "front-right quarter view",
        "right side view",
        # "back-right quarter view",
        "back view",
        # "back-left quarter view",
        "left side view",
        "front-left quarter view",
    ]

    DEFAULT_ELEVATIONS = [
        # "low-angle shot",
        # "eye-level shot",
        "elevated shot",
        # "high-angle shot",
    ]

    DEFAULT_DISTANCE = "medium shot"

    def __init__(
        self,
        model_path: str = "ovedrive/Qwen-Image-Edit-2511-4bit",
        lora_path: str = "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
        cache_dir: str = "/mnt/a/alejodosr/models/huggingface/",
        views: Optional[List[str]] = None,
        elevations: Optional[List[str]] = None,
        distance: Optional[str] = None,
    ):
        """
        Initialize QwenMultiCameraGenerator.

        Args:
            model_path: HuggingFace model ID or local path for the base model
            lora_path: HuggingFace model ID or local path for the multi-angle LoRA
            cache_dir: Directory to cache downloaded models
            views: List of view angle strings (default: 6 standard views)
            elevations: List of elevation strings (default: ["elevated shot"])
            distance: Distance string (default: "medium shot")
        """
        self.model_path = model_path
        self.lora_path = lora_path
        self.cache_dir = cache_dir
        self.pipeline = None
        self.device = None

        # Configure view/elevation/distance combinations
        self.views = views if views is not None else list(self.DEFAULT_VIEWS)
        self.elevations = elevations if elevations is not None else list(self.DEFAULT_ELEVATIONS)
        self.distance = distance if distance is not None else self.DEFAULT_DISTANCE

        # Build prompt combinations
        self._build_prompts()

    def _build_prompts(self) -> None:
        """Build all prompt combinations from views × elevations × distance."""
        self.prompt_combinations = [
            {
                "view": view,
                "elevation": elev,
                "distance": self.distance,
                "prompt": f"<sks> {view} {elev} {self.distance}",
            }
            for view, elev in product(self.views, self.elevations)
        ]

    def load_model(self, device: str = "cuda:0") -> None:
        """
        Load the Qwen multi-camera model with LoRA weights.

        Args:
            device: Device to load the model on (default: "cuda:0")
        """
        if self.pipeline is not None:
            print(f"Model already loaded on {self.device}")
            return

        print(f"Loading Qwen Multi-Camera model...")
        print(f"  Base model: {self.model_path}")
        print(f"  LoRA: {self.lora_path}")

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
        )

        self.pipeline.load_lora_weights(self.lora_path, cache_dir=self.cache_dir)

        self.device = device
        self.pipeline.to(device)

        print(f"Qwen Multi-Camera model loaded successfully on {device}")
        print(f"  Configured: {len(self.prompt_combinations)} perspective combinations")

    def unload_model(self) -> None:
        """Unload the model and free GPU memory."""
        if self.pipeline is None:
            print("No model to unload")
            return

        print("Unloading Qwen Multi-Camera model...")

        self.pipeline.to("cpu")
        del self.pipeline
        self.pipeline = None
        self.device = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Model unloaded successfully")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.pipeline is not None

    def generate_views(
        self,
        input_image: Union[str, Image.Image],
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        seed: Optional[int] = None,
        negative_prompt: str = "",
    ) -> Tuple[List[Image.Image], List[dict]]:
        """
        Generate multi-view images from a single input image.

        Args:
            input_image: Input image (file path or PIL Image)
            num_inference_steps: Number of denoising steps (default: 20)
            true_cfg_scale: True CFG scale for generation (default: 4.0)
            seed: Random seed for reproducible results
            negative_prompt: Negative prompt for generation

        Returns:
            Tuple of (individual_views, view_info)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image if path is provided
        if isinstance(input_image, str):
            input_image = Image.open(input_image).convert("RGB")
        elif isinstance(input_image, Image.Image):
            input_image = input_image.convert("RGB")
        else:
            raise TypeError("input_image must be a file path or PIL Image")

        print(f"Generating {len(self.prompt_combinations)} perspectives with {num_inference_steps} steps...")

        subimages = []
        view_info = []

        for i, combo in enumerate(self.prompt_combinations):
            prompt = combo["prompt"]
            print(f"  [{i + 1}/{len(self.prompt_combinations)}] {prompt}")

            # Set seed if provided (reset per image for reproducibility)
            generator = torch.manual_seed(seed) if seed is not None else None

            result = self.pipeline(
                image=input_image,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                true_cfg_scale=true_cfg_scale,
                negative_prompt=negative_prompt,
            )

            subimages.append(result.images[0])
            view_info.append(
                {
                    "index": i,
                    "view": combo["view"],
                    "elevation": combo["elevation"],
                    "distance": combo["distance"],
                    "prompt": prompt,
                    "generator": "qwen_multicamera",
                }
            )

            # # Free intermediate tensors between views to avoid OOM
            # # The model stays loaded — only cached/temporary allocations are released
            # del result
            # gc.collect()
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

        print(f"Generated {len(subimages)} individual views using Qwen Multi-Camera")
        return subimages, view_info

    def save_views(
        self,
        subimages: List[Image.Image],
        view_info: List[dict],
        output_dir: str = "./outputs",
        prefix: str = "",
    ) -> List[str]:
        """
        Save individual views to files.

        Args:
            subimages: List of individual view images
            view_info: List of view information dictionaries
            output_dir: Output directory
            prefix: Filename prefix

        Returns:
            List of saved filenames
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        for i, (subimage, info) in enumerate(zip(subimages, view_info)):
            view_clean = info["view"].replace(" ", "_").replace("-", "_")
            elev_clean = info["elevation"].replace(" ", "_").replace("-", "_")
            filename = f"{prefix}view_{i:02d}_{view_clean}__{elev_clean}.png"
            filepath = os.path.join(output_dir, filename)
            subimage.save(filepath)
            saved_files.append(filepath)
            print(f"Saved {filepath} - View: {info['view']}, Elevation: {info['elevation']}")

        return saved_files

    def get_view_by_name(
        self,
        subimages: List[Image.Image],
        view_info: List[dict],
        target_view: str,
        target_elevation: Optional[str] = None,
    ) -> Tuple[Image.Image, dict]:
        """
        Get the view matching the target view name.

        Args:
            subimages: List of individual view images
            view_info: List of view information dictionaries
            target_view: Target view name (e.g., "front view", "right side view")
            target_elevation: Target elevation name (optional)

        Returns:
            Tuple of (matching_view, view_info)
        """
        target_view_lower = target_view.lower()

        for i, info in enumerate(view_info):
            if info["view"].lower() == target_view_lower:
                if target_elevation is None or info["elevation"].lower() == target_elevation.lower():
                    return subimages[i], view_info[i]

        raise ValueError(
            f"No matching view found for view='{target_view}', elevation='{target_elevation}'"
        )

    def print_view_info(self, view_info: List[dict]) -> None:
        """Print information about all views."""
        print(f"\nView configurations for Qwen Multi-Camera:")
        print("-" * 60)
        for info in view_info:
            print(
                f"  View {info['index']}: {info['view']} | "
                f"{info['elevation']} | {info['distance']}"
            )


# Example usage
if __name__ == "__main__":
    generator = QwenMultiCameraGenerator(
        cache_dir="/mnt/a/alejodosr/models/openfabrik_cachedir/"
    )

    try:
        generator.load_model()

        views, info = generator.generate_views(
            "/home/alejodosr/Downloads/a2rlgate_object.jpg",
            num_inference_steps=20,
            seed=1069426278,
        )

        generator.print_view_info(info)

        saved_files = generator.save_views(
            views, info, output_dir="./outputs", prefix="qwen_mc_"
        )

        # Access specific view by name
        front, front_info = generator.get_view_by_name(views, info, "front view")
        print(f"\nFound front view: {front_info}")

    finally:
        generator.unload_model()