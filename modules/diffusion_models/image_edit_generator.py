"""
Image Edit Generator Module

Provides an abstract base class for image editing models and a concrete
implementation using Qwen Image Edit (via diffusers).

Usage:
    from diffusion_models.image_edit_generator import QwenImageEditGenerator

    editor = QwenImageEditGenerator(model_path="ovedrive/qwen-image-edit-4bit", cache_dir="./cache")
    editor.load_model()
    editor.generate_image("make the background white", "input.png", "output.png")
    editor.unload_model()

    # Or use as context manager:
    with QwenImageEditGenerator(model_path="ovedrive/qwen-image-edit-4bit") as editor:
        editor.generate_image("make the background white", "input.png", "output.png")
"""

import gc
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image


class ImageEditGenerator(ABC):
    """Abstract base class for image editing generators."""

    @abstractmethod
    def load_model(self):
        """Load the model pipeline into memory (GPU)."""
        ...

    @abstractmethod
    def unload_model(self):
        """Free VRAM by deleting the pipeline and clearing caches."""
        ...

    @abstractmethod
    def generate_image(self, prompt: str, image_path: str, output_path: str) -> str:
        """
        Generate an edited image from a prompt and source image.

        Args:
            prompt: Text description of the desired edit.
            image_path: Path to the input image.
            output_path: Path where the output image will be saved.

        Returns:
            The output path of the generated image.
        """
        ...

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_model()
        return False


class QwenImageEditGenerator(ImageEditGenerator):
    """Image editing generator backed by QwenImageEditPipeline (diffusers)."""

    def __init__(
        self,
        model_path: str = "ovedrive/qwen-image-edit-4bit",
        cache_dir: str = None,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        negative_prompt: str = " ",
    ):
        """
        Args:
            model_path: HuggingFace model ID or local path.
            cache_dir: Directory for caching downloaded model weights.
            num_inference_steps: Number of diffusion denoising steps.
            true_cfg_scale: Classifier-free guidance scale.
            negative_prompt: Negative prompt for generation.
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        self.num_inference_steps = num_inference_steps
        self.true_cfg_scale = true_cfg_scale
        self.negative_prompt = negative_prompt
        self.pipeline = None

    def load_model(self):
        """Load QwenImageEditPipeline into memory with CPU offloading."""
        if self.pipeline is not None:
            return

        from diffusers import QwenImageEditPipeline

        print(f"Loading QwenImageEdit model from {self.model_path} ...")
        kwargs = {"torch_dtype": torch.bfloat16}
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir

        self.pipeline = QwenImageEditPipeline.from_pretrained(self.model_path, **kwargs)
        self.pipeline.enable_model_cpu_offload()
        print("QwenImageEdit model loaded (CPU offload enabled)")

    def unload_model(self):
        """Delete the pipeline and free VRAM."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("QwenImageEdit model unloaded, VRAM freed")

    def generate_image(self, prompt: str, image_path: str, output_path: str) -> str:
        """Generate an edited image.

        The model must be loaded before calling this method (via ``load_model``
        or a context manager).

        Args:
            prompt: Text description of the desired edit.
            image_path: Path to the source image.
            output_path: Path where the result will be saved.

        Returns:
            ``output_path`` after saving.
        """
        if self.pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        image = Image.open(image_path).convert("RGB")

        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=self.negative_prompt,
                image=image,
                num_inference_steps=self.num_inference_steps,
                true_cfg_scale=self.true_cfg_scale,
            ).images[0]

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        return str(output_path)
