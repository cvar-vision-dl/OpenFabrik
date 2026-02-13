import os
import random
import gc
import torch
from diffusers import FluxPipeline
from typing import List, Optional, Union
from pathlib import Path
import uuid
from datetime import datetime


class FluxImageGenerator:
    """
    A class for generating images using the FLUX.1-schnell diffusion model.
    Now with proper memory management and cleanup capabilities.
    """

    def __init__(self,
                 model_path: str = "black-forest-labs/FLUX.1-schnell",
                 cache_dir: Optional[str] = None,
                 torch_dtype: torch.dtype = torch.float16,
                 enable_cpu_offload: bool = True):
        """
        Initialize the FluxImageGenerator.

        Args:
            model_path: Path to the model (HuggingFace repo or local path)
            cache_dir: Directory to cache downloaded models
            torch_dtype: PyTorch data type for the model
            enable_cpu_offload: Whether to enable CPU offloading for low VRAM
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.pipe = None
        self.is_loaded = False

        # Set up cache directory
        if cache_dir:
            os.environ['HF_HOME'] = cache_dir
        elif 'HF_HOME' not in os.environ:
            # Default cache directory
            os.environ['HF_HOME'] = os.path.expanduser('~/models/diffusion-models/')

    def load_model(self) -> None:
        """
        Load the FLUX diffusion model pipeline.

        Raises:
            RuntimeError: If model loading fails
        """
        try:
            print(f"Loading model from: {self.model_path}")

            # Clear any existing model first
            if self.is_loaded:
                print("Cleaning up existing model before loading new one...")
                self.cleanup()

            self.pipe = FluxPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                cache_dir=os.environ['HF_HOME']
            )

            # Configure memory optimizations
            self._configure_memory_optimization()

            self.is_loaded = True
            print("Model loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def cleanup(self) -> None:
        """
        Properly clean up the model and free memory.
        This should be called when done with the generator.
        """
        print("Cleaning up FluxImageGenerator...")

        if self.pipe is not None:
            try:
                # Move pipeline components to meta device to free all memory
                if hasattr(self.pipe, 'unet') and self.pipe.unet is not None:
                    try:
                        self.pipe.unet.to_empty(device='meta')
                    except Exception as e:
                        print(f"Warning: Failed to move unet to meta device: {e}")

                if hasattr(self.pipe, 'vae') and self.pipe.vae is not None:
                    try:
                        self.pipe.vae.to_empty(device='meta')
                    except Exception as e:
                        print(f"Warning: Failed to move vae to meta device: {e}")

                if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
                    try:
                        self.pipe.text_encoder.to_empty(device='meta')
                    except Exception as e:
                        print(f"Warning: Failed to move text_encoder to meta device: {e}")

                if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None:
                    try:
                        self.pipe.text_encoder_2.to_empty(device='meta')
                    except Exception as e:
                        print(f"Warning: Failed to move text_encoder_2 to meta device: {e}")

                # Try to move transformer if it exists (for newer FLUX models)
                if hasattr(self.pipe, 'transformer') and self.pipe.transformer is not None:
                    try:
                        self.pipe.transformer.to_empty(device='meta')
                    except Exception as e:
                        print(f"Warning: Failed to move transformer to meta device: {e}")

            except Exception as e:
                print(f"Warning during pipeline component cleanup: {e}")

            # Delete the pipeline
            del self.pipe
            self.pipe = None

        self.is_loaded = False

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("CUDA cache cleared")

        print("Cleanup completed")

    def __del__(self):
        """Destructor to ensure cleanup when object is deleted."""
        if self.is_loaded:
            self.cleanup()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()

    def _configure_memory_optimization(self) -> None:
        """Configure memory optimization settings for the pipeline."""
        if self.enable_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()
            print("CPU offloading enabled for memory efficiency")

    def enable_vae_optimizations(self,
                                 enable_slicing: bool = True,
                                 enable_tiling: bool = True) -> None:
        """
        Enable VAE optimizations for better memory usage and performance.

        Args:
            enable_slicing: Enable VAE slicing for memory efficiency
            enable_tiling: Enable VAE tiling for high-resolution images
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before enabling optimizations")

        if enable_slicing:
            self.pipe.vae.enable_slicing()
            print("VAE slicing enabled")

        if enable_tiling:
            self.pipe.vae.enable_tiling()
            print("VAE tiling enabled")

    def generate_image(self,
                       prompt: str,
                       guidance_scale: float = 7.5,
                       num_inference_steps: int = 4,
                       height: Optional[int] = None,
                       width: Optional[int] = None,
                       seed: Optional[int] = None,
                       **kwargs) -> object:
        """
        Generate a single image from a text prompt.

        Args:
            prompt: Text description of the desired image
            guidance_scale: How closely to follow the prompt (higher = more adherence)
            num_inference_steps: Number of denoising steps (more = better quality)
            height: Image height in pixels
            width: Image width in pixels
            seed: Random seed for reproducible results (None for random)
            **kwargs: Additional arguments passed to the pipeline

        Returns:
            PIL Image object

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before generating images. Call load_model() first.")

        # Set up generator with seed
        if seed is None:
            seed = random.randint(0, 2 ** 32 - 1)  # Random seed
        generator = torch.Generator("cpu").manual_seed(seed)

        # Prepare generation parameters
        generation_params = {
            'prompt': prompt,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_inference_steps,
            # 'generator': generator,
            **kwargs
        }

        # Add dimensions if specified
        if height is not None:
            generation_params['height'] = height
        if width is not None:
            generation_params['width'] = width

        print(f"Generating image for prompt: '{prompt[:50]}{'...' if len(prompt) > 50 else ''}'")

        # Generate the image
        result = self.pipe(**generation_params)

        # Get the image and immediately clear the result to free memory
        image = result.images[0]
        del result

        # Force cleanup of intermediate tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return image

    def generate_images(self,
                        prompts: List[str],
                        **kwargs) -> List[object]:
        """
        Generate multiple images from a list of prompts.

        Args:
            prompts: List of text descriptions
            **kwargs: Arguments passed to generate_image()

        Returns:
            List of PIL Image objects
        """
        images = []
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i + 1}/{len(prompts)}")
            image = self.generate_image(prompt, **kwargs)
            images.append(image)

            # Periodic cleanup during batch generation
            if (i + 1) % 5 == 0:  # Every 5 images
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return images

    def save_image(self,
                   image: object,
                   filename: Optional[str] = None,
                   output_dir: str = "generated_images") -> str:
        """
        Save a generated image to disk.

        Args:
            image: PIL Image object to save
            filename: Custom filename (auto-generated if None)
            output_dir: Directory to save the image

        Returns:
            Path to the saved image
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"flux_generated_{timestamp}_{unique_id}.png"

        # Ensure .png extension
        if not filename.lower().endswith('.png'):
            filename += '.png'

        filepath = os.path.join(output_dir, filename)
        image.save(filepath)
        print(f"Image saved to: {filepath}")
        return filepath

    def generate_and_save(self,
                          prompt: str,
                          filename: Optional[str] = None,
                          show_image: bool = False,
                          output_dir: str = 'generated_images',
                          **kwargs) -> tuple:
        """
        Generate an image and save it in one step.

        Args:
            prompt: Text description of the desired image
            filename: Custom filename for saving
            show_image: Whether to display the image
            **kwargs: Arguments passed to generate_image()

        Returns:
            Tuple of (image, filepath)
        """
        image = self.generate_image(prompt, **kwargs)

        if show_image:
            image.show()

        filepath = self.save_image(image, filename, output_dir)
        return image, filepath

    def batch_generate_and_save(self,
                                prompts: List[str],
                                output_dir: str = "generated_images",
                                show_images: bool = False,
                                **kwargs) -> List[tuple]:
        """
        Generate and save multiple images from prompts with memory management.

        Args:
            prompts: List of text descriptions
            output_dir: Directory to save images
            show_images: Whether to display generated images
            **kwargs: Arguments passed to generate_image()

        Returns:
            List of (image, filepath) tuples
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {i + 1}/{len(prompts)}")

            # Generate custom filename for each image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flux_batch_{timestamp}_{i + 1:03d}.png"

            image, filepath = self.generate_and_save(
                prompt=prompt,
                filename=filename,
                show_image=show_images,
                output_dir=output_dir,
                **kwargs
            )
            results.append((image, filepath))

            # Clean up the image from memory after saving (keep only the filepath)
            # Note: This modifies the return format - comment out if you need the images
            # del image
            # results[-1] = (None, filepath)  # Keep only filepath to save memory

            # Force periodic cleanup
            if (i + 1) % 3 == 0:  # Every 3 images
                print(f"Performing periodic memory cleanup...")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        print(f"\nBatch generation complete! {len(results)} images saved to {output_dir}")
        return results

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        return {
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'torch_dtype': str(self.torch_dtype),
            'cpu_offload_enabled': self.enable_cpu_offload,
            'cache_dir': os.environ.get('HF_HOME', 'Not set')
        }

# Example usage
if __name__ == '__main__':
    # Initialize the generator
    generator = FluxImageGenerator(
        cache_dir='/mnt/a/alejodosr/models/diffusion-models/'
    )

    # Load the model
    generator.load_model()

    # Optional: Enable VAE optimizations
    # generator.enable_vae_optimizations()

    # Define prompts (example prompts for testing)
    prompts = [
        "Professional photograph of modern office desk with laptop, coffee mug, and notebook in natural lighting. High resolution, shallow depth of field, minimalist composition.",
        "Industrial warehouse interior with metal shelving, cardboard boxes, and forklift in the background. Wide angle shot with dramatic lighting from overhead industrial lights.",
        "Close-up photograph of electronic circuit board showing detailed components, solder points, and microchips. Macro lens with sharp focus on central processing unit."
    ]

    # Generate and save all images
    results = generator.batch_generate_and_save(
        prompts=prompts,
        guidance_scale=7.5,
        num_inference_steps=4,
        output_dir="generated_images",
        show_images=True
    )

    # Print model info
    print("\nModel Information:")
    for key, value in generator.get_model_info().items():
        print(f"{key}: {value}")