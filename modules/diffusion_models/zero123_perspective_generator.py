import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import gc
from typing import List, Optional, Tuple, Union


class Zero123Plus:
    """
    Zero123Plus pipeline wrapper for generating multi-view images from a single input image.

    Supports both v1.1 and v1.2 models with automatic view extraction.
    """

    def __init__(self, version: str = "v1.2", cache_dir: str = "/mnt/a/alejodosr/models/huggingface/"):
        """
        Initialize Zero123Plus class.

        Args:
            version: Model version ("v1.1" or "v1.2")
            cache_dir: Directory to cache downloaded models
        """
        self.version = version
        self.cache_dir = cache_dir
        self.pipeline = None
        self.device = None

        # Fixed viewing angles for each version
        self.azimuths = [30, 90, 150, 210, 270, 330]
        if version == "v1.1":
            self.elevations = [30, -20, 30, -20, 30, -20]
        elif version == "v1.2":
            self.elevations = [20, -10, 20, -10, 20, -10]
        else:
            raise ValueError("Version must be 'v1.1' or 'v1.2'")

    def load_model(self, device: str = "cuda:0") -> None:
        """
        Load the Zero123Plus model.

        Args:
            device: Device to load the model on (default: "cuda:0")
        """
        if self.pipeline is not None:
            print(f"Model already loaded on {self.device}")
            return

        print(f"Loading Zero123Plus {self.version}...")

        # Load the pipeline
        model_name = f"sudo-ai/zero123plus-{self.version}"
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=torch.float16,
            cache_dir=self.cache_dir
        )

        # Configure scheduler
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )

        # Move to device
        self.device = device
        self.pipeline.to(device)

        print(f"Zero123Plus {self.version} loaded successfully on {device}")

    def unload_model(self) -> None:
        """
        Unload the model and free GPU memory.
        """
        if self.pipeline is None:
            print("No model to unload")
            return

        print(f"Unloading Zero123Plus {self.version}...")

        # Move to CPU and delete
        self.pipeline.to("cpu")
        del self.pipeline
        self.pipeline = None
        self.device = None

        gc.collect()

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("Model unloaded successfully")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.pipeline is not None

    def _extract_subimages(self, combined_image: Image.Image) -> Tuple[List[Image.Image], List[dict]]:
        """
        Extract individual subimages from the combined output.

        Args:
            combined_image: Combined image containing all 6 views

        Returns:
            Tuple of (subimages, view_info)
        """
        # Extract subimages using the official demo method
        side_len = combined_image.width // 2
        subimages = [
            combined_image.crop((x, y, x + side_len, y + side_len))
            for y in range(0, combined_image.height, side_len)
            for x in range(0, combined_image.width, side_len)
        ]

        # Create view information
        view_info = []
        for i in range(len(subimages)):
            view_info.append({
                'index': i,
                'azimuth': self.azimuths[i],
                'elevation': self.elevations[i],
                'version': self.version,
                'generator': 'zero123plus'
            })

        return subimages, view_info

    def generate_views(
            self,
            input_image: Union[str, Image.Image],
            num_inference_steps: int = 75,
            guidance_scale: float = 4.0,
            seed: Optional[int] = None
    ) -> Tuple[List[Image.Image], List[dict]]:
        """
        Generate multi-view images from a single input image.

        Args:
            input_image: Input image (file path or PIL Image)
            num_inference_steps: Number of denoising steps (default: 75)
            guidance_scale: Guidance scale for generation (default: 4.0)
            seed: Random seed for reproducible results

        Returns:
            Tuple of (individual_views, view_info)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load image if path is provided
        if isinstance(input_image, str):
            input_image = Image.open(input_image)
        elif not isinstance(input_image, Image.Image):
            raise TypeError("input_image must be a file path or PIL Image")

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)

        print(f"Generating views with {num_inference_steps} steps...")

        # Generate the multiview image
        result = self.pipeline(
            input_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )

        # Get combined image
        combined_image = result.images[0]

        # Store combined image for optional access
        self._last_combined_image = combined_image

        # Extract individual views
        subimages, view_info = self._extract_subimages(combined_image)

        print(f"Generated {len(subimages)} individual views using Zero123Plus {self.version}")

        return subimages, view_info

    def get_last_combined_image(self) -> Optional[Image.Image]:
        """
        Get the last combined image from generate_views().

        Returns:
            Combined image or None if no generation has been done
        """
        return getattr(self, '_last_combined_image', None)

    def save_views(
            self,
            subimages: List[Image.Image],
            view_info: List[dict],
            output_dir: str = "./outputs",
            prefix: str = ""
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
            filename = f"{prefix}view_{i:02d}_az{info['azimuth']:03d}_el{info['elevation']:+03d}.png"
            filepath = os.path.join(output_dir, filename)
            subimage.save(filepath)
            saved_files.append(filepath)
            print(f"Saved {filepath} - Azimuth: {info['azimuth']}°, Elevation: {info['elevation']}°")

        return saved_files

    def get_view_by_angle(
            self,
            subimages: List[Image.Image],
            view_info: List[dict],
            target_azimuth: int,
            target_elevation: Optional[int] = None
    ) -> Tuple[Image.Image, dict]:
        """
        Get the view closest to target angles.

        Args:
            subimages: List of individual view images
            view_info: List of view information dictionaries
            target_azimuth: Target azimuth angle
            target_elevation: Target elevation angle (optional)

        Returns:
            Tuple of (closest_view, view_info)
        """
        best_match = None
        min_diff = float('inf')

        for i, info in enumerate(view_info):
            az_diff = abs(info['azimuth'] - target_azimuth)

            if target_elevation is not None:
                el_diff = abs(info['elevation'] - target_elevation)
                total_diff = az_diff + el_diff
            else:
                total_diff = az_diff

            if total_diff < min_diff:
                min_diff = total_diff
                best_match = i

        return subimages[best_match], view_info[best_match]

    def print_view_info(self, view_info: List[dict]) -> None:
        """Print information about all views."""
        print(f"\nView angles for Zero123Plus {self.version}:")
        print("-" * 50)
        for info in view_info:
            print(f"View {info['index']}: Azimuth={info['azimuth']:3d}°, Elevation={info['elevation']:+3d}°")


# Example usage
if __name__ == "__main__":
    # Create Zero123Plus instance
    zero123 = Zero123Plus(version="v1.2")

    try:
        # Load the model
        zero123.load_model()

        # Generate views from an image
        views, info = zero123.generate_views(
            "/home/alejodosr/Downloads/a2rlgate_object.jpg",
            num_inference_steps=100,
            # seed=42
        )

        # Print view information
        zero123.print_view_info(info)

        # Save all views
        saved_files = zero123.save_views(views, info, output_dir="./outputs", prefix="v12_")

        # Access specific views
        front_view = views[0]  # Azimuth: 30°, Elevation: 20°
        right_view = views[1]  # Azimuth: 90°, Elevation: -10°
        left_view = views[4]  # Azimuth: 270°, Elevation: 20°

        # Get a specific view by angle
        side_view, side_info = zero123.get_view_by_angle(views, info, target_azimuth=90)
        print(f"\nFound side view: {side_info}")

        # Save combined image (optional)
        combined = zero123.get_last_combined_image()
        if combined:
            combined.save("./outputs/combined_all_views.png")

    finally:
        # Always unload the model to free memory
        zero123.unload_model()