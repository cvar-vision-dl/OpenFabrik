import random
from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import io


class ComputerVisionAugmentation:
    """Computer Vision augmentations to be applied after Kontext generation."""

    def __init__(self,
                 # Individual probabilities for each augmentation type
                 bw_probability: float = 0.5,
                 saturation_probability: float = 0.3,
                 contrast_probability: float = 0.3,
                 brightness_probability: float = 0.3,
                 motion_blur_probability: float = 0.2,
                 compression_noise_probability: float = 0.2,

                 # Augmentation parameters
                 saturation_range: Tuple[float, float] = (0.5, 1.5),
                 contrast_range: Tuple[float, float] = (0.7, 1.3),
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 motion_blur_range: Tuple[int, int] = (3, 15),

                 # Compression noise parameters
                 compression_iterations_range: Tuple[int, int] = (5, 20),
                 compression_quality_range: Tuple[int, int] = (20, 80)):
        """
        Initialize Computer Vision augmentation parameters.

        Args:
            bw_probability: Probability of converting image to black and white
            saturation_probability: Probability of applying saturation adjustment
            contrast_probability: Probability of applying contrast adjustment
            brightness_probability: Probability of applying brightness adjustment
            motion_blur_probability: Probability of applying motion blur
            compression_noise_probability: Probability of applying compression noise

            saturation_range: Range for saturation adjustment (0.5 = desaturated, 1.5 = oversaturated)
            contrast_range: Range for contrast adjustment
            brightness_range: Range for brightness adjustment
            motion_blur_range: Range for motion blur kernel size (pixels) with random direction

            compression_iterations_range: Range for number of compression iterations (5-20)
            compression_quality_range: Range for JPEG compression quality (20-80, lower = more artifacts)
        """
        # Individual probabilities
        self.bw_probability = bw_probability
        self.saturation_probability = saturation_probability
        self.contrast_probability = contrast_probability
        self.brightness_probability = brightness_probability
        self.motion_blur_probability = motion_blur_probability
        self.compression_noise_probability = compression_noise_probability

        # Augmentation parameters
        self.saturation_range = saturation_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.motion_blur_range = motion_blur_range

        # Compression noise parameters
        self.compression_iterations_range = compression_iterations_range
        self.compression_quality_range = compression_quality_range

        # Available augmentation types with their probabilities
        self.augmentation_config = {
            'bw': self.bw_probability,
            'saturation': self.saturation_probability,
            'contrast': self.contrast_probability,
            'brightness': self.brightness_probability,
            'motion_blur': self.motion_blur_probability,
            'compression_noise': self.compression_noise_probability
        }

    def apply_bw_conversion(self, image: Image.Image) -> Image.Image:
        """Convert image to black and white while keeping 3 channels."""
        grayscale = image.convert('L')
        # Convert back to RGB to maintain 3 channels
        bw_image = Image.merge('RGB', (grayscale, grayscale, grayscale))
        return bw_image

    def apply_saturation_adjustment(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Apply saturation adjustment."""
        factor = random.uniform(*self.saturation_range)
        enhancer = ImageEnhance.Color(image)
        adjusted_image = enhancer.enhance(factor)
        description = f"saturation_{factor:.2f}"
        return adjusted_image, description

    def apply_contrast_adjustment(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Apply contrast adjustment."""
        factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(image)
        adjusted_image = enhancer.enhance(factor)
        description = f"contrast_{factor:.2f}"
        return adjusted_image, description

    def apply_brightness_adjustment(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Apply brightness adjustment."""
        factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(image)
        adjusted_image = enhancer.enhance(factor)
        description = f"brightness_{factor:.2f}"
        return adjusted_image, description

    def apply_motion_blur(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """Apply motion blur with random kernel size and direction."""
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Random motion blur parameters
        kernel_size = random.randint(*self.motion_blur_range)
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Random angle in degrees (0-360)
        angle = random.uniform(0, 360)

        # Create motion blur kernel with random direction
        kernel = np.zeros((kernel_size, kernel_size))

        # Calculate the line coordinates for the motion blur direction
        center = kernel_size // 2

        # Convert angle to radians
        angle_rad = np.radians(angle)

        # Calculate end points of the line
        length = kernel_size // 2
        dx = int(length * np.cos(angle_rad))
        dy = int(length * np.sin(angle_rad))

        # Draw line from center-length to center+length
        cv2.line(kernel,
                 (center - dx, center - dy),
                 (center + dx, center + dy),
                 1, 1)

        # Normalize the kernel (avoid division by zero)
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel = kernel / kernel_sum
        else:
            # Fallback: create a simple horizontal line if line drawing failed
            kernel[center, :] = 1
            kernel = kernel / kernel_size

        # Apply motion blur
        blurred = cv2.filter2D(opencv_image, -1, kernel)

        # Convert back to PIL
        blurred_pil = Image.fromarray(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))

        description = f"motion_blur_{kernel_size}_angle_{angle:.0f}"
        return blurred_pil, description

    def apply_compression_noise(self, image: Image.Image) -> Tuple[Image.Image, str]:
        """
        Apply compression noise by compressing the image N times with varying quality.

        Args:
            image: Input PIL Image

        Returns:
            Tuple of (compressed_image, description)
        """
        # Random number of compression iterations
        iterations = random.randint(*self.compression_iterations_range)

        # Starting image
        compressed_image = image.copy()

        quality_values = []

        for i in range(iterations):
            # Random quality for this iteration
            quality = random.randint(*self.compression_quality_range)
            quality_values.append(quality)

            # Save to bytes buffer with JPEG compression
            buffer = io.BytesIO()
            compressed_image.save(buffer, format='JPEG', quality=quality, optimize=True)

            # Load back from buffer
            buffer.seek(0)
            compressed_image = Image.open(buffer).copy()
            buffer.close()

        # Create description
        avg_quality = sum(quality_values) / len(quality_values)
        description = f"compression_noise_{iterations}x_avgq{avg_quality:.0f}"

        return compressed_image, description

    def apply_cascade_augmentations(self, image: Image.Image) -> Tuple[Image.Image, List[str]]:
        """
        Apply augmentations based on individual probabilities.
        Each augmentation type is checked independently against its probability.

        Args:
            image: Input PIL Image

        Returns:
            Tuple of (augmented_image, list_of_applied_augmentations)
        """
        augmented_image = image.copy()
        applied_descriptions = []

        # Check each augmentation type independently based on its probability
        for aug_type, probability in self.augmentation_config.items():
            if random.random() < probability:

                if aug_type == 'bw':
                    augmented_image = self.apply_bw_conversion(augmented_image)
                    applied_descriptions.append('bw_conversion')

                elif aug_type == 'saturation':
                    augmented_image, description = self.apply_saturation_adjustment(augmented_image)
                    applied_descriptions.append(description)

                elif aug_type == 'contrast':
                    augmented_image, description = self.apply_contrast_adjustment(augmented_image)
                    applied_descriptions.append(description)

                elif aug_type == 'brightness':
                    augmented_image, description = self.apply_brightness_adjustment(augmented_image)
                    applied_descriptions.append(description)

                elif aug_type == 'motion_blur':
                    augmented_image, description = self.apply_motion_blur(augmented_image)
                    applied_descriptions.append(description)

                elif aug_type == 'compression_noise':
                    augmented_image, description = self.apply_compression_noise(augmented_image)
                    applied_descriptions.append(description)

        return augmented_image, applied_descriptions

    def get_augmentation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current augmentation configuration.

        Returns:
            Dictionary with configuration details
        """
        return {
            "augmentation_probabilities": self.augmentation_config.copy(),
            "parameter_ranges": {
                "saturation_range": self.saturation_range,
                "contrast_range": self.contrast_range,
                "brightness_range": self.brightness_range,
                "motion_blur_range": self.motion_blur_range,
                "compression_iterations_range": self.compression_iterations_range,
                "compression_quality_range": self.compression_quality_range
            }
        }