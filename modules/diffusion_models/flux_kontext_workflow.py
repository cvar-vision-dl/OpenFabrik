import requests
import json
import time
import io
from PIL import Image
import base64
import uuid
import os
import subprocess
import signal
from typing import Optional
import numpy as np

class ComfyUIAPI:
    def __init__(self, server_address: str = "127.0.0.1:8188"):
        """
        Initialize ComfyUI API client

        Args:
            server_address: ComfyUI server address (default: "127.0.0.1:8188")
        """
        self.server_address = server_address
        self.base_url = f"http://{server_address}"
        self.server_process = None

    def start_server(self, wait_for_ready: bool = True, timeout: int = 60) -> bool:
        """
        Start ComfyUI server in the background

        Args:
            wait_for_ready: Whether to wait for server to be ready before returning
            timeout: Maximum time to wait for server to be ready (seconds)

        Returns:
            bool: True if server started successfully
        """
        if self.is_server_running():
            print("Server is already running")
            return True

        try:
            print("Starting ComfyUI server...")
            self.server_process = subprocess.Popen(
                ["comfy", "launch", "--background"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            print(f"Server process started with PID: {self.server_process.pid}")

            if wait_for_ready:
                return self.wait_for_server(timeout)

            return True

        except FileNotFoundError:
            print("Error: 'comfy' command not found. Make sure ComfyUI CLI is installed and in PATH")
            return False
        except Exception as e:
            print(f"Error starting server: {e}")
            return False

    def stop_server(self) -> bool:
        """
        Stop the ComfyUI server

        Returns:
            bool: True if server stopped successfully
        """
        if self.server_process is None:
            print("No server process to stop")
            return True

        try:
            print("Stopping ComfyUI server...")

            # Try graceful shutdown first
            self.server_process.terminate()

            # Wait up to 10 seconds for graceful shutdown
            try:
                self.server_process.wait(timeout=10)
                os.system(f"comfy stop")
                print("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                print("Server didn't stop gracefully, forcing shutdown...")
                self.server_process.kill()
                self.server_process.wait()
                print("Server forcefully stopped")

            self.server_process = None
            return True

        except Exception as e:
            print(f"Error stopping server: {e}")
            return False

    def is_server_running(self) -> bool:
        """
        Check if ComfyUI server is running and responsive

        Returns:
            bool: True if server is running and responsive
        """
        try:
            response = requests.get(f"{self.base_url}/system_stats", timeout=5)
            return response.status_code == 200
        except:
            return False

    def wait_for_server(self, timeout: int = 60) -> bool:
        """
        Wait for server to become ready

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if server became ready within timeout
        """
        print("Waiting for server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_server_running():
                print("Server is ready!")
                return True

            time.sleep(2)

        print(f"Server did not become ready within {timeout} seconds")
        return False

    def restart_server(self, timeout: int = 60) -> bool:
        """
        Restart the ComfyUI server

        Args:
            timeout: Maximum time to wait for server to be ready

        Returns:
            bool: True if server restarted successfully
        """
        print("Restarting ComfyUI server...")
        self.stop_server()
        time.sleep(2)  # Brief pause between stop and start
        return self.start_server(wait_for_ready=True, timeout=timeout)

    def upload_image(self, image_path: str, image_name: Optional[str] = None) -> str:
        """
        Upload image to ComfyUI server

        Args:
            image_path: Path to the image file
            image_name: Optional custom name for the image

        Returns:
            str: The filename of the uploaded image
        """
        if image_name is None:
            image_name = f"{uuid.uuid4()}.png"

        with open(image_path, 'rb') as f:
            files = {
                'image': (image_name, f, 'image/png'),
                'type': (None, 'input'),
                'overwrite': (None, 'true')
            }

            response = requests.post(f"{self.base_url}/upload/image", files=files)
            response.raise_for_status()

        return image_name

    def upload_image_from_bytes(self, image_bytes: bytes, image_name: Optional[str] = None) -> str:
        """
        Upload image from bytes to ComfyUI server

        Args:
            image_bytes: Image data as bytes
            image_name: Optional custom name for the image

        Returns:
            str: The filename of the uploaded image
        """
        if image_name is None:
            image_name = f"{uuid.uuid4()}.png"

        files = {
            'image': (image_name, io.BytesIO(image_bytes), 'image/png'),
            'type': (None, 'input'),
            'overwrite': (None, 'true')
        }

        response = requests.post(f"{self.base_url}/upload/image", files=files)
        response.raise_for_status()

        return image_name

    def queue_prompt(self, workflow: dict) -> str:
        """
        Queue a workflow for execution

        Args:
            workflow: The workflow dictionary

        Returns:
            str: The prompt ID
        """
        payload = {"prompt": workflow}
        response = requests.post(f"{self.base_url}/prompt", json=payload)
        response.raise_for_status()

        result = response.json()
        return result["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        """
        Get execution history for a prompt

        Args:
            prompt_id: The prompt ID

        Returns:
            dict: History data
        """
        response = requests.get(f"{self.base_url}/history/{prompt_id}")
        response.raise_for_status()
        return response.json()

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """
        Download image from ComfyUI server

        Args:
            filename: Name of the image file
            subfolder: Subfolder path
            folder_type: Type of folder ("output", "input", etc.)

        Returns:
            bytes: Image data
        """
        url = f"{self.base_url}/view"
        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": folder_type
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.content

    def wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        """
        Wait for workflow completion

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum time to wait in seconds

        Returns:
            dict: Final history data
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            history = self.get_history(prompt_id)

            if prompt_id in history:
                return history[prompt_id]

            time.sleep(1)

        raise TimeoutError(f"Workflow did not complete within {timeout} seconds")


def create_workflow(prompt: str, image_filename: str) -> dict:
    """
    Create workflow with custom prompt and image

    Args:
        prompt: Text prompt for generation
        image_filename: Filename of the uploaded image

    Returns:
        dict: Modified workflow
    """
    workflow = {
        "6": {
            "inputs": {
                "text": prompt,  # Custom prompt here
                "clip": ["38", 0]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "CLIP Text Encode (Positive Prompt)"}
        },
        "8": {
            "inputs": {
                "samples": ["31", 0],
                "vae": ["39", 0]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "31": {
            "inputs": {
                "seed": np.random.randint(0, 2**63),
                "steps": 20,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "beta",
                "denoise": 1,
                "model": ["247", 0],
                "positive": ["35", 0],
                "negative": ["135", 0],
                "latent_image": ["124", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "35": {
            "inputs": {
                "guidance": 2.5,
                "conditioning": ["177", 0]
            },
            "class_type": "FluxGuidance",
            "_meta": {"title": "FluxGuidance"}
        },
        "38": {
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp16.safetensors",
                "type": "flux",
                "device": "default"
            },
            "class_type": "DualCLIPLoader",
            "_meta": {"title": "DualCLIPLoader"}
        },
        "39": {
            "inputs": {
                "vae_name": "ae.safetensors"
            },
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"}
        },
        "42": {
            "inputs": {
                "image": ["146", 0]
            },
            "class_type": "FluxKontextImageScale",
            "_meta": {"title": "FluxKontextImageScale"}
        },
        "124": {
            "inputs": {
                "pixels": ["42", 0],
                "vae": ["39", 0]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode"}
        },
        "135": {
            "inputs": {
                "conditioning": ["6", 0]
            },
            "class_type": "ConditioningZeroOut",
            "_meta": {"title": "ConditioningZeroOut"}
        },
        "136": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Image"}
        },
        "146": {
            "inputs": {
                "direction": "right",
                "match_image_size": True,
                "spacing_width": 0,
                "spacing_color": "white",
                "image1": ["190", 0]
            },
            "class_type": "ImageStitch",
            "_meta": {"title": "Image Stitch"}
        },
        "173": {
            "inputs": {
                "images": ["42", 0]
            },
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Image"}
        },
        "177": {
            "inputs": {
                "conditioning": ["6", 0],
                "latent": ["124", 0]
            },
            "class_type": "ReferenceLatent",
            "_meta": {"title": "ReferenceLatent"}
        },
        "190": {
            "inputs": {
                "image": image_filename  # Custom image filename here
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Image"}
        },
        "246": {
            "inputs": {
                "unet_name": "flux1-kontext-dev-Q8_0.gguf"
            },
            "class_type": "UnetLoaderGGUF",
            "_meta": {"title": "Unet Loader (GGUF)"}
        },
        "247": {
            "inputs": {
                "model_type": "flux",
                "rel_l1_thresh": 0.4,
                "start_percent": 0,
                "end_percent": 1,
                "cache_device": "cuda",
                "model": ["246", 0]
            },
            "class_type": "TeaCache",
            "_meta": {"title": "TeaCache"}
        }
    }

    return workflow


def generate_image(prompt: str, image_path: str, server_address: str = "127.0.0.1:8188",
                   output_path: str = "output.png") -> str:
    """
    Main function to generate image using ComfyUI API

    Args:
        prompt: Text prompt for generation
        image_path: Path to input image
        server_address: ComfyUI server address
        output_path: Path to save output image

    Returns:
        str: Path to the generated image
    """
    # Initialize API client
    api = ComfyUIAPI(server_address)

    print("Uploading image...")
    # Upload the input image
    image_filename = api.upload_image(image_path)
    print(f"Image uploaded as: {image_filename}")

    print("Creating workflow...")
    # Create workflow with custom prompt and image
    workflow = create_workflow(prompt, image_filename)

    print("Queuing workflow...")
    # Queue the workflow
    prompt_id = api.queue_prompt(workflow)
    print(f"Workflow queued with ID: {prompt_id}")

    print("Waiting for completion...")
    # Wait for completion
    history = api.wait_for_completion(prompt_id)

    print("Downloading result...")
    # Get the output images
    outputs = history.get("outputs", {})
    if "136" in outputs:  # Node 136 is the SaveImage node
        images = outputs["136"]["images"]
        for image_info in images:
            filename = image_info["filename"]
            subfolder = image_info.get("subfolder", "")

            # Download the image
            image_data = api.get_image(filename, subfolder)

            # Save the image
            with open(output_path, 'wb') as f:
                f.write(image_data)

            print(f"Image saved to: {output_path}")
            return output_path

    raise Exception("No output image found")


# Example usage
if __name__ == "__main__":
    # Initialize API client
    api = ComfyUIAPI("127.0.0.1:8188")

    try:
        # Start server if not running
        if not api.is_server_running():
            print("Starting ComfyUI server...")
            if not api.start_server():
                print("Failed to start server")
                exit(1)

        # Example parameters
        prompt = "make the purple gate in a beach at sunrise"
        input_image_path = "/home/alejodosr/Downloads/a2rlgate_object.jpg"  # Path to your input image
        output_image_path = "generated_output.png"

        result_path = generate_image(
            prompt=prompt,
            image_path=input_image_path,
            server_address="127.0.0.1:8188",
            output_path=output_image_path
        )
        print(f"Success! Generated image saved to: {result_path}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Optionally stop the server when done
        api.stop_server()
