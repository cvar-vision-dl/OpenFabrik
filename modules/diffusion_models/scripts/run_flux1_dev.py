import os

# set your local working directory for downloading 'black-forest-labs/FLUX.1-schnell' models
os.environ['HF_HOME'] = '/mnt/a/alejodosr/models/diffusion-models/'

import torch
from diffusers import FluxPipeline


def main():
    # Load the model pipeline (either from Huggingface or local path)
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.float16, cache_dir=os.environ['HF_HOME'])

    # For low VRAM GPUs (i.e. between 4 and 32 GB VRAM)
    pipe.enable_sequential_cpu_offload()  # Enable CPU offload to save GPU memory
    # pipe.vae.enable_slicing()  # Enable VAE slicing for memory efficiency
    # pipe.vae.enable_tiling()  # Enable VAE tiling for better performance on high-res images

    # Move the pipeline to GPU
    # pipe.to("cuda")  # Ensure the pipeline is using the GPU (cuda) <= no need to convert cuda
    # Sequential Offloading이 활성화된 상태에서는 파이프라인이 자동으로 필요한 모델 파라미터를 GPU로 전송하고, 나머지는 CPU에서 유지합니다.
    # 따로 to("cuda")를 호출할 필요가 없습니다.

    # casting here instead of in the pipeline constructor because doing so in the constructor loads all models into CPU memory at once
    # pipe.to(torch.float16)

    # Define the prompt
    # prompt = "A cat holding a sign that says hello world"
    prompts = ["Professional photograph of modern office desk with laptop, coffee mug, and notebook in natural lighting. High resolution, shallow depth of field, minimalist composition.",
               "Industrial warehouse interior with metal shelving, cardboard boxes, and forklift in the background. Wide angle shot with dramatic lighting from overhead industrial lights.",
               "Close-up photograph of electronic circuit board showing detailed components, solder points, and microchips. Macro lens with sharp focus on central processing unit."]

    for prompt in prompts:
        # Generate the image
        image = pipe(
            prompt,
            guidance_scale=7.5,  # Typically a higher value works better for stable diffusion
            num_inference_steps=4,
            # height=1024,
            # width=1024,
            # num_inference_steps=50,  # More steps provide better image quality
            # max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(-1),
            # generator=torch.Generator("cuda").manual_seed(0),  # Use GPU for generation
        ).images[0]

        # Show and save the generated image
        image.show()
        image.save("flux-schnell.png")

if __name__ == '__main__':
    main()