# Title: Text-to-Image with Stable Diffusion XL (Lightning)
#
# Description:
# This script demonstrates how to generate high-quality images from text prompts
# using "Stable Diffusion XL" (SDXL).
#
# specifically, we are using "SDXL-Lightning", a specialized version that is
# EXTREMELY fast. Standard SDXL takes 20-50 steps. Lightning takes only 4 steps!
#
# Key Concepts:
# 1. Pipeline: The 'manager' that coordinates the text encoder, U-Net (brain), and decoder.
# 2. Scheduler: The algorithm that iteratively refines the noise into an image.
# 3. Checkpoints: Specific model weights (we swap the standard U-Net for the Lightning one).
#
# Installation:
# pip install torch diffusers transformers accelerate safetensors huggingface_hub
#
# How to run:
# python text_to_image.py

import torch  # PyTorch: The core deep learning library used for tensor operations.
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler # Diffusers: The library for diffusion models.
from huggingface_hub import hf_hub_download # Helper to download files from Hugging Face Hub.
from safetensors.torch import load_file # Helper to load .safetensors model files (safer/faster than .bin).

# --- Configuration ---
# The base model ID on Hugging Face. This provides the VAE, Text Encoders, and Tokenizer.
base = "stabilityai/stable-diffusion-xl-base-1.0"

# The repository containing the specialized "Lightning" weights.
repo = "ByteDance/SDXL-Lightning"

# The specific checkpoint filename.
# "4step" means this model is trained to finish in exactly 4 inference steps.
ckpt = "sdxl_lightning_4step_unet.safetensors"

# --- Step 1: Load the Model Components ---

# Load the U-Net architecture configuration from the base SDXL model.
# The U-Net is the "brain" that actually denoises the image.
# .from_config() creates the empty skeleton of the model without downloading the heavy weights yet.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)

# Download the specific "Lightning" weights and load them into our U-Net.
# hf_hub_download(repo, ckpt): Downloads the file to your local cache.
# load_file(...): Reads the .safetensors file into a state dictionary.
# unet.load_state_dict(...): Fills the empty U-Net skeleton with the Lightning weights.
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))

# Create the full pipeline using the base SDXL components but our custom Lightning U-Net.
# The pipeline assembles the Text Encoder, VAE, Scheduler, and our custom U-Net.
pipe = StableDiffusionXLPipeline.from_pretrained(
    base,
    unet=unet, # Inject our lightning-fast U-Net
    torch_dtype=torch.float16, # Use half-precision (fp16) to save VRAM and speed up generation.
    variant="fp16" # Download the fp16 version of the base model weights if available.
).to("cuda") # Move the entire pipeline to the GPU.

# --- Step 2: Configure the Scheduler ---
# The Scheduler controls the noise removal process.
# SDXL-Lightning requires the "EulerDiscreteScheduler".
# timestep_spacing="trailing": A specific mathematical setting required for the Lightning distillation method.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# --- Step 3: Generate Image ---
# Improved Prompt: More descriptive keywords can help the model fill in details.
prompt = "A smiling girl."

print("Generating image...")
# pipe(...) calls the pipeline to generate the image.
# prompt: The text description.
# num_inference_steps=4: Must match the 4-step checkpoint we loaded for optimal performance.
# guidance_scale=0: Lightning models are specifically trained to work best with CFG set to 0.
# height=720, width=1280: Sets the output resolution to 16:9 HD (Landscape).
image = pipe(
    prompt,
    num_inference_steps=4,
    guidance_scale=0,
    height=720,
    width=1280
).images[0] # The pipeline returns a list of images. We take the first one.

# Save the result to the disk.
image.save("output.png")
print("Image saved to output.png")