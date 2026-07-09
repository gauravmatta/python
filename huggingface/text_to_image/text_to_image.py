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

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # PyTorch: The core deep learning library used for tensor operations.

# Shim for torch.xpu when PyTorch is built without Intel XPU support.
# Some `diffusers` versions reference `torch.xpu.empty_cache` at import time.
# Provide a no-op implementation so imports don't fail on systems without XPU.
if not hasattr(torch, "xpu"):
	class _XPUStub:
		@staticmethod
		def empty_cache():
			return None

		@staticmethod
		def device_count():
			return 0

		@staticmethod
		def manual_seed(seed):
			return None

		@staticmethod
		def is_available():
			return False

		@staticmethod
		def reset_peak_memory_stats():
			return None

		@staticmethod
		def max_memory_allocated():
			return 0

		@staticmethod
		def synchronize():
			return None

	setattr(torch, "xpu", _XPUStub())

# Some diffusers internals reference `torch.distributed.device_mesh`. Provide a minimal
# stub when it's missing to avoid import-time AttributeError on installations without
# distributed/device-mesh features.
if not hasattr(torch, "distributed"):
	import types as _types
	torch.distributed = _types.SimpleNamespace()

if not hasattr(torch.distributed, "device_mesh"):
	import types as _types

	class _DeviceMeshStub:
		"""Minimal placeholder for torch.distributed.device_mesh.DeviceMesh used for typing."""

		def __init__(self, *args, **kwargs):
			pass

	torch.distributed.device_mesh = _types.SimpleNamespace(DeviceMesh=_DeviceMeshStub)

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler # Diffusers: The library for diffusion models.
from huggingface_hub import hf_hub_download # Helper to download files from Hugging Face Hub.
from safetensors.torch import load_file # Helper to load .safetensors model files (safer/faster than .bin).


# Load model.
def get_device():
	"""Return the best available torch device as a torch.device or a framework-specific device.

	Priority: CUDA -> MPS (Apple) -> XLA (TPU via torch_xla) -> CPU
	"""
	# CUDA (NVIDIA)
	if torch.cuda.is_available():
		return torch.device("cuda")

	# Apple Metal Performance Shaders (Apple Silicon)
	if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
		return torch.device("mps")

	# TPU via torch_xla (if installed)
	try:
		import importlib
		if importlib.util.find_spec("torch_xla") is not None:
			# Use torch_xla's xla_device() if available
			import torch_xla.core.xla_model as xm
			return xm.xla_device()
	except Exception:
		# ignore import errors and fall back to CPU
		pass

	# Fallback to CPU
	return torch.device("cpu")


device = get_device()
device_type = getattr(device, "type", str(device).split(":")[0])
print("Using device:", device)

# Choose dtype based on device: float32 for CPU (no float16 support), float16 for GPU/accelerators
dtype_for_model = torch.float32 if device_type == "cpu" else torch.float16
print(f"Using dtype: {dtype_for_model}")

# Choose variant based on dtype: use fp16 variant only when using float16
variant_for_model = "fp16" if dtype_for_model == torch.float16 else None

total_vram_gib = 0
if device_type == "cuda":
	total_vram_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3
	print(f"GPU memory: {total_vram_gib:.2f} GiB")

low_vram_cuda = device_type == "cuda" and total_vram_gib < 10
if low_vram_cuda:
	print("Low VRAM CUDA mode: loading weights on CPU and enabling model CPU offload.")


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
# .load_config() downloads only the small config file, then .from_config() creates the skeleton.
unet_config = UNet2DConditionModel.load_config(base, subfolder="unet")
unet = UNet2DConditionModel.from_config(unet_config).to(dtype=dtype_for_model)

# Download the specific "Lightning" weights and load them into our U-Net.
# hf_hub_download(repo, ckpt): Downloads the file to your local cache.
# load_file(..., device="cpu"): Reads the .safetensors file into CPU RAM first.
# unet.load_state_dict(...): Fills the empty U-Net skeleton with the Lightning weights.
checkpoint_path = hf_hub_download(repo, ckpt)
state_dict = load_file(checkpoint_path, device="cpu")
unet.load_state_dict(state_dict)
del state_dict
if device_type == "cuda":
	torch.cuda.empty_cache()

# Create the full pipeline using the base SDXL components but our custom Lightning U-Net.
# The pipeline assembles the Text Encoder, VAE, Scheduler, and our custom U-Net.
pipe = StableDiffusionXLPipeline.from_pretrained(
    base,
    unet=unet, # Inject our lightning-fast U-Net
	torch_dtype=dtype_for_model, # Use appropriate precision: float32 on CPU, float16 on GPU
	variant=variant_for_model, # Download fp16 variant only when using float16; otherwise use default weights
	use_safetensors=True,
	low_cpu_mem_usage=True,
)

if low_vram_cuda:
	pipe.enable_model_cpu_offload()
else:
	pipe = pipe.to(device) # Move the entire pipeline to the specified device.

if device_type == "cuda":
	pipe.enable_vae_slicing()
	pipe.enable_vae_tiling()

# Ensure all components use the correct dtype, especially on CPU where float16 is unsupported
if dtype_for_model == torch.float32:
	# Explicitly convert all text encoders and VAE to float32 to avoid float16 issues on CPU
	pipe.text_encoder = pipe.text_encoder.to(dtype=dtype_for_model)
	pipe.text_encoder_2 = pipe.text_encoder_2.to(dtype=dtype_for_model)
	pipe.vae = pipe.vae.to(dtype=dtype_for_model)
	pipe.unet = pipe.unet.to(dtype=dtype_for_model)

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
