# To use intel gpu install following
# # Install Intel GPU drivers and oneAPI
# # Then clone PyTorch
# git clone https://github.com/pytorch/pytorch
# cd pytorch
# git submodule update --init --recursive
#
# # Install dependencies
# pip install -r requirements.txt
#
# # Build with XPU support
# python setup.py install -DUSE_XPU=ON
# # Or with CMake:
# mkdir build && cd build
# cmake .. -DUSE_XPU=ON
# make install
import torch

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

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

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
print("Using device:", device)

# Ensure we pass a string device to safetensors loader (works with 'cpu', 'cuda', 'mps', etc.)
device_for_loader = str(device)

unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float32)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device_for_loader))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float32).to(device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("the quick brown fox jumps over the lazy dog", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")
