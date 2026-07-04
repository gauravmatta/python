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
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
device = "cuda" if torch.cuda.is_available() else "cpu"
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch.float32)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float32).to(device)

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("the quick brown fox jumps over the lazy dog", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")
