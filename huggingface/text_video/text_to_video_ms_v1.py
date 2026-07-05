# ali-vilab/text-to-video-ms-1.7b — single file runner
#
# You only need this file. Edit PROMPT (and SEED, DURATION_SECONDS) below, SAVE, then run:
#   python text_to_video.py
#
# First run: script will install missing pip packages automatically.
# Prerequisites: NVIDIA GPU, PyTorch with CUDA, internet for first-run model download (~5.4GB).
# Output: output.mp4 in the same folder as this script (overwritten each run)
#
# Tuned for 6GB VRAM cards (e.g. RTX 3050 laptop). Uses sequential CPU offload + VAE slicing,
# which trades speed for a much lower peak VRAM footprint than the default .to("cuda") usage.

# ============ CONFIG (edit these, then save and run) ============
PROMPT = "Two men fighting in a park."   # Text description for the video to generate
NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, subtitles, text, static image, "
    "extra limbs, deformed hands, worst quality"
)
DURATION_SECONDS = 4      # This model generates a fixed short clip; ~4s (64 frames) is the safe default for 6GB.
SEED = None               # None = different video each run; set an int (e.g. 42) to reproduce a result
NUM_INFERENCE_STEPS = 25  # Fewer = faster/lower quality. 25 is a good balance. Try 15 for a quick preview.
GUIDANCE_SCALE = 9.0      # How closely the video follows the prompt. 7-12 is a reasonable range.
# ==================================================================

import os
import random
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "ali-vilab/text-to-video-ms-1.7b"
FPS = 8  # This model was trained/typically sampled at 8 fps; 64 frames @ 8fps = 8s of raw frames,
         # but we scale frame count to DURATION_SECONDS below.

def use_local_venv_if_available():
    """Re-run this script with the project venv so CUDA PyTorch is picked up."""
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    venv_python = os.path.join(script_dir, ".venv", "Scripts", "python.exe")
    if os.path.isfile(venv_python) and os.path.abspath(sys.executable).lower() != os.path.abspath(venv_python).lower():
        print(f"Using local virtual environment: {venv_python}", flush=True)
        result = subprocess.run([venv_python, script_path, *sys.argv[1:]], cwd=script_dir)
        sys.exit(result.returncode)


use_local_venv_if_available()


def ensure_dependencies():
    """Install required packages if missing. Torch/CUDA is assumed to already be installed."""
    required = {
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "imageio": "imageio",
        "imageio_ffmpeg": "imageio-ffmpeg",
    }
    missing = []
    for module_name, pip_name in required.items():
        try:
            __import__(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def check_cuda():
    """Return True if PyTorch can use CUDA (NVIDIA GPU); otherwise print help and return False."""
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {name} ({vram_gb:.1f} GB VRAM)")
        return True
    print("ERROR: PyTorch is not using CUDA. This model needs an NVIDIA GPU.")
    print("Install GPU-enabled PyTorch with:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    return False


def main():
    ensure_dependencies()

    import torch
    import imageio
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
    from diffusers.utils import export_to_video

    prompt = PROMPT.strip()
    if not prompt:
        print('ERROR: PROMPT is empty. Set PROMPT = "your description" at the top of this file, then run again.')
        sys.exit(1)

    if not check_cuda():
        sys.exit(1)

    seed = SEED if SEED is not None else random.randint(0, 2 ** 32 - 1)
    print(f"Using PROMPT: {repr(prompt)}")
    print(f"Seed: {seed} ({'random' if SEED is None else 'fixed'})")

    # This model's native frame count is 16 at default resolution, but it can generate more.
    # More frames = linearly more VRAM/time. On 6GB, stay near 24-48 frames.
    num_frames = max(16, min(64, int(DURATION_SECONDS * FPS)))
    print(f"Requesting {num_frames} frames (~{num_frames / FPS:.1f}s of raw output before any frame interpolation)")

    print("Loading pipeline (first run will download ~5.4GB)...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # --- Low-VRAM setup for 6GB cards ---
    # enable_sequential_cpu_offload moves weights to CPU layer-by-layer, only pulling what's
    # needed onto the GPU for each op. Much lower peak VRAM than enable_model_cpu_offload,
    # at the cost of noticeably slower generation. Necessary at this VRAM tier for this model.
    pipe.enable_sequential_cpu_offload()
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    print("Generating video (this can take several minutes on 6GB with CPU offload)...")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    video_frames = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_frames=num_frames,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
    ).frames[0]

    output_path = os.path.join(SCRIPT_DIR, "output.mp4")
    export_to_video(video_frames, output_path, fps=FPS)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()