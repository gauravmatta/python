# SkyReels-V2-DF-1.3B-540P text-to-video — single file runner
#
# You only need this file. Edit PROMPT (and SEED, DURATION_SECONDS) below, SAVE, then run:
#   python text_to_video.py
#
# First run: script will clone SkyReels-V2 next to this file if missing and install deps.
# Prerequisites: NVIDIA GPU, PyTorch with CUDA, git, and internet for first-run setup.
# Output: output.mp4 in the same folder as this script (overwritten each run)

# ============ CONFIG (edit these, then save and run) ============
PROMPT = "Two men fighting in a park."  # Text description for the video to generate
DURATION_SECONDS = 5  # Target length: 5→97 frames (~4s), 10→257, 15→377, 30→737 (longer = more VRAM/time)
SEED = None  # None = different video each run; set to an int (e.g. 42) for the same result every time
# ============================================`===================

import os  # For file paths, directory checks, and joining paths
import random  # For generating a random seed when SEED is None
import subprocess  # For running git clone and pip install in a subprocess
import sys  # For sys.path (import search path), sys.executable (Python path), sys.exit

# Path where this script file lives (e.g. D:\Udemy\AI COURSES\Ollama)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Folder for the SkyReels-V2 repo; we expect it next to this script (e.g. ...\Ollama\SkyReels-V2)
REPO_DIR = os.path.join(SCRIPT_DIR, "SkyReels-V2")
# Git URL to clone if the repo is missing
SKYREELS_REPO_URL = "https://github.com/SkyworkAI/SkyReels-V2.git"


def ensure_skyreels_repo():
    """Clone SkyReels-V2 next to this script if missing; then add repo to import path and chdir into it."""
    # Check if the inference package exists inside the repo (SkyReels-V2/skyreels_v2_infer)
    if not os.path.isdir(os.path.join(REPO_DIR, "skyreels_v2_infer")):
        # Repo folder itself might be missing (first run) or incomplete
        if not os.path.isdir(REPO_DIR):
            print("First run: cloning SkyReels-V2 (this may take a minute)...")
            try:
                # Run: git clone --depth 1 <url> <REPO_DIR> from SCRIPT_DIR so repo is created next to this file
                subprocess.run(
                    ["git", "clone", "--depth", "1", SKYREELS_REPO_URL, REPO_DIR],
                    check=True,  # Raise CalledProcessError if git exits with non-zero
                    cwd=SCRIPT_DIR,  # Run in script's folder so REPO_DIR is relative to it
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                # Git failed or not installed; tell user how to fix
                print("ERROR: Could not clone repo. Install git and run again, or clone manually:")
                print("  git clone https://github.com/SkyworkAI/SkyReels-V2.git")
                print("  Place the SkyReels-V2 folder in the same folder as this script.")
                raise SystemExit(1) from e
            print("Cloned. Installing dependencies (pip)...")
            req = os.path.join(REPO_DIR, "requirements.txt")
            if os.path.isfile(req):
                # Install torch, diffusers, etc. from SkyReels-V2/requirements.txt (-q = quiet)
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req], check=True)
            for pkg in ["decord", "imageio", "moviepy"]:
                # Extra deps the pipeline uses; check=False so one failure doesn't stop the rest
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
            print("Setup done.")

    # Always add REPO_DIR to Python's import path so "from skyreels_v2_infer import ..." works
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    # Change current working directory to repo root so relative paths inside the pipeline work correctly
    os.chdir(REPO_DIR)


# Run setup once at import time: clone repo if missing, then set sys.path and cwd for imports
ensure_skyreels_repo()


def check_cuda():
    """Return True if PyTorch can use CUDA (NVIDIA GPU); otherwise print help and return False."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    print("ERROR: PyTorch is not using CUDA. SkyReels-V2 needs an NVIDIA GPU.")
    print("Install: pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121")
    return False


def main():
    # ============ SECTION 1: Read config and validate ============
    prompt = PROMPT.strip()  # Get prompt from top of file; strip() removes leading/trailing whitespace
    if not prompt:
        print("ERROR: PROMPT is empty. Set PROMPT = \"your description\" at the top of this file, then run again.")
        sys.exit(1)

    # SEED controls randomness: None = different video each run; int = same video every time (reproducible)
    # Effect: Same seed + same prompt = identical video. Different seed = different composition/colors/details.
    # Use fixed seed (e.g. 42) when you want to compare prompts or debug; use None for variety.
    seed = SEED if SEED is not None else random.randint(0, 2 ** 32 - 1)
    print(f"Using PROMPT: {repr(prompt)}")
    print(f"Seed: {seed} ({'random' if SEED is None else 'fixed'})")
    print()

    if not check_cuda():
        sys.exit(1)

    # ============ SECTION 2: Video and model parameters ============
    # MODEL_ID: Hugging Face model identifier. The 1.3B model is smaller/faster; 14B is higher quality but needs more VRAM.
    # download_model() will cache it locally after first download (~5GB for 1.3B, ~28GB for 14B).
    model_id = "Skywork/SkyReels-V2-DF-1.3B-540P"

    # FPS (Frames Per Second): Output video playback speed. 24 fps is standard for film/cinematic look.
    # Effect: Doesn't change generation quality, only playback speed. Higher fps (30, 60) makes motion smoother
    # but the model still generates the same number of frames - you're just playing them faster.
    fps = 24

    # BASE_NUM_FRAMES: The "chunk size" for diffusion forcing. The model generates long videos in overlapping chunks.
    # Effect: Lower (e.g. 77, 57) = less VRAM but slightly lower quality. Higher (97, 121) = better quality, more VRAM.
    # For 540P: 97 is recommended. For 720P: 121. Don't go below ~57 or quality degrades noticeably.
    base_num_frames = 97

    # OVERLAP_HISTORY: How many frames from the previous chunk are reused as context for the next chunk.
    # Effect: Higher overlap (e.g. 17, 37) = smoother transitions between chunks but slower generation.
    # Lower overlap = faster but may have visible "cuts" between chunks. 17 is good for most cases.
    overlap_history = 17

    # HEIGHT, WIDTH: Output video resolution. 540P = 544x960, 720P = 720x1280.
    # Effect: Higher resolution = more detail but MUCH more VRAM and slower generation (roughly 4x slower for 2x resolution).
    # The model was trained on these exact sizes, so changing them may reduce quality.
    height, width = 544, 960  # 540P resolution

    # NUM_FRAMES: Total frames to generate. Determines video length: duration = num_frames / fps.
    # The model was trained on specific frame counts (97, 257, 377, 737), so using these gives best quality.
    # Effect: More frames = longer video but exponentially more VRAM/time (97→257 is ~3x longer generation).
    # We map DURATION_SECONDS to model-recommended values for optimal quality.
    if DURATION_SECONDS <= 5:
        num_frames = 97  # ~4 seconds: fastest, lowest VRAM (~15GB with offload)
    elif DURATION_SECONDS <= 12:
        num_frames = 257  # ~10 seconds: good balance, moderate VRAM (~18GB)
    elif DURATION_SECONDS <= 18:
        num_frames = 377  # ~15 seconds: longer, more VRAM (~20GB)
    elif DURATION_SECONDS <= 35:
        num_frames = 737  # ~30 seconds: very long, high VRAM (~25GB+)
    else:
        num_frames = min(1457, max(97, int(DURATION_SECONDS * fps)))  # Up to ~60 seconds max
    print(f"Duration: ~{round(num_frames / fps, 1)} s ({num_frames} frames @ {fps} fps)")

    # GUIDANCE_SCALE: How strongly the model follows your text prompt vs. its own creativity.
    # Effect: Lower (3-5) = more creative/artistic but may ignore prompt details. Higher (7-10) = strict prompt
    # adherence but can look "overcooked" or unnatural. 6.0 is a good balance for most prompts.
    # Trade-off: Higher guidance = slightly slower (more computation per step).
    guidance_scale = 6.0

    # SHIFT: Flow-matching scheduler parameter that controls the noise schedule (how noise is removed over steps).
    # Effect: Higher shift (8-10) = faster denoising, lower shift (5-7) = slower but potentially smoother.
    # For text-to-video: 8.0 is recommended. For image-to-video: 5.0 works better.
    # This is a technical parameter - changing it subtly affects the diffusion process timing.
    shift = 8.0

    # INFERENCE_STEPS: Number of denoising iterations. Each step refines the video from noise toward the prompt.
    # Effect: More steps (50-100) = higher quality, sharper details, better prompt adherence, but MUCH slower.
    # Fewer steps (20-30) = faster generation but may be blurrier or miss prompt details.
    # 30 steps is a good balance for speed/quality. For best quality, try 50-60 (but 2x slower).
    # Performance: Each step processes all frames, so 50 steps takes ~1.7x longer than 30 steps.
    inference_steps = 30

    # NEGATIVE_PROMPT: text describing what to avoid (bad quality, artifacts, static scenes, etc.).
    # Effect: Helps steer the model away from common failures (blurry, static, distorted faces/hands).
    # The model was trained with prompts, so this negative prompt is more effective than English.
    # You can modify this to avoid specific issues you're seeing (e.g., add "watermark" if videos have watermarks).
    negative_prompt = (
        "Garish colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, image, motionless, overall grayish cast,"
        "worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands,"
        "poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, static image,"
        "cluttered background, three legs, crowded background, walking backwards"
            )

    # ============ SECTION 3: Load model and pipeline ============
    import torch
    import imageio  # For writing video frames to MP4 file
    from skyreels_v2_infer import DiffusionForcingPipeline
    from skyreels_v2_infer.modules import download_model

    print("Downloading / loading model...")
    # download_model() checks Hugging Face cache first; if missing, downloads (~5GB for 1.3B model).
    # Returns local path to the model folder (usually in ~/.cache/huggingface/hub/).
    model_path = download_model(model_id)

    print("Loading pipeline...")
    # DiffusionForcingPipeline: The main class that orchestrates video generation.
    # It loads: transformer (diffusion model), VAE (encodes/decodes frames), text encoder (processes prompt).
    pipe = DiffusionForcingPipeline(
        model_path,  # Path to model folder (contains config.json, *.safetensors, VAE weights)
        dit_path=model_path,  # Same path (DIT = Diffusion Transformer, the core model)
        device=torch.device("cuda"),  # Run on GPU (required for video generation)
        weight_dtype=torch.bfloat16,  # Use bfloat16 precision: 2x faster than float32, minimal quality loss
        use_usp=False,  # USP = multi-GPU acceleration. False = single GPU (simpler, works for most)
        offload=False,  # Offload some model weights to CPU RAM when not in use.
        # Effect: Reduces peak VRAM by ~30% but slightly slower (CPU↔GPU transfers).
        # Set False if you have plenty of VRAM (>20GB) for faster generation.
    )

    print("Generating video...")
    # torch.amp.autocast("cuda"): Automatically uses bfloat16/float16 on GPU ops for speed (2x faster than float32).
    # torch.no_grad(): Disables gradient tracking (we're doing inference, not training) - saves memory and speed.
    # Together these optimize performance: faster generation, lower VRAM, minimal quality impact.
    with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
        # Call the pipeline to generate video frames. This is the main computation - can take minutes.
        video_frames = pipe(
            prompt=prompt,  # Your text description (what to generate)
            negative_prompt=negative_prompt,  # What to avoid (improves quality by steering away from failures)
            image=None,  # For image-to-video: provide PIL Image. None = text-to-video only.
            end_image=None,  # Optional: guide the last frame toward this image (rarely used)
            height=height,  # Output height in pixels (must match model training resolution)
            width=width,  # Output width in pixels
            num_frames=num_frames,  # Total frames to generate (determines video length)
            num_inference_steps=inference_steps,  # Denoising steps (more = better quality, slower)
            shift=shift,  # Flow-matching scheduler parameter (controls noise schedule)
            guidance_scale=guidance_scale,  # Prompt adherence strength (higher = stricter to prompt)
            generator=torch.Generator(device="cuda").manual_seed(seed),  # Random number generator with fixed seed
            overlap_history=overlap_history,  # Frames to overlap between chunks (for long videos)
            addnoise_condition=20,  # Adds slight noise to conditioning for smoother long videos.
            # Effect: Higher (30-50) = smoother but may reduce consistency.
            # Lower (10-15) = sharper but may have visible transitions.
            # 20 is a good default. Only matters for videos > base_num_frames.
            base_num_frames=base_num_frames,  # Chunk size for diffusion forcing (how many frames per chunk)
            ar_step=0,  # Autoregressive step count. 0 = synchronous (all frames at once).
            # Higher (5-10) = asynchronous (generates in blocks, slower but better
            # for very long videos). 0 is recommended for videos < 30 seconds.
            causal_block_size=1,  # Only used when ar_step > 0. Size of causal attention blocks.
            # Ignored when ar_step=0. Keep at 1 unless using async mode.
            fps=fps,  # Frames per second (for metadata and playback speed)
        )[0]  # pipe() returns a list of video frame arrays. [0] gets the first (only) element.
        # Each frame is a numpy array shape (height, width, 3) with RGB values 0-255.

    # ============ SECTION 4: Save video file ============
    output_path = os.path.join(SCRIPT_DIR, "output1.mp4")
    # imageio.mimwrite(): Writes a list of frames to an MP4 file.
    # fps=fps: Sets playback speed in the video file.
    # quality=8: FFmpeg quality (0-10, higher = better). 8 is good balance (smaller file, minimal quality loss).
    # output_params=["-loglevel", "error"]: Suppress FFmpeg info messages (only show errors).
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()  # Run main only when this file is executed directly (not when imported)
