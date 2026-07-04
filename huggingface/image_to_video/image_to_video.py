# SkyReels-V2 — Text-to-Video and Image-to-Video single-file runner
#
# Run:  python image_to_video.py   (after editing the CONFIG section below)
# First run: clones SkyReels-V2 repo next to this script and installs dependencies.
# Prerequisites: NVIDIA GPU, PyTorch with CUDA, git. Output: output.mp4 in this script's folder.

# ---------- PARAMETERS QUICK REFERENCE (how they affect the result) ----------
# CONFIG (top): PROMPT (what and who), DURATION_SECONDS (length → frame count), SEED (reproducibility),
#   IMAGE_PATH (start image → I2V), END_IMAGE_PATH (end image → DF only), I2V_INFERENCE_STEPS (25 vs 50).
# SECTION 2: fps (playback only), base_num_frames (DF chunk size), overlap_history (chunk overlap),
#   height/width (resolution), guidance_scale (prompt strength), shift (noise schedule), inference_steps,
#   negative_prompt (what to avoid). See inline comments for each.
# ----------------------------------------------------------------------------

# ============ CONFIG (edit these, then save and run) ============
# PROMPT: Text description of the video. More specific = more controlled result. With IMAGE_PATH, include person's appearance.
PROMPT = "A man with dark hair, short beard, wearing a blue suit waves and smiles at the camera."
# DURATION_SECONDS: Target length. 5→97 frames, 12→257, 18→377, 35→737. Longer = more VRAM and time.
DURATION_SECONDS = 5
# SEED: None = different video each run; int (e.g. 42) = same video every time (reproducible).
SEED = None

# --- Image-to-video (optional) ---
# IMAGE_PATH: Path to start image. Video begins from this image + prompt. None = text-to-video only.
# END_IMAGE_PATH: Path to end image. Video is guided to finish matching this. None = no end guidance. When set, DF model is used.
IMAGE_PATH = "start.jpg"
END_IMAGE_PATH = None
# I2V_INFERENCE_STEPS: Denoising steps when using I2V model. 25 = faster, 50 = slower, better quality.
I2V_INFERENCE_STEPS = 25
# ================================================================

import os        # Path operations: dirname, join, isdir, isfile, chdir
import random    # random.randint for generating a random seed when SEED is None
import subprocess # Run external commands: git clone, pip install
import sys       # sys.path (import search path), sys.executable (Python binary), sys.exit(1)
import argparse  # Lightweight CLI parsing for --dry-run

# SCRIPT_DIR: Absolute path to the folder containing this script. Used to find images and save output.mp4.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# REPO_DIR: Path to the SkyReels-V2 repo. We expect it next to this script (e.g. .../Ollama/SkyReels-V2).
REPO_DIR = os.path.join(SCRIPT_DIR, "SkyReels-V2")
# URL to clone if the repo is missing (--depth 1 = shallow clone, faster).
SKYREELS_REPO_URL = "https://github.com/SkyworkAI/SkyReels-V2.git"


def ensure_skyreels_repo(skip_install=False):
    """
    Ensure the SkyReels-V2 repo exists and is on the import path.
    - If skyreels_v2_infer is missing: clone the repo, install requirements.txt and extra deps (decord, imageio, moviepy).
    - Always: prepend REPO_DIR to sys.path so 'from skyreels_v2_infer import ...' works, and chdir to REPO_DIR
      so relative paths inside the pipeline (e.g. to VAE weights) resolve correctly.
    """
    if not os.path.isdir(os.path.join(REPO_DIR, "skyreels_v2_infer")):
        if not os.path.isdir(REPO_DIR):
            print("First run: cloning SkyReels-V2 (this may take a minute)...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", SKYREELS_REPO_URL, REPO_DIR],
                    check=True,
                    cwd=SCRIPT_DIR,
                )
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print("ERROR: Could not clone repo. Install git and run again, or clone manually:")
                print("  git clone https://github.com/SkyworkAI/SkyReels-V2.git")
                print("  Place the SkyReels-V2 folder in the same folder as this script.")
                raise SystemExit(1) from e
            print("Cloned.")
            if skip_install:
                print("Skipping dependency installation because --dry-run was provided.")
            else:
                print("Installing dependencies (pip)...")
                req = os.path.join(REPO_DIR, "requirements.txt")
                if os.path.isfile(req):
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req], check=True)
                for pkg in ["decord", "imageio", "moviepy"]:
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
                print("Setup done.")

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)
    os.chdir(REPO_DIR)


# Parse lightweight CLI args so users can run a dry-run without installing heavy deps
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--dry-run", action="store_true", help="Only clone repo and show device info; skip installing requirements")
args, _unknown = parser.parse_known_args()

# Run setup (clone repo). If --dry-run is provided, skip installing pip requirements.
ensure_skyreels_repo(skip_install=args.dry_run)

# If dry-run was requested, print Torch/device info and exit early to avoid heavy imports/downloads.
if args.dry_run:
    try:
        import torch
        print("dry-run: torch.__version__ =", getattr(torch, "__version__", "n/a"))
        print("dry-run: cuda available =", torch.cuda.is_available())
        print("dry-run: mps available =", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as e:
        print("dry-run: could not import torch:", e)
    sys.exit(0)


def check_cuda():
    """
    Return True if PyTorch can use CUDA (NVIDIA GPU). Otherwise print install help and return False.
    SkyReels needs a GPU; without it the script would be impractically slow or fail.
    """
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
    prompt = PROMPT.strip()
    if not prompt:
        print("ERROR: PROMPT is empty. Set PROMPT = \"your description\" at the top of this file, then run again.")
        sys.exit(1)

    # Seed: if SEED is None we pick a random integer so each run differs; otherwise we use SEED for reproducibility.
    seed = SEED if SEED is not None else random.randint(0, 2**32 - 1)
    print(f"Using PROMPT: {repr(prompt)}")
    print(f"Seed: {seed} ({'random' if SEED is None else 'fixed'})")
    print()

    if not check_cuda():
        sys.exit(1)

    # ============ SECTION 2: Video and model parameters ============
    # These are used by both I2V and DF pipelines. Model/pipeline choice is decided later (after loading images).

    # fps: Playback speed of the output video (frames per second). Does not change generation quality or frame count.
    # Effect: 24 = cinematic; 30/60 = smoother playback. Only affects how fast the saved MP4 plays.
    fps = 24

    # base_num_frames: Chunk size for the DF (Diffusion Forcing) pipeline when generating long videos.
    # The DF model generates in overlapping chunks; this is the number of frames per chunk.
    # Effect: 97 = recommended for 540P (balance quality/VRAM). Lower (57, 77) = less VRAM, slightly worse quality.
    # Only used by the DF pipeline; I2V generates all frames in one go.
    base_num_frames = 97

    # overlap_history: Frames from the previous chunk reused as context for the next chunk (DF pipeline only).
    # Effect: Higher (e.g. 37) = smoother transitions between chunks but slower. 17 is a good default.
    overlap_history = 17

    # height, width: Output resolution in pixels. Model is trained on 540P (544×960) and 720P (720×1280).
    # Effect: 544×960 = less VRAM, faster. 720×1280 = more detail, ~4× more VRAM and time. May be overridden to 960×544 if a portrait start image is used.
    height, width = 544, 960

    # num_frames: Total frames to generate. Video duration ≈ num_frames / fps. We map DURATION_SECONDS to model-friendly frame counts.
    # Effect: More frames = longer video but much more VRAM and generation time (e.g. 257 frames takes ~3× longer than 97).
    if DURATION_SECONDS <= 5:
        num_frames = 97
    elif DURATION_SECONDS <= 12:
        num_frames = 257
    elif DURATION_SECONDS <= 18:
        num_frames = 377
    elif DURATION_SECONDS <= 35:
        num_frames = 737
    else:
        num_frames = min(1457, max(97, int(DURATION_SECONDS * fps)))
    print(f"Duration: ~{round(num_frames / fps, 1)} s ({num_frames} frames @ {fps} fps)")

    # guidance_scale: How strongly the model follows the text prompt vs. unconditional (negative) prompt.
    # Effect: Lower (3–5) = more creative, may drift from prompt. Higher (7–10) = stricter to prompt, can look overcooked. 6.0 default; overridden to 5.0 for I2V.
    guidance_scale = 6.0

    # shift: Flow-matching scheduler parameter; controls the noise schedule over denoising steps.
    # Effect: 8.0 = typical for text-to-video; 5.0 = used for image-to-video (smoother). Technical; small changes affect diffusion timing.
    shift = 8.0

    # inference_steps: Number of denoising steps. More steps = better quality but proportionally slower.
    # Effect: 30 = default for DF. For I2V we override with I2V_INFERENCE_STEPS (25 or 50) when using start image only.
    inference_steps = 30

    # negative_prompt: Text describing what to avoid (blur, artifacts, bad hands/faces, etc.). Model was trained with Chinese prompts.
    # Effect: Reduces common failures; you can add more terms (e.g. "watermark") if you see specific issues.
    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    # --- Load start/end images for image-to-video (optional) ---
    # image: Encoded as first frame(s) of the latent sequence; model denoises the rest conditioned on it + prompt.
    # end_image: Encoded as last frame(s); model is guided to finish the video matching this (DF pipeline only).
    # Images are center-cropped (resizecrop) to height×width (544×960 or 960×544 for portrait).
    image_input = None
    end_image_input = None
    image_path_full = os.path.join(SCRIPT_DIR, IMAGE_PATH) if IMAGE_PATH else None
    end_image_path_full = os.path.join(SCRIPT_DIR, END_IMAGE_PATH) if END_IMAGE_PATH else None

    # Resolve start image path: try IMAGE_PATH as-is, then relative to SCRIPT_DIR.
    image_file = None
    if IMAGE_PATH:
        if os.path.isfile(IMAGE_PATH):
            image_file = IMAGE_PATH
        elif image_path_full and os.path.isfile(image_path_full):
            image_file = image_path_full
    if image_file:
        from diffusers.utils import load_image
        from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
        img = load_image(image_file)
        if img.height > img.width:
            height, width = 960, 544   # Use portrait resolution if input image is portrait
        image_input = resizecrop(img, height, width).convert("RGB")
        print(f"Using start image: {image_file}")
        print(" TIP FOR IDENTITY PRESERVATION:")
        print("   - Include person's appearance in PROMPT (e.g., 'A man with dark hair, beard, blue suit waves')")
        print("   - More inference steps (50) improve quality and identity preservation")
    elif IMAGE_PATH:
        print(f"Warning: IMAGE_PATH not found: {IMAGE_PATH}. Running text-to-video only.")

    # Load end image only when END_IMAGE_PATH is set. It is used only by the DF pipeline (not by I2V).
    # When end_image is set we choose the DF model and pass end_image_input to pipe(); I2V is never used with end_image.
    end_image_file = None
    if END_IMAGE_PATH:
        if os.path.isfile(END_IMAGE_PATH):
            end_image_file = END_IMAGE_PATH
        elif end_image_path_full and os.path.isfile(end_image_path_full):
            end_image_file = end_image_path_full
    if end_image_file:
        from diffusers.utils import load_image
        from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop
        img_end = load_image(end_image_file)
        if img_end.height > img_end.width and image_input is None:
            height, width = 960, 544
        end_image_input = resizecrop(img_end, height, width).convert("RGB")
        print(f"Using end image: {end_image_file}")
    elif END_IMAGE_PATH:
        print(f"Warning: END_IMAGE_PATH not found: {END_IMAGE_PATH}.")

    # Override settings when using a start image: I2V-friendly guidance_scale and shift; inference_steps from I2V_INFERENCE_STEPS when on I2V path.
    if image_input is not None:
        guidance_scale = 5.0
        shift = 5.0
        if end_image_input is None:
            inference_steps = I2V_INFERENCE_STEPS
        print(f"Using I2V settings: guidance_scale=5.0, shift=5.0, inference_steps={inference_steps}")

    # Choose model and pipeline:
    # - Start image only (no end image) → I2V model + Image2VideoPipeline (better identity preservation).
    # - Text-only or start+end image → DF model + DiffusionForcingPipeline (I2V pipeline does not support end_image).
    use_i2v_pipeline = (image_input is not None and end_image_input is None)
    if use_i2v_pipeline:
        model_id = "Skywork/SkyReels-V2-I2V-1.3B-540P"
        print("Using I2V model (Image2VideoPipeline) for better identity preservation")
    else:
        model_id = "Skywork/SkyReels-V2-DF-1.3B-540P"
        if image_input is not None and end_image_input is not None:
            print("Using DF model (end_image only supported with DF pipeline)")

    # ============ SECTION 3: Load model and pipeline, then generate video ============
    import torch
    import imageio
    from skyreels_v2_infer.modules import download_model

    # download_model: Downloads from Hugging Face if not cached; returns local path to model folder (config, weights, VAE).
    print("Downloading / loading model...")
    model_path = download_model(model_id)

    if use_i2v_pipeline:
        # --- I2V path: Image2VideoPipeline uses CLIP + VAE conditioning for the start image (clip_fea, y).
        from skyreels_v2_infer.pipelines.image2video_pipeline import Image2VideoPipeline
        print("Loading I2V pipeline...")
        pipe = Image2VideoPipeline(
            model_path,                    # Folder containing config, VAE, text encoder, image encoder (CLIP), transformer
            dit_path=model_path,          # Same path for the diffusion transformer (DIT) weights
            device=torch.device("cuda"),  # Run on GPU
            weight_dtype=torch.bfloat16,  # bfloat16 halves memory and speeds up with minimal quality loss
            use_usp=False,                # USP = multi-GPU sequence parallel; False = single GPU
            offload=False,                 # If True, offload some weights to CPU to save VRAM (slower)
        )
        print("Generating video (I2V)...")
        with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                image=image_input,         # PIL/image: start frame; pipeline encodes it and conditions generation on it
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                shift=shift,
                generator=torch.Generator(device="cuda").manual_seed(seed),
            )[0]
    else:
        # --- DF path: DiffusionForcingPipeline supports text-only and optional image + end_image (prefix latents).
        from skyreels_v2_infer import DiffusionForcingPipeline
        print("Loading DF pipeline...")
        pipe = DiffusionForcingPipeline(
            model_path,
            dit_path=model_path,
            device=torch.device("cuda"),
            weight_dtype=torch.bfloat16,
            use_usp=False,
            offload=False,
        )
        print("Generating video...")
        with torch.amp.autocast("cuda", dtype=pipe.transformer.dtype), torch.no_grad():
            video_frames = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=image_input,         # None or PIL: encoded as first-frame latent prefix
                end_image=end_image_input, # None or PIL: encoded as last-frame latent (DF only)
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=inference_steps,
                shift=shift,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(seed),
                overlap_history=overlap_history,   # Frames overlapping between chunks for long videos; more = smoother seams, slower
                addnoise_condition=20,             # 0=no noise on conditioning. 20=light noise for smoother chunk transitions; higher = more blur between chunks
                base_num_frames=base_num_frames,   # Chunk size for diffusion forcing (frames per chunk)
                ar_step=0,                         # 0 = all frames in sync. >0 = autoregressive (block-by-block), for very long videos only
                causal_block_size=1,               # Block size when ar_step > 0; ignored when ar_step=0
                fps=fps,
            )[0]

    # ============ SECTION 4: Save video file ============
    output_path = os.path.join(SCRIPT_DIR, "output.mp4")
    # video_frames: from I2V = (T,H,W,C) uint8 array; from DF = list of frames. imageio accepts both.
    # fps: Playback frame rate in the saved file. quality=8: FFmpeg CRF-like (0–10, higher = better). output_params: suppress FFmpeg logs.
    imageio.mimwrite(output_path, video_frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
