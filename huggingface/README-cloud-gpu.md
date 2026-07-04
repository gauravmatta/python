Running this project on a cloud GPU (short guide)

Checklist
- Pick a cloud provider and a GPU VM with compatible NVIDIA drivers. Common choices: AWS (p4/p3/g4), GCP (A2/instance), Azure (NC/ND), Lambda Labs, Paperspace.
- Use an Ubuntu 22.04 / Deep Learning VM image with NVIDIA drivers installed, and install NVIDIA Container Toolkit (nvidia-docker) so Docker containers can access GPUs.
- Build the GPU Docker image included here or use the host to install Python + CUDA-enabled PyTorch directly.

Quick overview (recommended path: Docker container with CUDA runtime)

1) Build the image locally or on the cloud VM

```bash
# From the repository root (/workspace)
# Adjust PYTORCH_VERSION and TORCH_CUDA_TAG if your CUDA version differs
docker build \
  --build-arg PYTORCH_VERSION=2.5.1 \
  --build-arg TORCH_CUDA_TAG=cu121 \
  -t hf-skyreels:gpu .
```

2) Run a shell inside the container with GPU access

```bash
docker run --gpus all --ipc=host -v "$(pwd)":/workspace -w /workspace -it --rm hf-skyreels:gpu
# then inside container, you can run:
# python text_video/text_to_video.py
# or python image_to_video/image_to_video.py --dry-run
```

3) Run the SkyReels pipeline directly (example)

```bash
# Run the text_to_video runner (this expects CUDA + a supported PyTorch wheel)
docker run --gpus all --ipc=host -v "$(pwd)":/workspace -w /workspace -it --rm hf-skyreels:gpu python text_video/text_to_video.py
```

Provider-specific notes

AWS (EC2)
- Launch an EC2 instance with a GPU (e.g., p4d, g5, g4dn); pick Ubuntu 22.04 or a Deep Learning AMI.
- Install NVIDIA drivers and the NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- SSH to the instance, clone this repo and run the build/run commands above.
- If you prefer not to build the image, you can install Python and the exact CUDA-enabled PyTorch wheel directly on the instance:
  - Example install (for CUDA 12.1):
    ```bash
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    ```

GCP
- Use a Deep Learning VM image or launch an instance and install NVIDIA drivers and NVIDIA Container Toolkit.
- Build and run the container as above.

Azure
- Use NC/ND series VMs with NVIDIA drivers. Install NVIDIA Container Toolkit and run the container.

Lambda Labs / Paperspace
- These providers often give ready-to-use GPU VMs with drivers and Docker already configured — just build and run the container as above.

Troubleshooting
- "Docker run fails with: could not select device driver "nvidia": could not find" — install the NVIDIA Container Toolkit on the host.
- "PyTorch can't find CUDA" — ensure the host has matching NVIDIA drivers and that you used the correct CUDA tag (e.g., cu121). You can call `nvidia-smi` on the host to check driver/CUDA compatibility.
- If a specific PyTorch version+CUDA wheel isn't available for your Python version, build the container with a different PYTORCH_VERSION or change `TORCH_CUDA_TAG` to match your drivers (e.g., cu118/cu121). See https://pytorch.org/get-started/locally/ for exact index URLs.

Security & cost
- GPU instances are expensive — stop or terminate your instance when not in use.
- If you don't need to persist large model downloads between runs, run containers with `--rm` to remove them afterwards.

Advanced: Using a prebuilt base image
- If you want faster builds, use NVIDIA NGC or PyTorch's official container images as base images (they come with CUDA and PyTorch preinstalled).
  Example: `nvcr.io/nvidia/pytorch:23.11-py3` or equivalent. Adjust your Dockerfile accordingly.

If you tell me which provider you plan to use and which GPU (or the CUDA version / driver), I will produce an exact set of commands (and a tuned Dockerfile) for that target.

