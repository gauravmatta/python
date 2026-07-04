# Minimal GPU-ready Dockerfile for running HuggingFace + SkyReels-V2
# Base image: CUDA runtime with cuDNN (adjust CUDA tag if needed)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git wget ca-certificates build-essential && \
    rm -rf /var/lib/apt/lists/*

# Make python point to python3.11 and upgrade pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && python -m pip install --upgrade pip

# Allow build args to select PyTorch/CUDA versions at build-time
ARG PYTORCH_VERSION=2.5.1
ARG TORCH_CUDA_TAG=cu121
ENV TORCH_VERSION=${PYTORCH_VERSION}
ENV TORCH_CUDA_TAG=${TORCH_CUDA_TAG}

# Install PyTorch and torchvision from the official PyTorch index for the selected CUDA
# If the indexed wheel fails, fall back to a plain pip install (useful for testing)
RUN pip --no-cache-dir install "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" "torchvision==0.20.1+${TORCH_CUDA_TAG}" --index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} || \
    pip --no-cache-dir install "torch==${TORCH_VERSION}" "torchvision==0.20.1"

# Install common Python deps used by the project (adjust or pin versions if needed)
RUN pip --no-cache-dir install git+https://github.com/huggingface/diffusers.git transformers accelerate tokenizers safetensors ftfy einops imageio imageio-ffmpeg opencv-python

# Create workspace and copy repository into container
WORKDIR /workspace
COPY . /workspace

# If SkyReels requirements exist, try to install them (ignore failures so container still builds)
RUN if [ -f text_video/SkyReels-V2/requirements.txt ]; then pip --no-cache-dir install -r text_video/SkyReels-V2/requirements.txt || true; fi

# Copy an entrypoint helper
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["/bin/bash"]

