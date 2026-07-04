#!/usr/bin/env bash
set -euo pipefail

# Helper to build and run the Docker container locally (expects NVIDIA Container Toolkit installed)
IMAGE_NAME=hf-skyreels:gpu

# Build the image (customize PYTORCH_VERSION and TORCH_CUDA_TAG if needed)
docker build \
  --build-arg PYTORCH_VERSION=2.5.1 \
  --build-arg TORCH_CUDA_TAG=cu121 \
  -t ${IMAGE_NAME} .

# Run the container with GPU access, mounting current repo into /workspace
docker run --gpus all --ipc=host -v "$(pwd)":/workspace -w /workspace -it --rm ${IMAGE_NAME} "$@"

