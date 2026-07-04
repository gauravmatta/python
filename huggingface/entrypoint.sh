#!/usr/bin/env bash
set -euo pipefail

# Simple entrypoint: if no args given, drop to bash, otherwise execute the given command
if [ "$#" -eq 0 ]; then
  echo "No command supplied — opening shell. To run the SkyReels demo:"
  echo "  docker run --gpus all -it --rm hf-skyreels:gpu python text_video/text_to_video.py"
  exec /bin/bash
else
  exec "$@"
fi

