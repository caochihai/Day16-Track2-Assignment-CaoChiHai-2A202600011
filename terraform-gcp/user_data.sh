#!/bin/bash
set -e

echo "Starting vLLM Setup on GCP with NVIDIA L4"

# 1. Install NVIDIA Container Toolkit (if not present)
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Run vLLM with Dynamic Variables
docker run -d --name vllm \
  --runtime nvidia --gpus all \
  --restart unless-stopped \
  -p 8000:8000 \
  -e HUGGING_FACE_HUB_TOKEN=${hf_token} \
  vllm/vllm-openai:latest \
  --model ${model_id} \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9
