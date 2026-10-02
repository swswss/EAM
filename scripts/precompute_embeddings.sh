#!/bin/bash
# Pre-compute SD3.5-Medium text embeddings for the PickScore prompts (train: 25,432, test: 2,048).
# Needs one GPU; writes ~44 GB to dataset/pickscore/embeddings.
set -euo pipefail
cd "$(dirname "$0")/.."

python text_embedding_store.py \
    --model_id stabilityai/stable-diffusion-3.5-medium \
    --dataset_dir dataset/pickscore \
    --output_dir dataset/pickscore/embeddings \
    --batch_size 64 \
    --dtype bf16
