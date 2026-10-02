#!/bin/bash
# DrawBench evaluation of one LoRA checkpoint (or the base model if ADAPTER is empty).
#   1) evaluate.py     : 1,000 images = 200 DrawBench prompts x 5 (512px, DPM-Solver-2, 10 steps, no CFG) + PickScore
#   2) eval_metrics.py : ImageReward, PickScore, Aesthetic, CLIPScore, HPSv2.1 on the saved images
#
# Usage:
#   bash scripts/eval.sh <adapter_dir> <output_dir> [seed]
#   bash scripts/eval.sh checkpoints/<run>/checkpoints/<iter>/lora_current_policy eval/<name>
#   bash scripts/eval.sh "" eval/sd35m_base            # pretrained SD3.5-Medium
# We report the mean over seeds 42, 77, 369 and 2000.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER=${1:-}
OUT=${2:?usage: bash scripts/eval.sh <adapter_dir|""> <output_dir> [seed]}
SEED=${3:-42}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29510}

ADAPTER_ARGS=()
if [ -n "${ADAPTER}" ]; then
    ADAPTER_ARGS=(--adapter_path "${ADAPTER}")
fi

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} evaluate.py \
    "${ADAPTER_ARGS[@]}" \
    --dataset_path dataset/drawbench \
    --output_dir "${OUT}/s${SEED}" \
    --seed "${SEED}" \
    --batch_size 16 \
    --score_batch_size 8 \
    --solver dpm2 \
    --num_inference_steps 10 \
    --guidance_scale 1.0 \
    --resolution 512 \
    --save_images

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} eval_metrics.py \
    --image_dir "${OUT}/s${SEED}/images" \
    --dataset_path dataset/drawbench \
    --output_dir "${OUT}/s${SEED}" \
    --batch_size 8
