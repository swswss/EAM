#!/bin/bash
# Adjoint Matching baseline, multi reward:
#   2.0 * PickScore/26 + 0.0 * CLIPScore/100 + 1.0 * HPSv2.1 + 0.04 * Aesthetic/10   (lambda 2000)
#
# Per-GPU batch 8 x accumulate 4 x num_train_timesteps 4 = 128 regressed states per optimizer step
# (our runs used batch 16 x accumulate 2 on 80GB GPUs, same number of states per step).
# As in our runs, gradients are not averaged across GPUs (each GPU trains its own LoRA on its data
# shard; the rank-0 adapter is saved). Add --sync_grads for standard data-parallel training.
# Requires the reward checkpoints: bash scripts/download_reward_ckpts.sh
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29503}

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_am.py \
    --config config/sd3.py \
    --embedding_dir dataset/pickscore/embeddings \
    --checkpoint_path checkpoints \
    --reward_fn pickscore_clipscore_hpsv2_aesthetics \
    --reward_ratios 2.0 0.0 1.0 0.04 \
    --lambda_reward 2000.0 \
    --training_nfe 40 \
    --num_train_timesteps 4 \
    --batch_size 8 \
    --accumulate_iter 4 \
    --bs_reward 4 \
    --lr 1e-4 \
    --num_epochs 1 \
    --ckpt_every 1
