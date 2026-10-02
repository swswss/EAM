#!/bin/bash
# Adjoint Matching baseline, single reward (raw PickScore, lambda 2000).
#
# Per-GPU batch 8 x accumulate 4 x num_train_timesteps 4 = 128 regressed states per optimizer step
# (our runs used batch 16 x accumulate 2 on 80GB GPUs, same number of states per step).
# As in our runs, gradients are not averaged across GPUs (each GPU trains its own LoRA on its data
# shard; the rank-0 adapter is saved). Add --sync_grads for standard data-parallel training.
# 40-step memoryless-SDE trajectories; one epoch = 199 optimizer steps.
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29502}

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_am.py \
    --config config/sd3.py \
    --embedding_dir dataset/pickscore/embeddings \
    --checkpoint_path checkpoints \
    --reward_fn pickscore \
    --lambda_reward 2000.0 \
    --training_nfe 40 \
    --num_train_timesteps 4 \
    --batch_size 8 \
    --accumulate_iter 4 \
    --bs_reward 4 \
    --lr 1e-4 \
    --num_epochs 1 \
    --ckpt_every 1
