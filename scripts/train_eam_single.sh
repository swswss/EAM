#!/bin/bash
# EAM, single reward (PickScore) — default configuration.
#
# Hardware: 4 GPUs (A100 40GB). Per-GPU batch 8 x accumulate 16 = 128 prompts per optimizer step.
# As in our runs, gradients are not averaged across GPUs: each GPU updates its own LoRA on its
# 1/4 shard of the prompts (49 optimizer steps per epoch) and the rank-0 adapter is saved.
# Add --sync_grads for standard data-parallel training (512 prompts per step on 4 GPUs).
#
# Reward: PickScore enters as PickScore/26, so --lambda_reward 52000 = 2000 x raw PickScore.
#
# Epsilon term: --eps_scale 0.0 removes the epsilon term from the regression target.
# Setting eps_scale=0 empirically shows better results since it does not involve stochasticity.
# (The epsilon term is switched off by --eps_scale, not by --c; --c 1 keeps the frozen
#  base-model velocity in the target.)
#
# LoRA checkpoints are saved every 2 optimizer steps from iter 300.
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${MASTER_PORT} train_eam.py \
    --config config/sd3.py \
    --embedding_dir dataset/pickscore/embeddings \
    --checkpoint_path checkpoints \
    --reward_model pickscore \
    --lambda_reward 52000.0 \
    --c 1 \
    --eps_scale 0.0 \
    --reward_coeff_max 100 \
    --training_nfe 40 \
    --eval_nfe 10 \
    --discrete_timesteps 1000 \
    --batch_size 8 \
    --accumulate_iter 16 \
    --bs_reward 4 \
    --lr 1e-4 \
    --num_epochs 2 \
    --ckpt_start_iter 300 \
    --ckpt_iter_step 32 \
    --eval_reward
