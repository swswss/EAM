# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
EAM fine-tuning of SD3.5-Medium (LoRA) with a differentiable reward.

One training iteration (per GPU, per dataloader batch):
  1. Sample x1 from the current policy with the deterministic DPM-Solver (no grad).
  2. reward_grad = lambda_reward * ∇_{x1} R(VAE.decode(x1)).
  3. Draw one t per sample from the discrete scheduler grid (stratified: half from the
     noise side, half from the image side) and eps ~ N(0, I); x_t = t * x1 + (1 - t) * eps.
  4. Regress the LoRA velocity onto the target (t = 1: data, t = 0: noise, D = 2c t^2 - 2t + 1)
         v_pred ≈ coeff_base * v_base + coeff_cv * (eps - x1) - eps_scale * coeff_e * eps - coeff_r * reward_grad
     with coeff_base = (2c-1) t^2 / D, coeff_cv = (1-t)^2 / D, coeff_e = (2c-1) t (1-t) / D,
     coeff_r = min((1-t)^0.9 / max(t, 0.02)^1.5, reward_coeff_max), and v_base the frozen
     base-model velocity. Each sample's loss is weighted by max(t, 0.02)^1.5 / (1-t)^0.9.
"""

import argparse
import logging
import os
from functools import partial

import torch
import torch.distributed as dist
import tqdm
from torch.amp import GradScaler, autocast as torch_autocast
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.functional import to_pil_image, to_tensor

from common import (
    EmbeddingDataset, collate_embedding_batch, load_negative_embeddings,
    setup_distributed, cleanup_distributed, is_main_process, set_seed, load_config,
    load_sd3_pipeline, add_lora_and_wrap_ddp, save_lora, average_gradients,
)
from eam_utils import compute_reward_grad, move_scorer
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.pickscore_scorer import PickScoreScorer

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="EAM fine-tuning of SD3.5-Medium.")
    # Data / IO
    parser.add_argument("--config", type=str, default="config/sd3.py")
    parser.add_argument("--embedding_dir", type=str, default="dataset/pickscore/embeddings",
                        help="Pre-computed text embeddings (see text_embedding_store.py).")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
                        help="Root directory; the run is saved to <checkpoint_path>/<run_name>.")
    parser.add_argument("--postfix", type=str, default=None, help="Suffix appended to the run name.")
    parser.add_argument("--seed", type=int, default=None, help="Override config.seed.")
    # Optimization
    parser.add_argument("--num_epochs", type=int, default=None, help="Override config.num_epochs.")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Stop after this many optimizer steps (for quick tests).")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-GPU batch size (config.sample.train_batch_size).")
    parser.add_argument("--accumulate_iter", type=int, default=1,
                        help="Gradient accumulation: optimizer.step() every this many batches.")
    parser.add_argument("--lr", type=float, default=None, help="Override config.train.learning_rate.")
    parser.add_argument("--sync_grads", action="store_true", default=False,
                        help="Average LoRA gradients across GPUs (data-parallel training, effective batch = "
                             "batch_size x accumulate_iter x num_gpus). Off by default, which is how the reported "
                             "models were trained: every GPU updates its own copy on its data shard "
                             "(batch_size x accumulate_iter per step) and the rank-0 adapter is saved.")
    parser.add_argument("--training_nfe", type=int, default=10, help="Sampler steps used to draw x1 during training.")
    parser.add_argument("--eval_nfe", type=int, default=10, help="Sampler steps for the in-training eval images.")
    parser.add_argument("--discrete_timesteps", type=int, default=1000,
                        help="t is drawn from the scheduler's N-step sigma grid (t = 1 - sigma, t = 0 dropped).")
    # EAM objective
    parser.add_argument("--lambda_reward", type=float, default=52000.0,
                        help="Reward scale. PickScore enters as PickScore/26, so 52000 = 2000 x raw PickScore.")
    parser.add_argument("--c", type=float, default=1.0,
                        help="EAM coefficient c (see module docstring). c < 0.50001 forces coeff_base=0, "
                             "coeff_cv=1; c > 100 forces coeff_base=1, coeff_cv=0.")
    parser.add_argument("--eps_scale", type=float, default=0.0,
                        help="Weight of the epsilon term (0 removes it).")
    parser.add_argument("--reward_coeff_max", type=float, default=100.0, help="Upper clamp of coeff_r.")
    # Reward
    parser.add_argument("--reward_model", type=str, default="pickscore", choices=["pickscore", "multi"],
                        help="'pickscore': PickScore/26. 'multi': PickScore/26 + clip_scale*CLIPScore/100 "
                             "+ hpsv2_scale*HPSv2.1 + aesthetic_scale*Aesthetic (scorers with weight 0 are not loaded).")
    parser.add_argument("--clip_scale", type=float, default=0.0, help="CLIPScore weight for --reward_model multi.")
    parser.add_argument("--hpsv2_scale", type=float, default=0.5, help="HPSv2.1 weight for --reward_model multi.")
    parser.add_argument("--aesthetic_scale", type=float, default=0.02, help="Aesthetic weight for --reward_model multi.")
    parser.add_argument("--bs_reward", type=int, default=4, help="Chunk size for the reward-gradient pass (reduce if OOM).")
    # Checkpointing / logging
    parser.add_argument("--ckpt_start_iter", type=int, default=0,
                        help="First dataloader iter (cumulative over epochs) at which LoRA checkpoints are saved.")
    parser.add_argument("--ckpt_iter_step", type=int, default=None,
                        help="Save a LoRA checkpoint every this many dataloader iters (default: every optimizer step).")
    parser.add_argument("--eval_reward", action="store_true", default=False,
                        help="Score the fixed in-training eval images with the training reward and append to <run>/log.txt.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.batch_size is not None:
        config.sample.train_batch_size = args.batch_size
    config.sample.num_steps = args.training_nfe
    config.sample.eval_num_steps = args.eval_nfe
    if args.lr is not None:
        config.train.learning_rate = args.lr
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.seed is not None:
        config.seed = args.seed
    if args.ckpt_iter_step is None:
        args.ckpt_iter_step = args.accumulate_iter

    # --- Distributed setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    # --- Run directory ---
    effective_bs = config.sample.train_batch_size * args.accumulate_iter * (world_size if args.sync_grads else 1)
    run_name = (
        f"eam_{args.reward_model}_rwd_{args.lambda_reward}_lr_{config.train.learning_rate}"
        f"_nfe_{args.training_nfe}_bs_{effective_bs}_c{args.c:g}_eps{args.eps_scale:g}"
    )
    if args.sync_grads:
        run_name = f"{run_name}_sync"
    if args.postfix is not None:
        run_name = f"{run_name}_{args.postfix}"
    run_dir = os.path.join(args.checkpoint_path, run_name)
    samples_dir = os.path.join(run_dir, "samples")
    if is_main_process(rank):
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(samples_dir, exist_ok=True)

    logger.info(f"\n{config}")
    logger.info(f"args: {vars(args)}")

    set_seed(config.seed, rank)

    # --- Mixed precision ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler("cuda", enabled=(mixed_precision_dtype == torch.float16))
    embed_dtype = mixed_precision_dtype or torch.float32

    # --- Model + LoRA ---
    pipeline = load_sd3_pipeline(config, device, rank)
    transformer_ddp, transformer_trainable_parameters = add_lora_and_wrap_ddp(pipeline, config, device, local_rank)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Data ---
    train_dataset = EmbeddingDataset("train", args.embedding_dir)
    test_dataset = EmbeddingDataset("test", args.embedding_dir)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.sample.train_batch_size,
        sampler=train_sampler,
        collate_fn=collate_embedding_batch,
        num_workers=8,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        sampler=test_sampler,
        collate_fn=collate_embedding_batch,
        num_workers=8,
        pin_memory=True,
    )

    # Empty-prompt embeddings: only needed by the sampler's API (guidance_scale=1.0 => no CFG).
    neg_prompt_embed, neg_pooled_prompt_embed = load_negative_embeddings(args.embedding_dir, device, embed_dtype)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)

    # --- Reward models (kept on CPU; moved to GPU only for the reward-gradient pass) ---
    train_scorers = {}
    pickscore_scorer = PickScoreScorer(device="cpu", dtype=torch.float32)
    for p in pickscore_scorer.model.parameters():
        p.requires_grad_(False)
    train_scorers["pickscore"] = pickscore_scorer
    scorer_weights = {"pickscore": 1.0}
    if args.reward_model == "multi":
        aux_weights = {"clipscore": args.clip_scale, "aesthetic": args.aesthetic_scale, "hpsv2": args.hpsv2_scale}
        aux_weights = {k: v for k, v in aux_weights.items() if v != 0.0}

        def _load_aux():
            from flow_grpo.aesthetic_scorer import AestheticScorer
            from flow_grpo.clip_scorer import ClipScorer
            from flow_grpo.hpsv2_scorer import HPSv2Scorer
            out = {}
            if "clipscore" in aux_weights:
                out["clipscore"] = ClipScorer(device="cpu")
            if "aesthetic" in aux_weights:
                out["aesthetic"] = AestheticScorer(dtype=torch.float32, device="cpu")
            if "hpsv2" in aux_weights:
                out["hpsv2"] = HPSv2Scorer(dtype=torch.float32, device="cpu")
            return out

        # rank 0 downloads first, the other ranks then load from the cache
        if rank == 0:
            aux_scorers = _load_aux()
        if world_size > 1:
            dist.barrier()
        if rank != 0:
            aux_scorers = _load_aux()
        for name, scorer in aux_scorers.items():
            module = scorer if name == "aesthetic" else scorer.model
            for p in module.parameters():
                p.requires_grad_(False)
            train_scorers[name] = scorer
        scorer_weights.update(aux_weights)
    logger.info(f"Training reward = {args.reward_model}, weights = {scorer_weights}")

    logger.info("***** Running EAM training *****")
    logger.info(f"  Num epochs = {config.num_epochs}")
    logger.info(f"  Per-GPU batch size = {config.sample.train_batch_size}, accumulate = {args.accumulate_iter}, "
                f"world size = {world_size}, sync_grads = {args.sync_grads}")

    # --- Fixed eval prompts/noises for the in-training eval (rank 0): 8 prompts x 4 seeds ---
    n_prompts = 8
    n_seeds_per_prompt = 4
    latent_channels = pipeline.transformer.config.in_channels  # 16
    latent_size = config.resolution // 8                       # 64 for 512px

    _eval_iter = iter(test_dataloader)
    _eval_batch = next(_eval_iter)
    base_prompt_embeds = _eval_batch["prompt_embeds"][:n_prompts].to(device=device, dtype=embed_dtype)
    base_pooled_embeds = _eval_batch["pooled_prompt_embeds"][:n_prompts].to(device=device, dtype=embed_dtype)
    eval_prompts_text = _eval_batch["prompts"][:n_prompts]
    n_prompts = base_prompt_embeds.shape[0]
    del _eval_iter

    # Seed of (prompt p, seed s) = config.seed + p + s * n_prompts (all distinct).
    eval_noises = torch.stack([
        torch.randn(
            latent_channels, latent_size, latent_size,
            generator=torch.Generator().manual_seed(config.seed + p + s * n_prompts),
        )
        for p in range(n_prompts)
        for s in range(n_seeds_per_prompt)
    ], dim=0)
    eval_prompt_embeds = base_prompt_embeds.repeat_interleave(n_seeds_per_prompt, dim=0)
    eval_pooled_embeds = base_pooled_embeds.repeat_interleave(n_seeds_per_prompt, dim=0)
    n_eval = n_prompts * n_seeds_per_prompt
    eval_neg_e = neg_prompt_embed.repeat(n_eval, 1, 1)
    eval_neg_p = neg_pooled_prompt_embed.repeat(n_eval, 1)

    def build_sample_kwargs(prompt_embeds, pooled_prompt_embeds, neg_embeds, neg_pooled_embeds, batch_size):
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": neg_embeds[:batch_size],
            "negative_pooled_prompt_embeds": neg_pooled_embeds[:batch_size],
            "num_inference_steps": config.sample.num_steps,
            "guidance_scale": config.sample.guidance_scale,
            "output_type": "pt",
            "height": config.resolution,
            "width": config.resolution,
            "noise_level": config.sample.noise_level,
            "deterministic": config.sample.deterministic,
            "solver": config.sample.solver,
        }

    def run_eval(tag):
        """Generate the fixed eval images on rank 0 (RNG state is preserved)."""
        if not is_main_process(rank):
            return
        rng_cpu = torch.random.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(device)
        was_training = pipeline.transformer.training
        pipeline.transformer.eval()

        eval_dir = os.path.join(samples_dir, tag)
        os.makedirs(eval_dir, exist_ok=True)
        eval_kwargs = build_sample_kwargs(eval_prompt_embeds, eval_pooled_embeds, eval_neg_e, eval_neg_p, n_eval)
        eval_kwargs["num_inference_steps"] = config.sample.eval_num_steps
        eval_kwargs["latents"] = eval_noises.to(device=device, dtype=embed_dtype)
        with torch.no_grad():
            with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                eval_images, _, _ = pipeline_with_logprob(pipeline, **eval_kwargs)

        pil_images = []
        for j, img in enumerate(eval_images[:n_eval]):
            img = to_pil_image(img.float().clamp(0, 1))
            img.save(os.path.join(eval_dir, f"img_p{j // n_seeds_per_prompt:02d}_s{j % n_seeds_per_prompt:02d}.png"))
            pil_images.append(img)

        if args.eval_reward:
            prompts_rep = [eval_prompts_text[j // n_seeds_per_prompt] for j in range(n_eval)]
            for scorer in train_scorers.values():
                move_scorer(scorer, device)
            bchw = None
            per_scorer = {}
            with torch.no_grad():
                for name, scorer in train_scorers.items():
                    if name == "pickscore":
                        s = scorer(prompts_rep, pil_images)  # already /26
                        if len(train_scorers) == 1:
                            s = s * 26.0                     # report raw PickScore (~20)
                    elif name == "aesthetic":
                        s = scorer(pil_images)
                    else:  # clipscore, hpsv2 take BCHW tensors in [0, 1]
                        if bchw is None:
                            bchw = torch.stack([to_tensor(img) for img in pil_images]).to(device)
                        s = scorer(bchw, prompts_rep)
                    per_scorer[name] = [float(v) for v in s.cpu().tolist()]
            for scorer in train_scorers.values():
                move_scorer(scorer, "cpu")
            torch.cuda.empty_cache()

            scores = [sum(v[j] for v in per_scorer.values()) for j in range(n_eval)]
            mean_score = sum(scores) / n_eval
            per_prompt = [
                sum(scores[p * n_seeds_per_prompt:(p + 1) * n_seeds_per_prompt]) / n_seeds_per_prompt
                for p in range(n_prompts)
            ]
            with open(os.path.join(run_dir, "log.txt"), "a") as f:
                f.write(f"{tag}: " + ", ".join(f"{v:.4f}" for v in [mean_score] + per_prompt) + "\n")
                f.write(", ".join(f"{v:.4f}" for v in scores) + "\n")
                if len(per_scorer) > 1:
                    parts = []
                    for name, vals in per_scorer.items():
                        m = sum(vals) / len(vals)
                        parts.append(f"{name}={m * 26.0 if name == 'pickscore' else m:.4f}")
                    f.write(f"  breakdown: {', '.join(parts)}\n")

        if was_training:
            pipeline.transformer.train()
        torch.random.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)

    optimizer.zero_grad()

    # --- Discrete t grid (AM convention: t = 1 - sigma; t = 1 clean, t = 0 noise) ---
    pipeline.scheduler.set_timesteps(args.discrete_timesteps, device=device)
    sched_sigmas = pipeline.scheduler.sigmas[:args.discrete_timesteps].float()  # drop trailing 0
    discrete_t_am_grid = (1.0 - sched_sigmas).to(device)
    discrete_t_am_grid = discrete_t_am_grid[discrete_t_am_grid > 0]  # drop t=0 (pure noise)
    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=device)
    logger.info(f"Discrete t grid: N={len(discrete_t_am_grid)}, "
                f"min={discrete_t_am_grid.min().item():.4f}, max={discrete_t_am_grid.max().item():.4f}")

    # --- One dataloader batch -> (x1, reward_grad) ---
    def sample_batch(batch_data):
        transformer_ddp.module.set_adapter("default")
        prompts_b = batch_data["prompts"]
        prompt_embeds_b = batch_data["prompt_embeds"].to(device=device, dtype=embed_dtype)
        pooled_prompt_embeds_b = batch_data["pooled_prompt_embeds"].to(device=device, dtype=embed_dtype)

        with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
            with torch.no_grad():
                sample_kwargs = build_sample_kwargs(
                    prompt_embeds_b, pooled_prompt_embeds_b,
                    sample_neg_prompt_embeds[:len(prompts_b)],
                    sample_neg_pooled_prompt_embeds[:len(prompts_b)],
                    len(prompts_b),
                )
                _, latents_b, _ = pipeline_with_logprob(pipeline, **sample_kwargs)
                x1_b = latents_b[-1]

        # Reward gradient: transformer -> CPU, reward models -> GPU
        transformer_ddp.module.to("cpu")
        for s in train_scorers.values():
            move_scorer(s, device)
        torch.cuda.empty_cache()
        if args.lambda_reward > 0.0:
            if args.reward_model == "multi":
                # per-scorer gradients, combined as g_pick + sum_k w_k * g_k
                per_grads = {
                    name: compute_reward_grad(x1_b, prompts_b, pipeline.vae, {name: scorer},
                                              bs_reward=args.bs_reward, lambda_r=1.0)
                    for name, scorer in train_scorers.items()
                }
                reward_grad_b = per_grads["pickscore"].clone()
                for name in train_scorers:
                    if name != "pickscore":
                        reward_grad_b = reward_grad_b + scorer_weights[name] * per_grads[name]
                reward_grad_b = args.lambda_reward * reward_grad_b
            else:
                reward_grad_b = compute_reward_grad(x1_b, prompts_b, pipeline.vae, train_scorers,
                                                    bs_reward=args.bs_reward, lambda_r=args.lambda_reward)
        else:
            reward_grad_b = torch.zeros_like(x1_b)
        for s in train_scorers.values():
            move_scorer(s, "cpu")
        transformer_ddp.module.to(device)
        torch.cuda.empty_cache()

        return x1_b.detach(), reward_grad_b.detach(), prompt_embeds_b.detach(), pooled_prompt_embeds_b.detach()

    # --- EAM regression step on one batch (backward; optimizer.step on the last accumulation) ---
    def train_on_samples(x1, reward_grad, prompt_embeds, pooled_prompt_embeds, accumulate_idx):
        B = x1.shape[0]
        # Stratified t: half of the batch from the noise side (lower 60% of the sorted grid),
        # half from the image side (upper 40%).
        N = len(discrete_t_am_grid)
        sorted_grid, sort_idx = discrete_t_am_grid.sort()
        middle = max(1, min(round(N * 0.6), N - 1))
        B_image = B // 2
        B_noise = B - B_image
        noise_pos = torch.randint(0, middle, (B_noise,), device=device)
        image_pos = torch.randint(middle, N, (B_image,), device=device)
        pos = torch.cat([noise_pos, image_pos], dim=0)
        pos = pos[torch.randperm(B, device=device)]
        t_am = discrete_t_am_grid[sort_idx[pos]].to(x1.dtype).view(-1, 1, 1, 1)

        transformer_ddp.train()
        transformer_ddp.module.set_adapter("default")

        eps = torch.randn_like(x1)
        xt = t_am * x1 + (1 - t_am) * eps
        timestep_train = ((1.0 - t_am).view(B) * 1000).long()

        with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
            v_pred = transformer_ddp(
                hidden_states=xt.detach(),
                timestep=timestep_train,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

        # Frozen base-model velocity
        with torch.no_grad():
            with transformer_ddp.module.disable_adapter():
                with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                    v_base = transformer_ddp.module(
                        hidden_states=xt.detach(),
                        timestep=timestep_train,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]
            transformer_ddp.module.set_adapter("default")

        # Coefficients in fp64 ((2c-1) can be tiny), cast to fp32 before touching model outputs.
        C = args.c
        t_am64 = t_am.to(torch.float64)
        D64 = 2 * C * t_am64**2 - 2 * t_am64 + 1

        _t_denom = t_am64.clamp(min=0.02)
        coeff_r = ((1 - t_am64)**0.9 / (_t_denom**1.5)).to(torch.float32).clamp(max=args.reward_coeff_max)
        reward_match = coeff_r * reward_grad

        coeff_e = ((2 * C - 1) * t_am64 * (1 - t_am64) / D64).to(torch.float32)
        eps_match = coeff_e * eps

        if C < 0.50001:
            coeff_cond_v = 1.0
        elif C > 100:
            coeff_cond_v = 0.0
        else:
            coeff_cond_v = ((1 - t_am64)**2 / D64).to(torch.float32)
        cond_v_match = coeff_cond_v * (eps - x1)

        if C < 0.50001:
            coeff_base = 0.0
        elif C > 100:
            coeff_base = 1.0
        else:
            coeff_base = ((2 * C - 1) * (t_am64**2) / D64).to(torch.float32)
        v_base_match = coeff_base * v_base

        loss_per_sample = 2.0 * (
            v_pred - v_base_match - cond_v_match + eps_match * args.eps_scale + reward_match
        ).pow(2).mean(dim=(1, 2, 3))  # [B]
        _t_denom_l = t_am64.clamp(min=0.02)
        loss_coeff = (_t_denom_l**1.5) / ((1 - t_am64)**0.9 + 1e-8)
        loss_per_sample = loss_coeff.view(-1).to(loss_per_sample.dtype) * loss_per_sample

        loss = loss_per_sample.mean()
        loss_scaled = loss / args.accumulate_iter

        is_last_accum = accumulate_idx == args.accumulate_iter - 1
        if mixed_precision_dtype == torch.float16:
            if is_last_accum:
                scaler.scale(loss_scaled).backward()
                if args.sync_grads:
                    average_gradients(transformer_trainable_parameters, world_size)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer_trainable_parameters, config.train.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                with transformer_ddp.no_sync():
                    scaler.scale(loss_scaled).backward()
        else:
            if is_last_accum:
                loss_scaled.backward()
                if args.sync_grads:
                    average_gradients(transformer_trainable_parameters, world_size)
                torch.nn.utils.clip_grad_norm_(transformer_trainable_parameters, config.train.max_grad_norm)
                optimizer.step()
            else:
                with transformer_ddp.no_sync():
                    loss_scaled.backward()
        return loss.item(), is_last_accum

    #######################
    #### TRAINING LOOP ####
    #######################
    global_step = 0
    training_done = False
    iters_per_epoch = len(train_dataloader)
    save_every_n_steps = max(1, args.ckpt_iter_step // args.accumulate_iter)
    for epoch in range(config.num_epochs):
        train_sampler.set_epoch(epoch)
        pipeline.transformer.eval()

        for i, batch_data in enumerate(tqdm(
            train_dataloader, desc=f"Epoch {epoch}", disable=not is_main_process(rank), position=0,
        )):
            x1, reward_grad, prompt_embeds, pooled_prompt_embeds = sample_batch(batch_data)

            if i % args.accumulate_iter == 0:
                optimizer.zero_grad()
            loss_val, is_last_accum = train_on_samples(
                x1, reward_grad, prompt_embeds, pooled_prompt_embeds,
                accumulate_idx=i % args.accumulate_iter,
            )

            if is_last_accum:
                global_step += 1
                logger.info(f"[rank{rank}] epoch={epoch} iter={i} step={global_step}  loss={loss_val:.6f}")
                run_eval(f"epoch{epoch:04d}iter{i:04d}")
                if world_size > 1:
                    dist.barrier()
                cum_iter = epoch * iters_per_epoch + i
                if cum_iter >= args.ckpt_start_iter and global_step % save_every_n_steps == 0:
                    if is_main_process(rank):
                        save_lora(transformer_ddp, run_dir, f"iter_e{epoch:04d}_i{i:06d}", logger)
                    if world_size > 1:
                        dist.barrier()
                if args.max_iter is not None and global_step >= args.max_iter:
                    training_done = True
                    break
        if world_size > 1:
            dist.barrier()
        if training_done:
            logger.info(f"Reached max_iter={args.max_iter}, stopping.")
            break

    if is_main_process(rank):
        save_lora(transformer_ddp, run_dir, "final", logger)
    if world_size > 1:
        dist.barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
