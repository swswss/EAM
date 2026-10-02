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
Adjoint Matching (AM) baseline for SD3.5-Medium (LoRA), ported to flow matching.

One training iteration (per GPU, per dataloader batch):
  1. Roll out the current policy with the memoryless flow SDE (training_nfe steps) and keep
     the whole trajectory {x_k}.
  2. Terminal adjoint a_N = -lambda_reward * ∇_{x_N} R(VAE.decode(x_N)).
  3. Backward adjoint ODE with the frozen base model (LoRA disabled):
         a_k = a_{k+1} + ∇_{x_k} < step_base(x_k) - x_k , a_{k+1} >   (one VJP per step)
  4. Pick num_train_timesteps indices per trajectory (stratified around 60% of the schedule) and
     regress the control onto the adjoint:  loss = || control_k * sqrt(dt) + std_k * a_k ||^2.
"""

import argparse
import json
import logging
import os
import shutil
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.amp import GradScaler, autocast as torch_autocast
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.functional import to_pil_image

from common import (
    EmbeddingDataset, collate_embedding_batch, load_negative_embeddings,
    setup_distributed, cleanup_distributed, is_main_process, set_seed, load_config,
    load_sd3_pipeline, add_lora_and_wrap_ddp, save_lora, average_gradients,
)
from am_utils import build_reward_scorers, move_reward_scorers, compute_reward_grad, compute_reward_scores
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.solver import flow_grpo_step

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Adjoint Matching fine-tuning of SD3.5-Medium.")
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
                        help="Prompt batches per optimizer step. optimizer.step() is called every "
                             "accumulate_iter * num_train_timesteps backward passes.")
    parser.add_argument("--num_train_timesteps", type=int, default=4,
                        help="Number of trajectory states (timesteps) regressed per prompt batch.")
    parser.add_argument("--lr", type=float, default=None, help="Override config.train.learning_rate.")
    parser.add_argument("--sync_grads", action="store_true", default=False,
                        help="Average LoRA gradients across GPUs (data-parallel training). Off by default, which is "
                             "how the reported models were trained: every GPU updates its own copy on its data shard "
                             "and the rank-0 adapter is saved.")
    parser.add_argument("--training_nfe", type=int, default=40, help="SDE steps of the training trajectories.")
    # Reward
    parser.add_argument("--lambda_reward", type=float, default=2000.0)
    parser.add_argument("--reward_fn", type=str, default="pickscore",
                        choices=["pickscore", "imagereward", "pickscore_clipscore_hpsv2",
                                 "pickscore_clipscore_hpsv2_aesthetics"],
                        help="'pickscore': raw PickScore. Multi rewards: r0*PickScore/26 + r1*CLIPScore/100 "
                             "+ r2*HPSv2.1 (+ r3*Aesthetic/10) with r = --reward_ratios.")
    parser.add_argument("--reward_ratios", type=float, nargs="+", default=None,
                        help="Weights of the multi reward (3 or 4 values, in the order of --reward_fn).")
    parser.add_argument("--bs_reward", type=int, default=4, help="Chunk size for the reward-gradient pass (reduce if OOM).")
    parser.add_argument("--keep_transformer_on_gpu", action="store_true", default=False,
                        help="Keep the transformer on GPU during the reward-gradient pass (default: offload to CPU).")
    # Checkpointing
    parser.add_argument("--ckpt_every", type=int, default=1, help="Save a LoRA checkpoint every N optimizer steps.")
    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.batch_size is not None:
        config.sample.train_batch_size = args.batch_size
    config.sample.num_steps = args.training_nfe
    if args.lr is not None:
        config.train.learning_rate = args.lr
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    if args.seed is not None:
        config.seed = args.seed
    # AM trajectories are sampled with the memoryless SDE.
    config.sample.solver = "flow"
    config.sample.deterministic = False

    # --- Distributed setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    # --- Run directory ---
    resolved_batch_size = config.sample.train_batch_size
    effective_bs = resolved_batch_size * args.accumulate_iter * args.num_train_timesteps * (world_size if args.sync_grads else 1)
    run_name = (
        f"am_{args.reward_fn}_lambda_{args.lambda_reward}_bs_{effective_bs}_lr_{config.train.learning_rate}"
        f"_numtime_{args.num_train_timesteps}_nfe_{args.training_nfe}"
    )
    if args.reward_ratios is not None:
        run_name = f"{run_name}_ratios_{'_'.join(f'{r:g}' for r in args.reward_ratios)}"
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
        batch_size=config.sample.train_batch_size,
        sampler=test_sampler,
        collate_fn=collate_embedding_batch,
        num_workers=8,
        pin_memory=True,
    )
    neg_prompt_embed, neg_pooled_prompt_embed = load_negative_embeddings(args.embedding_dir, device, embed_dtype)

    # --- Reward model(s): kept on CPU, moved to GPU only for the reward-gradient pass ---
    reward_scorers = build_reward_scorers(args.reward_fn, device="cpu", dtype=torch.float32)
    logger.info(f"Loaded reward model(s) for {args.reward_fn} (ratios={args.reward_ratios}) on CPU (frozen).")

    logger.info("***** Running AM training *****")
    logger.info(f"  Num epochs = {config.num_epochs}")
    logger.info(f"  Per-GPU batch size = {config.sample.train_batch_size}, accumulate = {args.accumulate_iter}, "
                f"num_train_timesteps = {args.num_train_timesteps}, world size = {world_size}, "
                f"sync_grads = {args.sync_grads}")

    # --- Fixed eval prompts/noises for the in-training eval (rank 0) ---
    n_eval = resolved_batch_size
    latent_channels = pipeline.transformer.config.in_channels
    latent_size = config.resolution // 8
    eval_noises = torch.stack([
        torch.randn(
            latent_channels, latent_size, latent_size,
            generator=torch.Generator().manual_seed(config.seed + i),
        )
        for i in range(n_eval)
    ], dim=0)
    _eval_iter = iter(test_dataloader)
    _eval_batch = next(_eval_iter)
    eval_prompt_embeds = _eval_batch["prompt_embeds"][:n_eval].to(device=device, dtype=embed_dtype)
    eval_pooled_embeds = _eval_batch["pooled_prompt_embeds"][:n_eval].to(device=device, dtype=embed_dtype)
    eval_prompts_text = _eval_batch["prompts"][:n_eval]
    n_eval = eval_prompt_embeds.shape[0]
    del _eval_iter
    eval_noises = eval_noises[:n_eval]
    eval_neg_e = neg_prompt_embed.repeat(n_eval, 1, 1)
    eval_neg_p = neg_pooled_prompt_embed.repeat(n_eval, 1)
    if is_main_process(rank):
        with open(os.path.join(run_dir, "prompts.txt"), "w") as f:
            for prompt in eval_prompts_text:
                f.write(f"{prompt}\n")
        copied_config_path = os.path.join(run_dir, os.path.basename(args.config))
        shutil.copy2(args.config, copied_config_path)
        with open(copied_config_path, "a") as f:
            f.write("\n\n# ---- Runtime CLI args ----\nRUNTIME_ARGS = ")
            json.dump(vars(args), f, indent=2, sort_keys=True)
            f.write("\n")

    def save_image_batch(images, output_dir, prefix):
        os.makedirs(output_dir, exist_ok=True)
        for j, img in enumerate(images):
            to_pil_image(img.detach().float().cpu().clamp(0, 1)).save(os.path.join(output_dir, f"{prefix}_{j:03d}.png"))

    def decode_latents_to_images(latents):
        scaling_factor = pipeline.vae.config.scaling_factor
        shift_factor = getattr(pipeline.vae.config, "shift_factor", 0.0)
        latents = latents.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            decoded = pipeline.vae.decode((latents / scaling_factor) + shift_factor, return_dict=False)[0]
        return (decoded * 0.5 + 0.5).clamp(0, 1)

    def build_sample_kwargs(prompt_embeds, pooled_prompt_embeds, neg_prompt_embeds, neg_pooled_prompt_embeds, batch_size):
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": neg_prompt_embeds[:batch_size],
            "negative_pooled_prompt_embeds": neg_pooled_prompt_embeds[:batch_size],
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
        """Generate the fixed eval images on rank 0 with the deterministic ODE (dpm2, 10 steps)."""
        if not is_main_process(rank):
            return
        rng_cpu = torch.random.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state(device)
        was_training = pipeline.transformer.training
        pipeline.transformer.eval()

        eval_dir = os.path.join(samples_dir, tag)
        eval_kwargs = build_sample_kwargs(eval_prompt_embeds, eval_pooled_embeds, eval_neg_e, eval_neg_p, n_eval)
        eval_kwargs.update({"num_inference_steps": 10, "solver": "dpm2", "deterministic": True})
        eval_kwargs["latents"] = eval_noises.to(device=device, dtype=embed_dtype)
        with torch.no_grad():
            with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                eval_images, _, _ = pipeline_with_logprob(pipeline, **eval_kwargs)

        save_image_batch(eval_images[:n_eval], eval_dir, "img")
        move_reward_scorers(reward_scorers, device)
        with torch.no_grad():
            reward_values = compute_reward_scores(
                eval_images[:n_eval].detach().float().to(device),
                eval_prompts_text[:n_eval],
                reward_scorers,
                args.reward_fn,
                args.reward_ratios,
            )
        reward_values = reward_values.detach().float().cpu().tolist()
        with open(os.path.join(eval_dir, "reward_values.json"), "w") as f:
            json.dump(
                {
                    "reward_fn": args.reward_fn,
                    "values": [
                        {"index": idx, "image": f"img_{idx:03d}.png", "prompt": prompt, "reward": reward}
                        for idx, (prompt, reward) in enumerate(zip(eval_prompts_text[:n_eval], reward_values))
                    ],
                },
                f,
                indent=2,
            )
        move_reward_scorers(reward_scorers, "cpu")
        torch.cuda.empty_cache()
        logger.info(f"Saved {n_eval} eval images to {eval_dir} (mean reward {np.mean(reward_values):.4f})")

        if was_training:
            pipeline.transformer.train()
        torch.random.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)

    optimizer.zero_grad()

    effective_grad_accum = args.accumulate_iter * args.num_train_timesteps
    total_backward_count = 0
    num_batches_per_epoch = len(train_dataloader)

    def sample_time_indices(all_t, all_x_t, num_timesteps_to_load):
        """Stratified timestep indices: half from the first 60% of the trajectory (high noise),
        half from the remaining 40% (more structure). Returned sorted ascending."""
        num_timesteps = len(all_t)
        num_timesteps_to_load = min(num_timesteps_to_load, num_timesteps)
        if num_timesteps_to_load == num_timesteps:
            return torch.arange(num_timesteps, dtype=torch.long, device=all_x_t.device)
        if num_timesteps <= 1:
            return torch.zeros(num_timesteps_to_load, dtype=torch.long, device=all_x_t.device)
        middle_timestep = round(num_timesteps * 0.6)
        if middle_timestep <= 0 or middle_timestep >= num_timesteps:
            indices_t = np.random.choice(np.arange(0, num_timesteps), num_timesteps_to_load, replace=False)
            indices_t = np.sort(indices_t)
            return torch.tensor(indices_t, dtype=torch.long, device=all_x_t.device)
        indices_t_1 = np.random.choice(
            np.arange(0, middle_timestep), num_timesteps_to_load // 2, replace=False
        )
        indices_t_2 = np.random.choice(
            np.arange(middle_timestep, num_timesteps),
            num_timesteps_to_load - num_timesteps_to_load // 2,
            replace=False,
        )
        indices_t = np.sort(np.concatenate((indices_t_1, indices_t_2)))
        return torch.tensor(indices_t, dtype=torch.long, device=all_x_t.device)

    def transformer_forward(transformer_module, hidden_states, timestep, prompt_embeds, pooled_prompt_embeds):
        timestep = timestep.expand(hidden_states.shape[0]) if timestep.ndim == 0 else timestep.reshape(-1)
        return transformer_module(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

    def grad_inner_product(transformer_module, x, t, adj_t, prompt_embeds, pooled_prompt_embeds, sigmas, index, eta):
        """VJP of the frozen base-model step: ∇_x < step_base(x) - x , adj_t >, plus the base velocity."""
        def inner_product(x):
            with transformer_module.disable_adapter():
                with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                    v_pred = transformer_forward(transformer_module, x, t, prompt_embeds, pooled_prompt_embeds)
                prev_sample_init, _, _, _, _ = flow_grpo_step(
                    v_pred, x, eta, sigmas, index, None, deterministic=True, return_stats=True
                )
            sum_inner_prod = torch.sum((prev_sample_init - x) * adj_t, dim=tuple(range(adj_t.ndim)))
            return sum_inner_prod, v_pred
        with torch.enable_grad():
            x = x.requires_grad_(True)
            sum_inner_prod, noise_pred = inner_product(x)
            output = torch.autograd.grad(sum_inner_prod, x)[0]
            x = x.requires_grad_(False)
            return output.detach(), noise_pred.detach()

    def compute_adjoints(transformer_module, all_x_t, all_t, prompt_embeds, pooled_prompt_embeds, reward_grad, sigmas, eta):
        """Backward adjoint simulation. Returns the adjoint states [B, nfe+1, ...] and the base velocities."""
        with torch.no_grad():
            assert all_x_t[:, :-1].shape[1] == len(all_t)
            adjoint_states = torch.zeros_like(all_x_t)
            num_states = all_x_t.shape[1]
            all_v_pred_init = torch.zeros_like(all_x_t[:, :-1])
            a = -reward_grad
            adjoint_states[:, -1] = a
            for k in range(num_states - 2, -1, -1):
                grad_inner_prod, v_pred_init = grad_inner_product(
                    transformer_module, all_x_t[:, k], all_t[k], a,
                    prompt_embeds, pooled_prompt_embeds, sigmas, k, eta,
                )
                a += grad_inner_prod
                adjoint_states[:, k] = a
                all_v_pred_init[:, k] = v_pred_init
                del grad_inner_prod, v_pred_init
            return adjoint_states, all_v_pred_init

    def evaluate_controls(transformer, prompt_embeds, pooled_prompt_embeds, xt, timestep_train, v_pred_init, sigmas, index, eta):
        """control * sqrt(dt) at state xt: difference between the policy step and the base step."""
        with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
            v_pred_eval = transformer_forward(transformer, xt.detach(), timestep_train, prompt_embeds, pooled_prompt_embeds)
        prev_sample, _, _, _, std_dev_t = flow_grpo_step(
            v_pred_eval, xt, eta, sigmas, index, None, deterministic=True, return_stats=True
        )
        prev_sample_init, _, _, _, _ = flow_grpo_step(
            v_pred_init, xt, eta, sigmas, index, None, deterministic=True, return_stats=True
        )
        control_times_sqrt_dt = 2 * (prev_sample - prev_sample_init) / std_dev_t
        return control_times_sqrt_dt, prev_sample, std_dev_t

    #######################
    #### TRAINING LOOP ####
    #######################
    global_step = 0
    training_done = False
    for epoch in range(config.num_epochs):
        train_sampler.set_epoch(epoch)
        pipeline.transformer.eval()
        train_iter = iter(train_dataloader)

        for i in tqdm(range(num_batches_per_epoch), desc=f"Epoch {epoch}", disable=not is_main_process(rank), position=0):
            batch_data = next(train_iter)
            transformer_ddp.module.set_adapter("default")
            prompts = batch_data["prompts"]
            prompt_embeds = batch_data["prompt_embeds"].to(device=device, dtype=embed_dtype)
            pooled_prompt_embeds = batch_data["pooled_prompt_embeds"].to(device=device, dtype=embed_dtype)

            # --- 1. SDE trajectory of the current policy ---
            pipeline.transformer.eval()
            with torch_autocast("cuda", enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    sample_kwargs = build_sample_kwargs(
                        prompt_embeds, pooled_prompt_embeds,
                        neg_prompt_embed.repeat(len(prompts), 1, 1),
                        neg_pooled_prompt_embed.repeat(len(prompts), 1),
                        len(prompts),
                    )
                    _, latents_list, _ = pipeline_with_logprob(pipeline, **sample_kwargs)
                    stacked_latents_tbc = torch.stack(latents_list)          # [nfe+1, B, C, H, W]
                    stacked_latents_btc = stacked_latents_tbc.transpose(0, 1)
                    x1 = stacked_latents_tbc[-1]
            saved_x1 = x1.detach().cpu().clone() if is_main_process(rank) else None

            sampler_timesteps = pipeline.scheduler.timesteps.to(device=device)
            sigmas = pipeline.scheduler.sigmas.to(device=device)
            num_available_train_steps = min(len(sampler_timesteps), stacked_latents_btc.shape[1] - 1)
            sampler_timesteps = sampler_timesteps[:num_available_train_steps]

            # --- 2. Terminal reward gradient ---
            if not args.keep_transformer_on_gpu:
                transformer_ddp.module.to("cpu")
            move_reward_scorers(reward_scorers, device)
            torch.cuda.empty_cache()
            if args.lambda_reward > 0.0:
                reward_grad = compute_reward_grad(
                    x1, prompts, pipeline.vae, reward_scorers, reward_fn=args.reward_fn,
                    bs_reward=args.bs_reward, lambda_r=args.lambda_reward, reward_ratios=args.reward_ratios,
                )
            else:
                reward_grad = 0.0
            move_reward_scorers(reward_scorers, "cpu")
            if not args.keep_transformer_on_gpu:
                transformer_ddp.module.to(device)
            torch.cuda.empty_cache()

            # --- 3. Backward adjoint simulation with the frozen base model ---
            adjoint_states, all_v_pred_init = compute_adjoints(
                transformer_ddp.module, stacked_latents_btc, sampler_timesteps,
                prompt_embeds, pooled_prompt_embeds, reward_grad, sigmas, eta=config.sample.noise_level,
            )

            # --- 4. Adjoint matching on num_train_timesteps states ---
            transformer_ddp.train()
            transformer_ddp.module.set_adapter("default")
            indices_t = sample_time_indices(sampler_timesteps, stacked_latents_tbc, args.num_train_timesteps)
            train_t = sampler_timesteps[indices_t]
            train_x_t = stacked_latents_tbc[indices_t]
            adjoint_states_eval = adjoint_states[:, indices_t]
            v_pred_init_eval = all_v_pred_init[:, indices_t]
            num_selected_timesteps = indices_t.shape[0]

            for j in range(num_selected_timesteps):
                control_times_sqrt_dt, _, std_dev_t = evaluate_controls(
                    transformer_ddp.module, prompt_embeds, pooled_prompt_embeds,
                    train_x_t[j], train_t[j], v_pred_init_eval[:, j], sigmas, indices_t[j],
                    eta=config.sample.noise_level,
                )
                loss_evals = torch.sum(
                    (control_times_sqrt_dt + adjoint_states_eval[:, j] * std_dev_t) ** 2,
                    dim=tuple(range(1, control_times_sqrt_dt.ndim)),
                )
                loss = loss_evals.mean()
                loss_scaled = loss / effective_grad_accum

                total_backward_count += 1
                is_last_accum = (
                    (total_backward_count % effective_grad_accum == 0)
                    or (i == num_batches_per_epoch - 1 and j == num_selected_timesteps - 1)
                )
                if mixed_precision_dtype == torch.float16:
                    if is_last_accum:
                        scaler.scale(loss_scaled).backward()
                        if args.sync_grads:
                            average_gradients(transformer_trainable_parameters, world_size)
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_trainable_parameters, config.train.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
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
                        optimizer.zero_grad()
                    else:
                        with transformer_ddp.no_sync():
                            loss_scaled.backward()

                if is_last_accum:
                    global_step += 1
                    logger.info(f"[rank{rank}] epoch={epoch} iter={i} step={global_step}  loss={loss.item():.6f}")
                    if global_step % args.ckpt_every == 0:
                        if is_main_process(rank):
                            save_lora(transformer_ddp, run_dir, f"iter_e{epoch:04d}_i{i:06d}_step{global_step:06d}", logger)
                        if world_size > 1:
                            dist.barrier()
                    save_tag = f"epoch{epoch:04d}iter{i:04d}"
                    run_eval(save_tag)
                    if is_main_process(rank) and saved_x1 is not None:
                        save_image_batch(decode_latents_to_images(saved_x1), os.path.join(samples_dir, save_tag), "train_img")
                    if world_size > 1:
                        dist.barrier()
                    if args.max_iter is not None and global_step >= args.max_iter:
                        training_done = True
                        break
            if training_done:
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
