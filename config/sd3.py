"""
Shared config for EAM (train_eam.py) and AM (train_am.py) on SD3.5-Medium.
Command-line flags of the training scripts override the fields marked (CLI).
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ── General ────────────────────────────────────────────────────────────────
    config.seed            = 42        # (CLI: --seed)
    config.num_epochs      = 2         # (CLI: --num_epochs)
    config.mixed_precision = "bf16"    # "no" | "fp16" | "bf16"
    config.allow_tf32      = True
    config.use_lora        = True
    config.resolution      = 512

    # ── Pretrained model ───────────────────────────────────────────────────────
    config.pretrained = ml_collections.ConfigDict()
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"

    # ── Sampling ───────────────────────────────────────────────────────────────
    config.sample = ml_collections.ConfigDict()
    config.sample.num_steps        = 10      # (CLI: --training_nfe)
    config.sample.eval_num_steps   = 10      # (CLI: --eval_nfe, EAM only)
    config.sample.guidance_scale   = 1.0     # no CFG
    config.sample.train_batch_size = 8       # per GPU (CLI: --batch_size)
    config.sample.noise_level      = 0.7
    config.sample.deterministic    = True
    config.sample.solver           = "dpm2"  # AM switches to the memoryless SDE ("flow")

    # ── Optimizer ──────────────────────────────────────────────────────────────
    config.train = ml_collections.ConfigDict()
    config.train.learning_rate     = 1e-4    # (CLI: --lr)
    config.train.adam_beta1        = 0.9
    config.train.adam_beta2        = 0.999
    config.train.adam_weight_decay = 1e-4
    config.train.adam_epsilon      = 1e-8
    config.train.max_grad_norm     = 1.0
    config.train.lora_path         = None    # optional LoRA to initialize from

    return config
