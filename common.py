"""Utilities shared by train_eam.py and train_am.py (SD3.5-Medium + LoRA)."""
import os
import random
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, PeftModel, get_peft_model


# ── Distributed ────────────────────────────────────────────────────────────────
def setup_distributed(rank, local_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size,
        timeout=timedelta(hours=2),
    )


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def average_gradients(parameters, world_size):
    """All-reduce (mean) the .grad of `parameters` across ranks.

    The training loops offload the transformer to CPU during the reward pass. Moving a parameter
    to another device resets its autograd grad-accumulator, which disconnects DDP's all-reduce
    hooks, so DDP alone does not average the LoRA gradients. This explicit all-reduce does.
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    if world_size <= 1 or not grads:
        return
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat.div_(world_size)
    for g, synced in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
        g.copy_(synced)


def load_config(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()


# ── Pre-computed text embeddings (see text_embedding_store.py) ─────────────────
class EmbeddingDataset(Dataset):
    """Lazy-loading dataset for pre-computed SD3.5 text embeddings."""

    def __init__(self, split, embedding_dir):
        index_path = os.path.join(embedding_dir, f"{split}_index.txt")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"{index_path} not found. Pre-compute the text embeddings first "
                f"(python text_embedding_store.py --output_dir {embedding_dir})."
            )
        with open(index_path) as f:
            self.prompts = [line.strip() for line in f if line.strip()]
        self.split_dir = os.path.join(embedding_dir, split)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        data = torch.load(
            os.path.join(self.split_dir, f"{idx:06d}.pt"),
            map_location="cpu",
            weights_only=True,
        )
        return {
            "prompt": self.prompts[idx],
            "prompt_embeds": data["prompt_embeds"],                # [205, 4096]
            "pooled_prompt_embeds": data["pooled_prompt_embeds"],  # [2048]
        }


def collate_embedding_batch(batch):
    return {
        "prompts": [x["prompt"] for x in batch],
        "prompt_embeds": torch.stack([x["prompt_embeds"] for x in batch]),
        "pooled_prompt_embeds": torch.stack([x["pooled_prompt_embeds"] for x in batch]),
    }


def load_negative_embeddings(embedding_dir, device, dtype):
    neg_data = torch.load(
        os.path.join(embedding_dir, "negative.pt"),
        map_location=device,
        weights_only=True,
    )
    neg_prompt_embed = neg_data["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)                # [1, 205, 4096]
    neg_pooled_prompt_embed = neg_data["pooled_prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype)  # [1, 2048]
    return neg_prompt_embed, neg_pooled_prompt_embed


# ── SD3.5 pipeline + LoRA ──────────────────────────────────────────────────────
LORA_TARGET_MODULES = [
    "attn.add_k_proj",
    "attn.add_q_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    "attn.to_k",
    "attn.to_out.0",
    "attn.to_q",
    "attn.to_v",
]


def get_lora_config():
    return LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=LORA_TARGET_MODULES,
    )


def load_sd3_pipeline(config, device, rank):
    """SD3.5 pipeline without text encoders (training uses pre-computed embeddings)."""
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,
        text_encoder=None, text_encoder_2=None, text_encoder_3=None,
        tokenizer=None, tokenizer_2=None, tokenizer_3=None,
    )
    pipeline.vae.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    pipeline.vae.to(device, dtype=torch.float32)
    # Only the VAE decoder is needed.
    pipeline.vae.encoder = None
    torch.cuda.empty_cache()
    return pipeline


def add_lora_and_wrap_ddp(pipeline, config, device, local_rank):
    """Attach a trainable LoRA ("default" adapter) to the transformer and wrap it in DDP.

    The frozen base model is recovered with `transformer_ddp.module.disable_adapter()`.
    """
    if not config.use_lora:
        raise NotImplementedError("Training expects LoRA adapters (config.use_lora=True).")

    transformer = pipeline.transformer.to(device)
    lora_config = get_lora_config()
    if config.train.lora_path:
        transformer = PeftModel.from_pretrained(
            transformer, config.train.lora_path, adapter_name="default", is_trainable=True,
        )
    else:
        transformer = get_peft_model(transformer, lora_config)
    transformer.set_adapter("default")

    transformer_ddp = DDP(
        transformer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    transformer_ddp.module.set_adapter("default")
    trainable_parameters = [p for p in transformer_ddp.module.parameters() if p.requires_grad]
    return transformer_ddp, trainable_parameters


def save_lora(transformer_ddp, run_dir, tag, logger):
    """Save only the LoRA adapter to <run_dir>/checkpoints/<tag>/lora_current_policy (call on rank 0)."""
    save_root = os.path.join(run_dir, "checkpoints", tag)
    os.makedirs(save_root, exist_ok=True)
    model = transformer_ddp.module
    model.set_adapter("default")
    model.save_pretrained(os.path.join(save_root, "lora_current_policy"))
    logger.info(f"Saved LoRA checkpoint to {save_root}")
