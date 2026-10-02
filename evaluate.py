"""
Generate DrawBench images with SD3.5-Medium (+ optional LoRA) and score them with PickScore.

Protocol (used for all reported numbers): 512x512, deterministic DPM-Solver-2 with 10 steps,
guidance_scale 1.0 (no CFG), bf16, one image per prompt with seed = --seed + prompt_index.
Images are saved as <output_dir>/images/<prompt_index:05d>.png so that eval_metrics.py can
compute the remaining metrics (ImageReward, Aesthetic, CLIPScore, HPSv2.1) on them.

    torchrun --nproc_per_node=4 evaluate.py --adapter_path <run>/checkpoints/<iter>/lora_current_policy \
        --output_dir <out> --save_images
"""
import argparse
import contextlib
import json
import os

import numpy as np
import torch
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.pickscore_scorer import PickScoreScorer


class TextPromptDataset(Dataset):
    def __init__(self, dataset_path, split="test", limit=None):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        self.prompts = prompts[:limit] if limit is not None else prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "sample_id": idx}


def collate_fn(examples):
    return {
        "prompts": [example["prompt"] for example in examples],
        "sample_ids": [example["sample_id"] for example in examples],
    }


def resolve_adapter_dir(adapter_path):
    adapter_path = os.path.abspath(adapter_path)
    if os.path.isfile(adapter_path):
        adapter_path = os.path.dirname(adapter_path)
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        # also accept the iteration folder that contains lora_current_policy/
        sub = os.path.join(adapter_path, "lora_current_policy")
        if os.path.exists(os.path.join(sub, "adapter_config.json")):
            adapter_path = sub
    if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_path}")
    if not os.path.exists(os.path.join(adapter_path, "adapter_model.safetensors")):
        raise FileNotFoundError(f"Missing adapter_model.safetensors in {adapter_path}")
    return adapter_path


def tensor_to_pil(image):
    image = image.detach().float().cpu().clamp(0, 1)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image tensor, got shape {tuple(image.shape)}")
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = (image.numpy() * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def move_generation_modules(pipeline, device):
    for name in ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3"):
        module = getattr(pipeline, name, None)
        if module is not None:
            module.to(device)


def get_distributed_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def setup_distributed(rank, world_size, local_rank):
    if world_size <= 1:
        return
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}"),
    )


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def gather_results(local_results, world_size):
    if world_size <= 1:
        return local_results
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_results)
    return [item for rank_results in gathered for item in rank_results]


def parse_args():
    parser = argparse.ArgumentParser(description="DrawBench generation + PickScore for SD3.5-Medium (+ LoRA).")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="LoRA adapter directory (or its adapter_model.safetensors). Omit to evaluate the base model.")
    parser.add_argument("--dataset_path", type=str, default="dataset/drawbench", help="Directory containing <split>.txt.")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for results and images.")
    parser.add_argument("--base_model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size per GPU.")
    parser.add_argument("--score_batch_size", type=int, default=8, help="PickScore batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N prompts.")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="dpm2", choices=["dpm1", "dpm2"])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_images", action="store_true", help="Save images (needed by eval_metrics.py).")
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = get_distributed_info()
    is_main = rank == 0

    adapter_dir = resolve_adapter_dir(args.adapter_path) if args.adapter_path else None
    output_dir = os.path.abspath(args.output_dir)
    image_dir = os.path.join(output_dir, "images")

    if not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires CUDA.")
    setup_distributed(rank, world_size, local_rank)

    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        if args.save_images:
            os.makedirs(image_dir, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.mixed_precision, torch.float32)
    enable_amp = args.mixed_precision != "no"

    if is_main:
        print(f"Loading base model: {args.base_model} ({world_size} GPU(s))")
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.base_model)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(leave=False, desc="Timestep", dynamic_ncols=True, disable=not is_main, position=1)

    if adapter_dir is not None:
        if is_main:
            print(f"Loading LoRA adapter: {adapter_dir}")
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer, adapter_dir, adapter_name="default", is_trainable=False,
        )
        pipeline.transformer.set_adapter("default")
    pipeline.transformer.eval()

    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=dtype)
    pipeline.text_encoder_2.to(device, dtype=dtype)
    pipeline.text_encoder_3.to(device, dtype=dtype)
    pipeline.transformer.to(device, dtype=dtype)

    dataset = TextPromptDataset(args.dataset_path, split=args.split, limit=args.limit)
    shard = Subset(dataset, list(range(rank, len(dataset), world_size)))
    dataloader = DataLoader(shard, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    if is_main:
        print(f"Loaded {len(dataset)} prompts from {dataset.file_path}")

    pickscore = PickScoreScorer(device=str(device), dtype=torch.float32)
    pickscore.model.to("cpu")
    torch.cuda.empty_cache()

    autocast_ctx = torch.autocast("cuda", enabled=True, dtype=dtype) if enable_amp else contextlib.nullcontext()
    results = []
    for batch in tqdm(dataloader, desc="Generating", dynamic_ncols=True, disable=not is_main, position=0):
        prompts = batch["prompts"]
        sample_ids = batch["sample_ids"]
        generators = [torch.Generator(device=device).manual_seed(args.seed + int(sid)) for sid in sample_ids]

        move_generation_modules(pipeline, device)
        with torch.no_grad():
            with autocast_ctx:
                images_tensor, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt=prompts,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    output_type="pt",
                    height=args.resolution,
                    width=args.resolution,
                    generator=generators,
                    deterministic=True,
                    solver=args.solver,
                )
        pil_images = [tensor_to_pil(image) for image in images_tensor]

        # free the generator while scoring
        move_generation_modules(pipeline, "cpu")
        torch.cuda.empty_cache()
        pickscore.model.to(device)
        batch_scores = []
        for start in range(0, len(prompts), args.score_batch_size):
            end = start + args.score_batch_size
            with torch.no_grad():
                scores = pickscore(prompts[start:end], pil_images[start:end])
            batch_scores.extend(scores.detach().float().cpu().tolist())
        pickscore.model.to("cpu")
        torch.cuda.empty_cache()

        for sample_id, prompt, image, score in zip(sample_ids, prompts, pil_images, batch_scores):
            item = {"sample_id": int(sample_id), "prompt": prompt, "scores": {"pickscore": float(score)}}
            if args.save_images:
                image_path = os.path.join(image_dir, f"{int(sample_id):05d}.png")
                image.save(image_path)
                item["image_path"] = image_path
            results.append(item)
        del images_tensor, pil_images

    results = gather_results(results, world_size)
    if not is_main:
        cleanup_distributed()
        return

    results.sort(key=lambda item: item["sample_id"])
    all_scores = [float(item["scores"]["pickscore"]) for item in results]
    with open(os.path.join(output_dir, "pickscore_results.jsonl"), "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "adapter_path": adapter_dir,
        "dataset_path": os.path.abspath(args.dataset_path),
        "split": args.split,
        "num_samples": len(all_scores),
        "pickscore_mean": float(np.mean(all_scores)),          # PickScore / 26
        "pickscore_std": float(np.std(all_scores)),
        "pickscore_mean_x26": float(np.mean(all_scores)) * 26.0,  # usual ~20 scale
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "solver": args.solver,
        "resolution": args.resolution,
        "seed": args.seed,
        "world_size": world_size,
    }
    with open(os.path.join(output_dir, "pickscore_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"PickScore mean: {summary['pickscore_mean']:.6f} (x26 = {summary['pickscore_mean_x26']:.4f})")
    print(f"Saved results to {output_dir}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
