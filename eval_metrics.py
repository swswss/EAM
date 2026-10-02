"""
Score saved DrawBench images (from evaluate.py --save_images) with ImageReward, PickScore,
Aesthetic, CLIPScore and HPSv2.1. Images are matched to the prompts of <dataset_path>/<split>.txt
by sorted file name (00000.png, 00001.png, ...). PickScore is reported /26.

    torchrun --nproc_per_node=4 eval_metrics.py --image_dir <out>/images --output_dir <out>
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from flow_grpo.rewards import multi_score


DEFAULT_METRICS = ("imagereward", "pickscore", "aesthetic", "clipscore", "hpsv2")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_existing_path(path):
    path = Path(path)
    if path.exists():
        return path

    script_relative = SCRIPT_DIR / path
    if script_relative.exists():
        return script_relative

    return path


class DrawBenchImageDataset(Dataset):
    def __init__(self, image_dir, dataset_path, split="test", limit=None, recursive=False):
        self.image_dir = resolve_existing_path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        self.dataset_path = resolve_existing_path(dataset_path)
        prompt_path = self.dataset_path / f"{split}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

        pattern_iter = self.image_dir.rglob("*") if recursive else self.image_dir.iterdir()
        image_paths = sorted(
            (path for path in pattern_iter if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
            key=self._image_sort_key,
        )

        if limit is not None:
            prompts = prompts[:limit]
            image_paths = image_paths[:limit]

        if len(image_paths) != len(prompts):
            raise ValueError(
                f"Found {len(image_paths)} images but {len(prompts)} prompts. "
                "Images are matched to DrawBench prompts by sorted order. "
                "Use --limit if you intentionally evaluate a prefix."
            )

        self.prompts = prompts
        self.image_paths = image_paths

    @staticmethod
    def _image_sort_key(path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, str(path))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as image:
            image = image.convert("RGB")
            tensor = pil_to_tensor(image).float() / 255.0

        return {
            "sample_id": idx,
            "prompt": self.prompts[idx],
            "image": tensor,
            "image_path": str(self.image_paths[idx]),
        }


def collate_fn(examples):
    return {
        "sample_ids": [example["sample_id"] for example in examples],
        "prompts": [example["prompt"] for example in examples],
        "images": torch.stack([example["image"] for example in examples]),
        "image_paths": [example["image_path"] for example in examples],
        "metadata": [{} for _ in examples],
    }


def get_distributed_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def setup_distributed(rank, world_size, local_rank):
    if world_size <= 1:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed evaluation requires CUDA.")
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


def is_main_process(rank):
    return rank == 0


def print_rank0(rank, *args, **kwargs):
    if is_main_process(rank):
        print(*args, **kwargs)


def gather_results(local_results, world_size):
    if world_size <= 1:
        return local_results
    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_results)
    return [item for rank_results in gathered for item in rank_results]


def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    if isinstance(value, np.generic):
        return float(value.item())
    return float(value)


def summarize(results, metric_names):
    summary = {"num_samples": len(results), "metrics": {}}
    for metric_name in metric_names:
        values = [
            float(item["scores"][metric_name])
            for item in results
            if metric_name in item["scores"] and float(item["scores"][metric_name]) != -10.0
        ]
        summary["metrics"][metric_name] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "stderr": float(np.std(values) / np.sqrt(len(values))) if values else None,
            "min": float(np.min(values)) if values else None,
            "max": float(np.max(values)) if values else None,
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate existing DrawBench images with ImageReward, PickScore, Aesthetic, CLIPScore, and HPSv2."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing generated images.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset/drawbench",
        help="Directory containing DrawBench test.txt prompts.",
    )
    parser.add_argument("--split", type=str, default="test", help="Prompt split name.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for eval_5_metric_results.jsonl and eval_5_metric_summary.json. Defaults to image_dir.",
    )
    parser.add_argument("--batch_size", type=int, default=5, help="Metric evaluation batch size.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N sorted images/prompts.")
    parser.add_argument("--recursive", action="store_true", help="Find image files recursively under image_dir.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=list(DEFAULT_METRICS),
        help="Subset of the five DrawBench metrics to compute.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank = get_distributed_info()
    setup_distributed(rank, world_size, local_rank)

    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"Rank {rank}: LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA devices are visible."
            )
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    image_dir = resolve_existing_path(args.image_dir)
    dataset_path = resolve_existing_path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir
    if is_main_process(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    dataset = DrawBenchImageDataset(
        image_dir=image_dir,
        dataset_path=dataset_path,
        split=args.split,
        limit=args.limit,
        recursive=args.recursive,
    )
    shard_indices = list(range(rank, len(dataset), world_size))
    dataloader = DataLoader(
        Subset(dataset, shard_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    metric_weights = {metric_name: 1.0 for metric_name in args.metrics}
    print_rank0(rank, f"Evaluating {len(dataset)} DrawBench images with metrics: {', '.join(args.metrics)}")
    print_rank0(rank, f"Loading reward models on {device}...")
    scoring_fn = multi_score(device, metric_weights)

    local_results = []
    progress = tqdm(
        dataloader,
        desc=f"Rank {rank} eval",
        dynamic_ncols=True,
        disable=not is_main_process(rank),
    )
    for batch in progress:
        images = batch["images"].to(device=device, non_blocking=True)
        prompts = batch["prompts"]

        with torch.no_grad():
            all_scores, _ = scoring_fn(images, prompts, batch["metadata"], only_strict=False)

        for row_idx, sample_id in enumerate(batch["sample_ids"]):
            scores = {}
            for metric_name in args.metrics:
                scores[metric_name] = to_float(all_scores[metric_name][row_idx])

            local_results.append(
                {
                    "sample_id": int(sample_id),
                    "prompt": prompts[row_idx],
                    "scores": scores,
                    "image_path": batch["image_paths"][row_idx],
                }
            )

        del images, all_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = gather_results(local_results, world_size)
    if not is_main_process(rank):
        cleanup_distributed()
        return

    results.sort(key=lambda item: item["sample_id"])
    results_path = output_dir / "eval_5_metric_results.jsonl"
    with open(results_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = summarize(results, args.metrics)
    summary.update(
        {
            "image_dir": str(image_dir.resolve()),
            "dataset_path": str(dataset_path.resolve()),
            "split": args.split,
            "world_size": world_size,
        }
    )
    summary_path = output_dir / "eval_5_metric_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved per-sample results to {results_path}")
    print(f"Saved summary to {summary_path}")
    for metric_name in args.metrics:
        stats = summary["metrics"][metric_name]
        mean = stats["mean"]
        stderr = stats["stderr"]
        if mean is not None:
            print(f"{metric_name:<12}: {mean:.6f} ± {stderr:.6f} (n={stats['count']})")

    cleanup_distributed()


if __name__ == "__main__":
    main()
