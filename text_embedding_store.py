"""
Pre-compute and store SD3.5 text embeddings for the PickScore prompts.

Output layout:
    dataset/pickscore/embeddings/
    ├── train/
    │   ├── 000000.pt   # {"prompt": str, "prompt_embeds": [205,4096], "pooled_prompt_embeds": [2048]}
    │   └── ...
    ├── test/
    │   └── ...
    ├── train_index.txt   # line number = index, value = prompt
    ├── test_index.txt
    └── negative.pt       # empty-prompt embedding

Usage:
    python text_embedding_store.py \
        --model_id stabilityai/stable-diffusion-3.5-medium \
        --dataset_dir dataset/pickscore \
        --output_dir dataset/pickscore/embeddings \
        --batch_size 64 \
        --dtype bf16
"""

import argparse
import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt


def load_prompts(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def encode_batch(prompts, text_encoders, tokenizers, device, max_sequence_length=128):
    prompt_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        prompts,
        max_sequence_length,
        device=device,
    )
    return prompt_embeds.cpu(), pooled_prompt_embeds.cpu()


def save_split(split, prompts, text_encoders, tokenizers, device, output_dir, batch_size):
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    index_path = os.path.join(output_dir, f"{split}_index.txt")
    with open(index_path, "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    global_idx = 0
    for i in tqdm(range(0, len(prompts), batch_size), desc=split):
        batch = prompts[i : i + batch_size]
        prompt_embeds, pooled_prompt_embeds = encode_batch(
            batch, text_encoders, tokenizers, device
        )
        # prompt_embeds:        [B, 205, 4096]
        # pooled_prompt_embeds: [B, 2048]
        for j, prompt in enumerate(batch):
            out_path = os.path.join(split_dir, f"{global_idx:06d}.pt")
            # .clone().contiguous() breaks the shared storage; otherwise torch.save
            # serializes the whole batch tensor into every file.
            torch.save(
                {
                    "prompt": prompt,
                    "prompt_embeds": prompt_embeds[j].clone().contiguous(),                # [205, 4096]
                    "pooled_prompt_embeds": pooled_prompt_embeds[j].clone().contiguous(),  # [2048]
                },
                out_path,
            )
            global_idx += 1

    print(f"  [{split}] {global_idx} files saved -> {split_dir}")
    print(f"  [{split}] index saved -> {index_path}")


def main(args):
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Text encoders only (transformer and VAE are dropped) ---
    print(f"Loading SD3.5 text encoders from '{args.model_id}' ...")
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    text_encoders = [
        pipeline.text_encoder.to(device).eval(),
        pipeline.text_encoder_2.to(device).eval(),
        pipeline.text_encoder_3.to(device).eval(),
    ]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    del pipeline.transformer
    del pipeline.vae
    torch.cuda.empty_cache()

    # --- Empty (negative) prompt ---
    print("Encoding negative prompt ...")
    neg_embeds, neg_pooled = encode_batch([""], text_encoders, tokenizers, device)
    torch.save(
        {
            "prompt_embeds": neg_embeds[0].clone().contiguous(),
            "pooled_prompt_embeds": neg_pooled[0].clone().contiguous(),
        },
        os.path.join(args.output_dir, "negative.pt"),
    )

    # --- Train / Test splits ---
    for split in ["train", "test"]:
        prompt_file = os.path.join(args.dataset_dir, f"{split}.txt")
        if not os.path.exists(prompt_file):
            print(f"Skipping {split}: {prompt_file} not found")
            continue
        prompts = load_prompts(prompt_file)
        print(f"\n[{split}] {len(prompts)} prompts")
        save_split(split, prompts, text_encoders, tokenizers, device, args.output_dir, args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--dataset_dir", type=str, default="dataset/pickscore")
    parser.add_argument("--output_dir", type=str, default="dataset/pickscore/embeddings")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    args = parser.parse_args()
    main(args)
