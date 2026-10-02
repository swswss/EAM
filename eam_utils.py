"""Differentiable reward gradients ∇_x R(VAE.decode(x)) used by train_eam.py."""
import torch
import torch.nn.functional as F

# CLIP normalization constants for PickScore (ViT-H-14)
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def _resize_norm_clip(decoded_01, device):
    """Resize [B,3,H,W] in [0,1] to 224 and CLIP-normalize."""
    decoded = F.interpolate(decoded_01, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(_CLIP_MEAN, device=device, dtype=decoded.dtype).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD,  device=device, dtype=decoded.dtype).view(1, 3, 1, 1)
    return (decoded - mean) / std


def _pickscore_diff(decoded_01, prompts, scorer, device):
    """Differentiable PickScore (returns logit_scale * cos_sim / 26 in [B])."""
    text_inputs = scorer.processor(
        text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embs = scorer.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

    decoded_norm = _resize_norm_clip(decoded_01, device)
    image_embs = scorer.model.get_image_features(pixel_values=decoded_norm)
    image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

    logit_scale = scorer.model.logit_scale.exp()
    scores = logit_scale * (text_embs * image_embs).sum(dim=-1)
    return scores / 26.0


def _clipscore_diff(decoded_01, prompts, scorer, device):
    """Differentiable ClipScore (logits_per_image.diagonal / 100 in [B])."""
    text_inputs = scorer.processor(
        text=prompts, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    decoded = scorer._process(decoded_01).to(device)
    outputs = scorer.model(pixel_values=decoded, **text_inputs)
    return outputs.logits_per_image.diagonal() / 100.0


def _hpsv2_diff(decoded_01, prompts, scorer, device):
    """Differentiable HPSv2.1 (image_features @ text_features.T diagonal in [B])."""
    image = scorer.preprocess_val(decoded_01.to(scorer.dtype).to(device))
    text = scorer.processor(prompts).to(device)
    outputs = scorer.model(image, text)
    image_features = outputs["image_features"]
    text_features = outputs["text_features"]
    logits_per_image = image_features @ text_features.T
    return torch.diagonal(logits_per_image, 0).contiguous()


def _aesthetic_diff(decoded_01, prompts, scorer, device):
    """Differentiable LAION aesthetic predictor (returns scalar score in [B])."""
    decoded_norm = _resize_norm_clip(decoded_01, device)
    image_embs = scorer.clip.get_image_features(pixel_values=decoded_norm)
    image_embs = image_embs / torch.linalg.vector_norm(image_embs, dim=-1, keepdim=True)
    # bypass @torch.no_grad on MLP.forward by calling layers directly
    score = scorer.mlp.layers(image_embs)
    return score.squeeze(1)


_SCORER_FNS = {
    "pickscore": _pickscore_diff,
    "clipscore": _clipscore_diff,
    "hpsv2": _hpsv2_diff,
    "aesthetic": _aesthetic_diff,
}


def compute_reward_grad(latents, prompts, vae, scorers, bs_reward=1, lambda_r=1.0):
    """
    Compute lambda_r * ∇_X R(decoder(X)) w.r.t. latents, where R is the sum of the
    differentiable scorers in `scorers` ({name: scorer}).

    Args:
        latents:   Tensor [B, C, H, W] — final denoised latents.
        prompts:   List[str] of length B.
        vae:       Frozen SD3.5 VAE (already on the correct device).
        scorers:   dict {name: scorer_obj}, name in {'pickscore','clipscore','hpsv2','aesthetic'}.
        bs_reward: Chunk size for the decode + reward backward (reduce if OOM).
        lambda_r:  Scalar multiplier on the reward.

    Returns:
        grad: Tensor [B, C, H, W] (fp32), detached.
    """
    device = latents.device
    B = latents.shape[0]
    scaling_factor = vae.config.scaling_factor
    shift_factor   = getattr(vae.config, "shift_factor", 0.0)

    all_grads = []
    for start in range(0, B, bs_reward):
        end = min(start + bs_reward, B)
        lat_chunk    = latents[start:end]
        prompt_chunk = prompts[start:end]

        with torch.enable_grad():
            x = lat_chunk.detach().float().requires_grad_(True)

            # VAE decode → [-1, 1] → [0, 1]
            decoded = vae.decode((x / scaling_factor) + shift_factor, return_dict=False)[0]
            decoded_01 = (decoded * 0.5 + 0.5).clamp(0, 1)

            total_score = None
            for name, scorer in scorers.items():
                s = _SCORER_FNS[name](decoded_01, prompt_chunk, scorer, device)
                total_score = s if total_score is None else total_score + s

            total_score = lambda_r * total_score
            grad_chunk = torch.autograd.grad(total_score.sum(), x)[0]

        all_grads.append(grad_chunk.detach())

    return torch.cat(all_grads, dim=0)


def move_scorer(scorer, device):
    """Move a reward scorer (and the device attribute its __call__ reads) to `device`."""
    if hasattr(scorer, "model"):
        scorer.model.to(device)
        if hasattr(scorer.model, "device"):
            try:
                scorer.model.device = device
            except Exception:
                pass
    else:
        scorer.to(device)
    if hasattr(scorer, "device"):
        scorer.device = device
