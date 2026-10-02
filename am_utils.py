"""Reward models and differentiable reward gradients used by train_am.py (Adjoint Matching)."""
import torch
import torch.nn.functional as F


# CLIP normalization constants for PickScore (ViT-H-14)
_CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
_CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]


def _freeze_module(module):
    for param in module.parameters():
        param.requires_grad_(False)
    module.eval()


MULTI_REWARD_FNS = {"pickscore_clipscore_hpsv2", "pickscore_clipscore_hpsv2_aesthetics"}


def build_reward_scorers(reward_fn, device="cpu", dtype=torch.float32):
    reward_fn = reward_fn.lower()
    scorers = {}

    if reward_fn in {"pickscore", *MULTI_REWARD_FNS}:
        from flow_grpo.pickscore_scorer import PickScoreScorer

        scorers["pickscore"] = PickScoreScorer(device=device, dtype=dtype)

    if reward_fn == "imagereward":
        from flow_grpo.imagereward_scorer import ImageRewardScorer

        scorers["imagereward"] = ImageRewardScorer(device=device, dtype=dtype)

    if reward_fn in MULTI_REWARD_FNS:
        from flow_grpo.clip_scorer import ClipScorer
        from flow_grpo.hpsv2_scorer import HPSv2Scorer

        scorers["clipscore"] = ClipScorer(device=device)
        scorers["hpsv2"] = HPSv2Scorer(dtype=dtype, device=device)

    if reward_fn == "pickscore_clipscore_hpsv2_aesthetics":
        from flow_grpo.aesthetic_scorer import AestheticScorer

        scorers["aesthetic"] = AestheticScorer(dtype=dtype, device=device)

    if not scorers:
        raise ValueError(f"Unsupported reward_fn: {reward_fn}")

    for scorer in scorers.values():
        _freeze_module(scorer)
    return scorers


def move_reward_scorers(scorers, device):
    for scorer in scorers.values():
        scorer.to(device)
        if hasattr(scorer, "device"):
            try:
                scorer.device = device
            except AttributeError:
                pass
        if hasattr(scorer, "model") and hasattr(scorer.model, "device"):
            try:
                scorer.model.device = device
            except AttributeError:
                pass


def _pickscore_scores(decoded, prompts, pickscore_scorer, normalize=False):
    device = decoded.device
    text_inputs = pickscore_scorer.processor(
        text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_embs = pickscore_scorer.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

    decoded_224 = F.interpolate(decoded, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(_CLIP_MEAN, device=device, dtype=decoded_224.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device, dtype=decoded_224.dtype).view(1, 3, 1, 1)
    decoded_norm = (decoded_224 - mean) / std

    image_embs = pickscore_scorer.model.get_image_features(pixel_values=decoded_norm)
    image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
    scores = pickscore_scorer.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
    if normalize:
        scores = scores / 26.0
    return scores


def _imagereward_scores(decoded, prompts, imagereward_scorer):
    device = decoded.device
    model = imagereward_scorer.model
    text_input = model.blip.tokenizer(
        prompts, padding="max_length", truncation=True, max_length=35, return_tensors="pt"
    ).to(device)

    image = F.interpolate(decoded, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(_CLIP_MEAN, device=device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=device, dtype=image.dtype).view(1, 3, 1, 1)
    image = (image - mean) / std

    rewards = model.score_gard(text_input.input_ids, text_input.attention_mask, image)
    return rewards.flatten()


def _clipscore_scores(decoded, prompts, clip_scorer):
    texts = clip_scorer.processor(
        text=prompts, padding="max_length", truncation=True, return_tensors="pt"
    ).to(decoded.device)
    pixels = clip_scorer._process(decoded).to(decoded.device)
    outputs = clip_scorer.model(pixel_values=pixels, **texts)
    return outputs.logits_per_image.diagonal() / 100.0


def _hpsv2_scores(decoded, prompts, hpsv2_scorer):
    image = hpsv2_scorer.preprocess_val(decoded.to(hpsv2_scorer.dtype).to(device=decoded.device, non_blocking=True))
    text = hpsv2_scorer.processor(prompts).to(device=decoded.device, non_blocking=True)
    outputs = hpsv2_scorer.model(image, text)
    image_features, text_features = outputs["image_features"], outputs["text_features"]
    return torch.diagonal(image_features @ text_features.T, 0).contiguous()


def _aesthetic_scores(decoded, aesthetic_scorer):
    image = F.interpolate(decoded, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor(_CLIP_MEAN, device=decoded.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_CLIP_STD, device=decoded.device, dtype=image.dtype).view(1, 3, 1, 1)
    image = (image - mean) / std
    embed = aesthetic_scorer.clip.get_image_features(pixel_values=image.to(aesthetic_scorer.device))
    embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
    return aesthetic_scorer.mlp.layers(embed).squeeze(1)


def _resolve_reward_ratios(reward_ratios, reward_fn="pickscore_clipscore_hpsv2"):
    reward_fn = reward_fn.lower()
    if reward_ratios is None:
        if reward_fn == "pickscore_clipscore_hpsv2_aesthetics":
            return 0.25, 0.25, 0.25, 0.25
        return 0.333, 0.333, 0.333
    expected_len = 4 if reward_fn == "pickscore_clipscore_hpsv2_aesthetics" else 3
    if len(reward_ratios) != expected_len:
        if expected_len == 4:
            raise ValueError("reward_ratios must contain four values: pickscore clipscore hpsv2 aesthetics")
        raise ValueError("reward_ratios must contain three values: pickscore clipscore hpsv2")
    return tuple(float(ratio) for ratio in reward_ratios)


def compute_reward_scores(decoded, prompts, reward_scorers, reward_fn, reward_ratios=None):
    """Reward of decoded images in [0, 1].

    'pickscore'  : raw PickScore (~20).
    'imagereward': ImageReward.
    multi        : r0 * PickScore/26 + r1 * CLIPScore/100 + r2 * HPSv2.1 (+ r3 * Aesthetic/10).
    """
    reward_fn = reward_fn.lower()
    if reward_fn == "pickscore":
        return _pickscore_scores(decoded, prompts, reward_scorers["pickscore"], normalize=False)
    if reward_fn == "imagereward":
        return _imagereward_scores(decoded, prompts, reward_scorers["imagereward"])
    if reward_fn in MULTI_REWARD_FNS:
        ratios = _resolve_reward_ratios(reward_ratios, reward_fn)
        combined = (
            ratios[0] * _pickscore_scores(decoded, prompts, reward_scorers["pickscore"], normalize=True)
            + ratios[1] * _clipscore_scores(decoded, prompts, reward_scorers["clipscore"])
            + ratios[2] * _hpsv2_scores(decoded, prompts, reward_scorers["hpsv2"])
        )
        if reward_fn == "pickscore_clipscore_hpsv2_aesthetics":
            combined = combined + ratios[3] * (_aesthetic_scores(decoded, reward_scorers["aesthetic"]) / 10.0)
        return combined
    raise ValueError(f"Unsupported reward_fn: {reward_fn}")


def compute_reward_grad(
    latents,
    prompts,
    vae,
    reward_scorers,
    reward_fn="pickscore",
    bs_reward=1,
    lambda_r=1.0,
    reward_ratios=None,
):
    """
    Compute lambda_r * ∇_X R(decoder(X)) w.r.t. latents [B, C, H, W].

    VAE and reward-model parameters are frozen; only the gradient w.r.t. latents
    is returned. Processes latents in chunks of bs_reward to avoid OOM.
    """
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

            # 1. VAE decode → [-1, 1]
            decoded = vae.decode((x / scaling_factor) + shift_factor, return_dict=False)[0]

            # 2. Map to [0, 1]
            decoded = (decoded * 0.5 + 0.5).clamp(0, 1)

            # 3. Score
            scores = lambda_r * compute_reward_scores(
                decoded, prompt_chunk, reward_scorers, reward_fn, reward_ratios
            )

            # 4. Gradient w.r.t. latent chunk
            grad_chunk = torch.autograd.grad(scores.sum(), x)[0]

        all_grads.append(grad_chunk.detach())

    return torch.cat(all_grads, dim=0)
