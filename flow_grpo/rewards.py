"""Non-differentiable reward functions used for evaluation (DrawBench 5-metric eval)."""
from PIL import Image
import numpy as np
import torch


def _to_bchw_float01(images):
    """Convert an image batch to B,C,H,W float in [0, 1].

    Supports torch/numpy in channel-first or channel-last layouts.
    """
    if isinstance(images, torch.Tensor):
        x = images.detach().cpu()
        if x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D tensor shape: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported tensor rank for images: {x.ndim}")

        x = x.float()
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        return x.clamp(0, 1)

    if isinstance(images, np.ndarray):
        x = images
        if x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.transpose(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D ndarray shape: {x.shape}")
        else:
            raise ValueError(f"Unsupported ndarray rank for images: {x.ndim}")

        x = torch.from_numpy(x).float()
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        return x.clamp(0, 1)

    raise ValueError(f"Unsupported input type for image conversion: {type(images)}")


def _to_bhwc_uint8(images):
    bchw = _to_bchw_float01(images)
    arr = (bchw * 255).round().to(torch.uint8).numpy()
    return arr.transpose(0, 2, 3, 1)


def aesthetic_score(device):
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = (_to_bchw_float01(images) * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        images = _to_bchw_float01(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bchw_float01(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def multi_score(device, score_dict):
    """Returns fn(images, prompts, metadata) -> ({name: scores, ..., "avg": weighted sum}, {})."""
    score_functions = {
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "aesthetic": aesthetic_score,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device)

    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn
