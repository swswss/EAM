import os
import torch
import ImageReward as RM


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = (
            RM.load(
                "ImageReward-v1.0",
                device=device,
                download_root=os.path.expanduser(os.path.join(os.environ.get("HF_HOME", "~/.cache/"), "ImageReward")),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts, images):
        _, rewards = self.model.inference_rank(prompts, images)
        rewards = torch.diagonal(torch.Tensor(rewards).to(self.device).reshape(len(prompts), len(prompts)), 0)
        return rewards.contiguous()
