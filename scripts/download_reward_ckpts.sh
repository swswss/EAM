#!/bin/bash
# Download the reward-model weights that are not fetched automatically from the HF hub.
# (PickScore, CLIP and ImageReward are downloaded on first use.)
#   reward_ckpts/HPS_v2.1_compressed.pt            HPSv2.1
#   reward_ckpts/open_clip_pytorch_model.bin       OpenCLIP ViT-H-14 backbone of HPSv2.1
#   reward_ckpts/sac+logos+ava1-l14-linearMSE.pth  LAION aesthetic predictor head
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p reward_ckpts

download() {
    local url="$1" out="reward_ckpts/$2"
    if [ -f "$out" ]; then
        echo "exists: $out"
    else
        echo "downloading: $out"
        wget -q --show-progress -O "$out.part" "$url" && mv "$out.part" "$out"
    fi
}

download "https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt" "HPS_v2.1_compressed.pt"
download "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin" "open_clip_pytorch_model.bin"
download "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth" "sac+logos+ava1-l14-linearMSE.pth"
