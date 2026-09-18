"""
Download the RemoteCLIP and GeoRSCLIP checkpoints -- public HuggingFace Hub
files, no auth needed -- so they can be benchmarked as OpenCLIP models loaded
from a raw state_dict rather than a pretrained tag:

    python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-B-32 \
        --pretrained none --checkpoint checkpoints/remoteclip_vit_b32.pt

    python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-B-32 \
        --pretrained none --checkpoint checkpoints/georsclip_vit_b32.pt

Run from the repo root:
    python scripts/download_remote_sensing_checkpoints.py
"""

import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DEST = Path("checkpoints")

# ViT-B-32 chosen for both so results are comparable to the base OpenCLIP
# ViT-B-32 baseline -- any difference in retrieval quality is then about
# training data (remote-sensing vs. general web images), not model size.
TARGETS = {
    "remoteclip_vit_b32.pt": ("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
    "georsclip_vit_b32.pt": ("Zilun/GeoRSCLIP", "ckpt/RS5M_ViT-B-32.pt"),
}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for out_name, (repo_id, filename) in TARGETS.items():
        out_path = DEST / out_name
        if out_path.exists():
            print(f"{out_name} already present, skipping")
            continue
        print(f"Downloading {repo_id}/{filename} ...")
        cached_path = hf_hub_download(repo_id, filename)
        shutil.copy(cached_path, out_path)
        print(f"  -> {out_path}")

    print(
        "\nDone. Note: if a download above 404s, double-check the exact filename "
        "on the model's HuggingFace Hub page -- it may have changed since this "
        "script was written (chendelong/RemoteCLIP, Zilun/GeoRSCLIP)."
    )


if __name__ == "__main__":
    main()
