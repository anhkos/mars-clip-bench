"""
Download trained model checkpoints (.pt files) from cloud storage links.

Setup (one-time):
    cp scripts/checkpoint_urls.example.json scripts/checkpoint_urls.json
    # then edit checkpoint_urls.json with real shareable links

Run from the repo root:
    python scripts/download_checkpoints.py
"""

import json
import re
from pathlib import Path

import requests
from tqdm import tqdm

CONFIG_PATH = Path("scripts/checkpoint_urls.json")
DEST_DIR = Path("checkpoints")

GDRIVE_ID_RE = re.compile(r"(?:id=|/d/)([a-zA-Z0-9_-]{20,})")


def resolve_gdrive_url(url: str) -> str:
    """Turn a Google Drive share/view link into a direct-download link."""
    m = GDRIVE_ID_RE.search(url)
    if "drive.google.com" in url and m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url


def download_file(url: str, out_path: Path):
    if "drive.google.com" in url:
        try:
            import gdown
            gdown.download(url=resolve_gdrive_url(url), output=str(out_path), quiet=False, fuzzy=True)
            return
        except ImportError:
            print("  (tip: `pip install gdown` handles large Drive files more reliably)")

    url = resolve_gdrive_url(url)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(out_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                pbar.update(len(chunk))


def main():
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"{CONFIG_PATH} not found.\n"
            f"Copy scripts/checkpoint_urls.example.json to {CONFIG_PATH} "
            f"and fill in real download links first."
        )

    urls = json.loads(CONFIG_PATH.read_text())
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in urls.items():
        if "REPLACE_WITH" in url:
            print(f"Skipping {filename}: placeholder URL not filled in")
            continue

        out_path = DEST_DIR / filename
        if out_path.exists():
            print(f"{filename} already present, skipping")
            continue

        print(f"Downloading {filename} ...")
        download_file(url, out_path)

    print(f"\nDone. Checkpoints in {DEST_DIR}/")


if __name__ == "__main__":
    main()
