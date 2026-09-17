"""
Download the MSL and HiRISE labeled image datasets from Zenodo.

Both are public releases from Wagstaff et al. (JPL):
  MSL:    "Mars surface image (Curiosity rover) labeled data set"
          DOI 10.5281/zenodo.1049137
  HiRISE: HiRISE orbital image landmark labeled data set
          DOI 10.5281/zenodo.2538136

Run from the repo root:
    python scripts/download_data.py --dataset all
    python scripts/download_data.py --dataset msl
    python scripts/download_data.py --dataset hirise
"""

import argparse
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORDS = {
    "msl":    {"record_id": "1049137", "dest": Path("data/msl")},
    "hirise": {"record_id": "2538136", "dest": Path("data/hirise")},
}


def download_file(url, out_path: Path):
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(out_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc=out_path.name
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                pbar.update(len(chunk))


def download_record(name, record_id, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n[{name}] Querying Zenodo record {record_id} ...")
    r = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=30)
    r.raise_for_status()
    files = r.json()["files"]

    for f in files:
        fname = f["key"]
        url = f["links"]["self"]
        out_path = dest / fname

        if out_path.exists():
            print(f"  {fname} already present, skipping download")
        else:
            print(f"  Downloading {fname} ({f['size'] / 1e6:.1f} MB) ...")
            download_file(url, out_path)

        if fname.lower().endswith(".zip"):
            marker = dest / f".{fname}.extracted"
            if marker.exists():
                print(f"  {fname} already extracted, skipping")
                continue
            print(f"  Extracting {fname} ...")
            with zipfile.ZipFile(out_path) as zf:
                zf.extractall(dest)
            marker.touch()

    print(f"[{name}] Done -> {dest}/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["msl", "hirise", "all"], default="all")
    args = parser.parse_args()

    targets = ZENODO_RECORDS.keys() if args.dataset == "all" else [args.dataset]
    for name in targets:
        cfg = ZENODO_RECORDS[name]
        download_record(name, cfg["record_id"], cfg["dest"])

    print("\nIf a dataset's ZIP didn't extract into the expected structure "
          "(e.g. an extra top-level folder), move files so you end up with:\n"
          "  data/msl/calibrated/, data/msl/*-calibrated-shuffled.txt\n"
          "  data/hirise/map-proj-v3/, data/hirise/labels-map-proj-v3.txt")


if __name__ == "__main__":
    main()
