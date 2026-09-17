"""
Create train/val/test split manifests for the HiRISE dataset.

Rules:
- Augmented images (suffixes -r90, -r180, -r270, -fv, -fh, -brt) go to TRAIN only.
- Original images are split 70/10/20 by observation ID so no observation's tiles
  appear in more than one split.
- Val and test therefore contain only original, non-augmented images.

Deterministic (SEED=42) so results are reproducible across machines/teammates
from just the raw dataset — no split files need to be shared separately.

Run from the repo root (after scripts/download_data.py --dataset hirise):
    python finetune/make_hirise_splits.py

Output:
    data/hirise/train.txt
    data/hirise/val.txt
    data/hirise/test.txt
"""

import random
from pathlib import Path
from collections import defaultdict

HIRISE_ROOT  = "data/hirise"
LABELS_FILE  = "data/hirise/labels-map-proj-v3.txt"
IMAGE_SUBDIR = "map-proj-v3"
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.10
TEST_RATIO   = 0.20
SEED         = 42

CLASS_NAMES = {
    0: "other",
    1: "crater",
    2: "dark dune",
    3: "slope streak",
    4: "bright dune",
    5: "impact ejecta",
    6: "swiss cheese",
    7: "spider",
}

AUG_SUFFIXES = {'r90', 'r180', 'r270', 'fv', 'fh', 'brt'}


def is_augmented(fname):
    stem = fname.replace('.jpg', '')
    return stem.split('-')[-1] in AUG_SUFFIXES


def main():
    if not Path(LABELS_FILE).exists():
        raise SystemExit(
            f"{LABELS_FILE} not found. Run scripts/download_data.py --dataset hirise first."
        )

    random.seed(SEED)
    root = Path(HIRISE_ROOT)

    aug_by_class  = defaultdict(list)   # augmented -> train only
    orig_by_class = defaultdict(lambda: defaultdict(list))  # original -> split by obs ID

    with open(LABELS_FILE) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            filename, cid = parts[0], int(parts[1])
            full = root / IMAGE_SUBDIR / filename
            if not full.exists():
                continue
            if is_augmented(filename):
                aug_by_class[cid].append(filename)
            else:
                obs_id = "_".join(filename.split("_")[:2])
                orig_by_class[cid][obs_id].append(filename)

    train_samples, val_samples, test_samples = [], [], []

    print(f"{'Class':<22} {'Obs':>5} {'Orig':>6} {'Aug->tr':>7} {'Train':>6} {'Val':>5} {'Test':>5}")
    print("-" * 60)

    for cid in sorted(CLASS_NAMES.keys()):
        # All augmented images -> training
        for fname in aug_by_class[cid]:
            train_samples.append(f"{IMAGE_SUBDIR}/{fname} {cid}")

        # Original images split by observation ID
        obs_dict = orig_by_class[cid]
        obs_ids  = list(obs_dict.keys())
        random.shuffle(obs_ids)

        n_obs   = len(obs_ids)
        n_val   = max(1, round(n_obs * VAL_RATIO)) if n_obs >= 3 else 0
        n_test  = max(1, round(n_obs * TEST_RATIO)) if n_obs >= 3 else 0
        n_train = n_obs - n_val - n_test

        train_obs = obs_ids[:n_train]
        val_obs   = obs_ids[n_train:n_train + n_val]
        test_obs  = obs_ids[n_train + n_val:]

        tr, va, te = 0, 0, 0
        for obs in train_obs:
            for fname in obs_dict[obs]:
                train_samples.append(f"{IMAGE_SUBDIR}/{fname} {cid}")
                tr += 1
        for obs in val_obs:
            for fname in obs_dict[obs]:
                val_samples.append(f"{IMAGE_SUBDIR}/{fname} {cid}")
                va += 1
        for obs in test_obs:
            for fname in obs_dict[obs]:
                test_samples.append(f"{IMAGE_SUBDIR}/{fname} {cid}")
                te += 1

        aug_count = len(aug_by_class[cid])
        orig_count = sum(len(v) for v in obs_dict.values())
        print(f"{CLASS_NAMES[cid]:<22} {n_obs:>5} {orig_count:>6} {aug_count:>7} {tr+aug_count:>6} {va:>5} {te:>5}")

    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)

    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        out = root / f"{split_name}.txt"
        with open(out, "w") as f:
            f.write("\n".join(samples) + "\n")

    total = len(train_samples) + len(val_samples) + len(test_samples)
    print("-" * 60)
    print(f"{'TOTAL':<22} {'':>5} {'':>6} {'':>7} {len(train_samples):>6} {len(val_samples):>5} {len(test_samples):>5}")
    print("\nAugmented images -> train only. Val/test contain original images only.")
    print("No observation ID appears in more than one split.")
    print(f"Saved: {root}/train.txt, val.txt, test.txt")


if __name__ == "__main__":
    main()
