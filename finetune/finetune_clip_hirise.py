"""
Fine-tune CLIP ViT-B/32 on HiRISE orbital image data.

Each image is classified against all 8 class text embeddings using cross-entropy loss.
A class-balanced sampler handles the heavy imbalance (other = 83% of dataset).

Run from the repo root (after scripts/download_data.py --dataset hirise
and finetune/make_hirise_splits.py):
    python finetune/finetune_clip_hirise.py

Output:
    checkpoints/hirise_clip_finetuned.pt                       -- best checkpoint (by val accuracy)
    eval/runs/embedding_cache/hirise_finetuned_images.npy      -- re-embedded images with fine-tuned model
    eval/runs/embedding_cache/hirise_finetuned_image_paths.txt -- ordered list of all image paths
"""

import torch
import clip
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CLIP_MODEL_ID   = "ViT-B/32"
HIRISE_ROOT     = "data/hirise"
CHECKPOINT_PATH = "checkpoints/hirise_clip_finetuned.pt"
EMBEDDINGS_OUT  = "eval/runs/embedding_cache/hirise_finetuned_images.npy"
IMAGE_PATHS_OUT = "eval/runs/embedding_cache/hirise_finetuned_image_paths.txt"

EPOCHS          = 10
BATCH_SIZE      = 64
LR              = 1e-6
WEIGHT_DECAY    = 0.2
WARMUP_STEPS    = 100
# ──────────────────────────────────────────────────────────────────────────────

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

CLASS_IDX = {cid: i for i, cid in enumerate(CLASS_NAMES.keys())}


class HiRISEDataset(Dataset):
    def __init__(self, split, root_dir, preprocess):
        self.preprocess = preprocess
        self.samples = []
        root = Path(root_dir)
        manifest = root / f"{split}.txt"
        with open(manifest) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                rel, class_id = parts[0], int(parts[1])
                full = root / rel
                if full.exists() and class_id in CLASS_NAMES:
                    self.samples.append((str(full), class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, class_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.preprocess(img), CLASS_IDX[class_id]


def make_balanced_sampler(dataset):
    counts = Counter(CLASS_IDX[cid] for _, cid in dataset.samples)
    weights = [1.0 / counts[CLASS_IDX[cid]] for _, cid in dataset.samples]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def train_epoch(model, loader, class_tokens, optimizer, scaler, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, label_idxs in tqdm(loader, desc="  train", leave=False):
        imgs, label_idxs = imgs.to(device), label_idxs.to(device)
        optimizer.zero_grad()

        with autocast():
            img_feats = model.encode_image(imgs).float()
            txt_feats = model.encode_text(class_tokens).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * img_feats @ txt_feats.T
            loss = nn.CrossEntropyLoss()(logits, label_idxs)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        with torch.no_grad():
            model.logit_scale.clamp_(0, np.log(100))

        total_loss += loss.item()
        correct += (logits.argmax(1) == label_idxs).sum().item()
        total += label_idxs.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def val_epoch(model, loader, class_tokens, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    txt_feats = model.encode_text(class_tokens).float()
    txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    for imgs, label_idxs in tqdm(loader, desc="  val  ", leave=False):
        imgs, label_idxs = imgs.to(device), label_idxs.to(device)
        with autocast():
            img_feats = model.encode_image(imgs).float()
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = model.logit_scale.exp() * img_feats @ txt_feats.T
            loss = nn.CrossEntropyLoss()(logits, label_idxs)

        total_loss += loss.item()
        correct += (logits.argmax(1) == label_idxs).sum().item()
        total += label_idxs.size(0)

    return total_loss / len(loader), correct / total


AUG_SUFFIXES = {'r90', 'r180', 'r270', 'fv', 'fh', 'brt'}


def is_augmented(path):
    stem = Path(path).stem
    return stem.split('-')[-1] in AUG_SUFFIXES


def compute_and_save_embeddings(model, preprocess, device):
    print("\nRecomputing embeddings with fine-tuned model (original images only)...")
    root = Path(HIRISE_ROOT)
    all_samples = []
    for split in ["train", "val", "test"]:
        with open(root / f"{split}.txt") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    full = str(root / parts[0])
                    if Path(full).exists() and not is_augmented(full):
                        all_samples.append(full)

    Path(EMBEDDINGS_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_PATHS_OUT, "w") as f:
        f.write("\n".join(all_samples) + "\n")

    model.eval()
    embeddings = []
    with torch.no_grad():
        for path in tqdm(all_samples, desc="Encoding"):
            try:
                img = Image.open(path).convert("RGB")
                t = preprocess(img).unsqueeze(0).to(device)
                emb = model.encode_image(t).float()
                emb /= emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy().flatten())
            except Exception as e:
                print(f"  skip {path}: {e}")
                embeddings.append(np.zeros(512))

    np.save(EMBEDDINGS_OUT, np.vstack(embeddings))
    print(f"Saved {len(embeddings)} embeddings -> {EMBEDDINGS_OUT}")
    print(f"Saved image paths -> {IMAGE_PATHS_OUT}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)

    if not (Path(HIRISE_ROOT) / "train.txt").exists():
        raise SystemExit(
            f"{HIRISE_ROOT}/train.txt not found. Run finetune/make_hirise_splits.py first."
        )

    print(f"Loading CLIP ({CLIP_MODEL_ID})...")
    model, preprocess = clip.load(CLIP_MODEL_ID, device=device)
    model = model.float()

    prompts = [f"a photo of a {CLASS_NAMES[cid]}" for cid in CLASS_NAMES]
    class_tokens = clip.tokenize(prompts).to(device)

    train_ds = HiRISEDataset("train", HIRISE_ROOT, preprocess)
    val_ds   = HiRISEDataset("val",   HIRISE_ROOT, preprocess)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    print("\nClass distribution in train set:")
    counts = Counter(cid for _, cid in train_ds.samples)
    for cid, n in sorted(counts.items()):
        print(f"  {CLASS_NAMES[cid]:<20} {n:>5}  ({n/len(train_ds)*100:.1f}%)")

    sampler      = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = GradScaler()

    print("\nBaseline (no fine-tuning):")
    val_loss, val_acc = val_epoch(model, val_loader, class_tokens, device)
    print(f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  ({val_acc*100:.1f}%)\n")

    best_val_acc = 0.0
    print(f"Training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, class_tokens,
                                      optimizer, scaler, scheduler, device)
        vl_loss, vl_acc = val_epoch(model, val_loader, class_tokens, device)

        print(f"Epoch {epoch:2d}/{EPOCHS}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.3f}  ({vl_acc*100:.1f}%)")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
            }, CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved (best val_acc={vl_acc:.3f})")

    print(f"\nBest val_acc: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    compute_and_save_embeddings(model, preprocess, device)
    print("\nDone. Checkpoint at:", CHECKPOINT_PATH)
    print("To benchmark it: python eval/model_agnostic_retrieval_eval.py "
          "--model clip --model-id ViT-B/32 --dataset hirise --checkpoint checkpoints/hirise_clip_finetuned.pt")


if __name__ == "__main__":
    main()
