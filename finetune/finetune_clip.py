"""
Fine-tune CLIP ViT-B/32 on MSL labeled data.

Each image is classified against all 24 class text embeddings using cross-entropy loss.
A class-balanced sampler handles the heavy imbalance (Drill hole = 40% of dataset).
Both the visual and text encoders are updated.

Run from the repo root (after scripts/download_data.py --dataset msl):
    python finetune/finetune_clip.py

Output:
    checkpoints/clip_finetuned.pt                  -- best checkpoint (by val accuracy)
    eval/runs/embedding_cache/msl_finetuned_images.npy  -- re-embedded images with fine-tuned model
"""

import clip
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
CLIP_MODEL_ID   = "ViT-B/32"
MSL_ROOT        = "data/msl"
CHECKPOINT_PATH = "checkpoints/clip_finetuned.pt"
EMBEDDINGS_OUT  = "eval/runs/embedding_cache/msl_finetuned_images.npy"
IMAGE_PATHS_OUT = "eval/runs/embedding_cache/msl_finetuned_image_paths.txt"

EPOCHS          = 10
BATCH_SIZE      = 64
LR              = 1e-6
WEIGHT_DECAY    = 0.2
WARMUP_STEPS    = 100
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0:  "APXS",
    1:  "APXS calibration target",
    2:  "Artifact",
    3:  "ChemCam calibration target",
    4:  "CheMin inlet open",
    5:  "Close-up rock",
    6:  "Distant landscape",
    7:  "Drill",
    8:  "Drill hole",
    9:  "DRT",
    10: "DRT spot",
    11: "Float",
    12: "Ground",
    13: "Horizon",
    14: "Inlet",
    15: "Layers",
    16: "Light-toned veins",
    17: "MAHLI",
    18: "MAHLI calibration target",
    19: "Mastcam",
    20: "Mastcam calibration target",
    21: "Nearby surface",
    23: "Observation tray",
    24: "Portion box",
}

# Maps class_id -> contiguous index 0..23 for cross-entropy targets
CLASS_IDX = {cid: i for i, cid in enumerate(CLASS_NAMES.keys())}


AUG_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
])


class MSLDataset(Dataset):
    def __init__(self, split, root_dir, preprocess, augment=False):
        self.preprocess = preprocess
        self.augment = augment
        self.samples = []
        root = Path(root_dir)
        manifest_map = {
            "train": "train-calibrated-shuffled.txt",
            "val":   "val-calibrated-shuffled.txt",
            "test":  "test-calibrated-shuffled.txt",
        }
        with open(root / manifest_map[split]) as f:
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
        if self.augment:
            img = AUG_TRANSFORM(img)
        return self.preprocess(img), CLASS_IDX[class_id]


def make_balanced_sampler(dataset):
    """Inverse-frequency weighting so every class gets equal expected samples."""
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

        # Clamp logit_scale to avoid instability
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


def compute_and_save_embeddings(model, preprocess, device):
    print("\nRecomputing embeddings with fine-tuned model...")
    root = Path(MSL_ROOT)
    paths = []
    for manifest_name in ("train-calibrated-shuffled.txt", "val-calibrated-shuffled.txt", "test-calibrated-shuffled.txt"):
        with open(root / manifest_name) as f:
            for line in f:
                rel = line.strip().split()[0]
                full = root / rel
                if full.exists():
                    paths.append(str(full))

    Path(EMBEDDINGS_OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(IMAGE_PATHS_OUT, "w") as f:
        f.write("\n".join(paths) + "\n")

    model.eval()
    embeddings = []
    with torch.no_grad():
        for path in tqdm(paths, desc="Encoding"):
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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path(CHECKPOINT_PATH).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading CLIP ({CLIP_MODEL_ID})...")
    model, preprocess = clip.load(CLIP_MODEL_ID, device=device)
    model = model.float()

    # Tokenize all 24 class prompts once
    prompts = [f"a photo of a {CLASS_NAMES[cid]}" for cid in CLASS_NAMES]
    class_tokens = clip.tokenize(prompts).to(device)

    train_ds = MSLDataset("train", MSL_ROOT, preprocess, augment=True)
    val_ds   = MSLDataset("val",   MSL_ROOT, preprocess, augment=False)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    sampler      = make_balanced_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * len(train_loader)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # Baseline before any training
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
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": vl_loss,
                "val_acc": vl_acc,
            }, CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved (best val_acc={vl_acc:.3f})")

    print(f"\nBest val_acc: {best_val_acc:.3f} ({best_val_acc*100:.1f}%)")

    # Load best and recompute embeddings
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    compute_and_save_embeddings(model, preprocess, device)

    print("\nDone. Checkpoint at:", CHECKPOINT_PATH)
    print("To benchmark it: python eval/model_agnostic_retrieval_eval.py "
          "--model clip --model-id ViT-B/32 --checkpoint checkpoints/clip_finetuned.pt")


if __name__ == "__main__":
    main()
