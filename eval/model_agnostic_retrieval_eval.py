"""
Text-to-image retrieval evaluation that works across model families, so
CLIP, OpenCLIP, and SigLIP checkpoints can be benchmarked with the exact
same metrics (P@k, R@k, MAP, NDCG@10) used in reports/msl_results_report.md
and reports/hirise_results_report.md.

Image embeddings are computed fresh per model (cached to disk afterward,
keyed by model spec) rather than reusing any precomputed .npy — those are
only valid for the exact checkpoint that produced them.

Run from the repo root. Examples:

    # Zero-shot base CLIP
    python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32

    # Fine-tuned CLIP (matches msl_results_report.md numbers)
    python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --checkpoint checkpoints/clip_finetuned.pt

    # OpenCLIP (any model/pretrained pair from open_clip.list_pretrained())
    python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-L-14 --pretrained openai

    # SigLIP (any HF model id)
    python eval/model_agnostic_retrieval_eval.py --model siglip --model-id google/siglip-so400m-patch14-384

    # HiRISE instead of MSL
    python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --dataset hirise --checkpoint checkpoints/hirise_clip_finetuned.pt
"""

import os
import csv
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

K_VALUES = [1, 5, 10, 20]


# ── Encoders ─────────────────────────────────────────────────────────────────
# Every encoder exposes encode_images(list[PIL.Image]) and encode_texts(list[str]),
# both returning L2-normalized float32 numpy arrays, so ranking code downstream
# never needs to know which model family produced the embeddings.

class ClipEncoder:
    """OpenAI CLIP (the `clip` package), optionally with fine-tuned weights."""

    def __init__(self, model_id, checkpoint, device):
        import clip
        self._clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(model_id, device=device)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            self.model.load_state_dict(ckpt["model_state"])
            print(f"  Loaded fine-tuned weights (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.3f})")
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, images, batch_size=64):
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(img) for img in images[i:i + batch_size]]).to(self.device)
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=256):
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self._clip.tokenize(texts[i:i + batch_size]).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype(np.float32)


class OpenClipEncoder:
    """OpenCLIP models (open_clip_torch) — covers most non-OpenAI CLIP variants."""

    def __init__(self, model_id, pretrained, checkpoint, device):
        import open_clip
        self._open_clip = open_clip
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_id)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)
            print("  Loaded fine-tuned weights from", checkpoint)
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_images(self, images, batch_size=64):
        out = []
        for i in range(0, len(images), batch_size):
            batch = torch.stack([self.preprocess(img) for img in images[i:i + batch_size]]).to(self.device)
            emb = self.model.encode_image(batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=256):
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[i:i + batch_size]).to(self.device)
            emb = self.model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype(np.float32)


class SiglipEncoder:
    """SigLIP models via HuggingFace transformers (e.g. google/siglip-so400m-patch14-384)."""

    def __init__(self, model_id, checkpoint, device):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt.get("model_state", ckpt)
            self.model.load_state_dict(state)
            print("  Loaded fine-tuned weights from", checkpoint)
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        out = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=images[i:i + batch_size], return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=256):
        out = []
        for i in range(0, len(texts), batch_size):
            # SigLIP was trained with fixed-length padded text inputs.
            inputs = self.processor(
                text=texts[i:i + batch_size], return_tensors="pt", padding="max_length"
            ).to(self.device)
            emb = self.model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype(np.float32)


def build_encoder(args, device):
    if args.model == "clip":
        return ClipEncoder(args.model_id, args.checkpoint, device)
    if args.model == "open_clip":
        return OpenClipEncoder(args.model_id, args.pretrained, args.checkpoint, device)
    if args.model == "siglip":
        return SiglipEncoder(args.model_id, args.checkpoint, device)
    raise ValueError(f"Unknown --model {args.model}")


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_msl_dataset(root_dir="data/msl"):
    manifests = [
        "train-calibrated-shuffled.txt",
        "val-calibrated-shuffled.txt",
        "test-calibrated-shuffled.txt",
    ]
    paths, labels = [], []
    for manifest_name in manifests:
        manifest_path = Path(root_dir) / manifest_name
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                full_path = Path(root_dir) / parts[0]
                if full_path.exists():
                    paths.append(str(full_path))
                    labels.append(int(parts[1]))
    return paths, np.array(labels)


def load_hirise_dataset(root_dir="data/hirise"):
    labels_file = Path(root_dir) / "labels-map-proj-v3.txt"
    paths, labels = [], []
    with open(labels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            full_path = Path(root_dir) / "map-proj-v3" / parts[0]
            if full_path.exists():
                paths.append(str(full_path))
                labels.append(int(parts[1]))
    return paths, np.array(labels)


DATASET_LOADERS = {"msl": load_msl_dataset, "hirise": load_hirise_dataset}


# ── Metrics ──────────────────────────────────────────────────────────────────

def precision_at_k(sorted_labels, true_label, k):
    return sum(l == true_label for l in sorted_labels[:k]) / k


def recall_at_k(sorted_labels, true_label, k, total_relevant):
    if total_relevant == 0:
        return 0.0
    return sum(l == true_label for l in sorted_labels[:k]) / total_relevant


def average_precision(sorted_labels, true_label, total_relevant):
    if total_relevant == 0:
        return 0.0
    ap, hits = 0.0, 0
    for i, label in enumerate(sorted_labels):
        if label == true_label:
            hits += 1
            ap += hits / (i + 1)
    return ap / total_relevant


def ndcg_at_k(sorted_labels, true_label, k):
    dcg = sum(
        (1.0 / np.log2(i + 2))
        for i, label in enumerate(sorted_labels[:k])
        if label == true_label
    )
    total_relevant = min(sum(l == true_label for l in sorted_labels), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(total_relevant))
    return dcg / idcg if idcg > 0 else 0.0


# ── Image embedding cache ────────────────────────────────────────────────────

def cache_key(args, num_images):
    raw = f"{args.model}|{args.model_id}|{args.pretrained}|{args.checkpoint}|{args.dataset}|{num_images}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def get_image_embeddings(encoder, image_paths, args):
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_key(args, len(image_paths))}.npy"

    if cache_file.exists() and not args.no_cache:
        print(f"Loading cached image embeddings from {cache_file}")
        return np.load(cache_file)

    print(f"Encoding {len(image_paths)} images with {args.model}:{args.model_id} ...")
    images = []
    valid_idx = []
    for i, p in enumerate(tqdm(image_paths, desc="Loading images")):
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_idx.append(i)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    embeddings = encoder.encode_images(images)
    if len(valid_idx) != len(image_paths):
        full = np.zeros((len(image_paths), embeddings.shape[1]), dtype=np.float32)
        full[valid_idx] = embeddings
        embeddings = full

    if not args.no_cache:
        np.save(cache_file, embeddings)
        print(f"Cached image embeddings to {cache_file}")
    return embeddings


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["clip", "open_clip", "siglip"])
    parser.add_argument("--model-id", required=True, help="e.g. ViT-B/32, ViT-L-14, google/siglip-so400m-patch14-384")
    parser.add_argument("--pretrained", default="openai", help="open_clip only: pretrained tag (see open_clip.list_pretrained())")
    parser.add_argument("--checkpoint", default=None, help="optional fine-tuned .pt to load on top of the base model")
    parser.add_argument("--dataset", default="msl", choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--dataset-root", default=None, help="defaults to data/msl or data/hirise")
    parser.add_argument("--queries", default=None, help="defaults to eval/queries/msl_queries.csv or eval/queries/hirise_queries.csv")
    parser.add_argument("--out", default=None, help="defaults to eval/runs/<model>_<model-id>_<dataset>_results.csv")
    parser.add_argument("--cache-dir", default="eval/runs/embedding_cache")
    parser.add_argument("--no-cache", action="store_true", help="recompute image embeddings even if a cache hit exists")
    args = parser.parse_args()

    if args.dataset_root is None:
        args.dataset_root = "data/msl" if args.dataset == "msl" else "data/hirise"
    if args.queries is None:
        args.queries = "eval/queries/msl_queries.csv" if args.dataset == "msl" else "eval/queries/hirise_queries.csv"
    if args.out is None:
        safe_id = args.model_id.replace("/", "-")
        args.out = f"eval/runs/{args.model}_{safe_id}_{args.dataset}_results.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset '{args.dataset}' from {args.dataset_root} ...")
    image_paths, labels = DATASET_LOADERS[args.dataset](args.dataset_root)
    print(f"  {len(image_paths)} images, {np.sum(labels >= 0)} labeled")
    if not image_paths:
        raise SystemExit(
            f"No images found under {args.dataset_root}. "
            f"Run scripts/download_data.py --dataset {args.dataset} first."
        )

    print(f"Loading model {args.model}:{args.model_id} ...")
    encoder = build_encoder(args, device)

    embeddings = get_image_embeddings(encoder, image_paths, args)

    with open(args.queries, newline="", encoding="utf-8") as f:
        queries = list(csv.DictReader(f))
    print(f"  {len(queries)} queries to evaluate\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    results = []
    query_texts = [row["query_text"] for row in queries]
    text_embeddings = encoder.encode_texts(query_texts)

    for row, text_emb in zip(queries, text_embeddings):
        class_id = int(row["class_id"])
        class_name = row["class_name"]

        sims = embeddings.dot(text_emb)
        ranked_idxs = np.argsort(sims)[::-1]
        sorted_labels = labels[ranked_idxs].tolist()
        total_relevant = int(np.sum(labels == class_id))

        metrics = {
            "class_id": class_id,
            "class_name": class_name,
            "query_text": row["query_text"],
            "num_relevant": total_relevant,
        }
        for k in K_VALUES:
            metrics[f"p@{k}"] = round(precision_at_k(sorted_labels, class_id, k), 4)
            metrics[f"r@{k}"] = round(recall_at_k(sorted_labels, class_id, k, total_relevant), 4)
        metrics["map"] = round(average_precision(sorted_labels, class_id, total_relevant), 4)
        metrics["ndcg@10"] = round(ndcg_at_k(sorted_labels, class_id, 10), 4)
        results.append(metrics)

        print(
            f"[{class_id:>2}] {class_name:<30}  "
            f"P@1={metrics['p@1']:.3f}  P@5={metrics['p@5']:.3f}  "
            f"P@10={metrics['p@10']:.3f}  MAP={metrics['map']:.3f}  "
            f"NDCG@10={metrics['ndcg@10']:.3f}  (n={total_relevant})"
        )

    fieldnames = list(results[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'-' * 60}")
    print(f"SUMMARY — {args.model}:{args.model_id} on {args.dataset} (macro-averaged)")
    print(f"{'-' * 60}")
    for k in K_VALUES:
        avg_p = np.mean([r[f"p@{k}"] for r in results])
        avg_r = np.mean([r[f"r@{k}"] for r in results])
        print(f"  P@{k:<3} = {avg_p:.4f}    R@{k:<3} = {avg_r:.4f}")
    print(f"  MAP    = {np.mean([r['map'] for r in results]):.4f}")
    print(f"  NDCG@10= {np.mean([r['ndcg@10'] for r in results]):.4f}")
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
