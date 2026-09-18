"""
Text-to-image (and, for DINOv2, image-to-image) retrieval evaluation that
works across model families, so every baseline gets benchmarked with the
exact same metrics (see metrics.py: class-first macro-averaged P@k/R@k/MAP/
NDCG@10, plus a majority-class baseline for context).

Image embeddings are computed fresh per model (cached to disk afterward,
keyed by model spec) rather than reusing any precomputed .npy -- those are
only valid for the exact checkpoint that produced them.

Run from the repo root. Examples:

    # Zero-shot base CLIP
    python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32

    # Fine-tuned CLIP (matches reports/msl_results_report.md numbers)
    python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --checkpoint checkpoints/clip_finetuned.pt

    # OpenCLIP, LAION-pretrained
    python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-B-32 --pretrained laion2b_s34b_b79k

    # RemoteCLIP / GeoRSCLIP -- OpenCLIP architecture loaded from a raw checkpoint
    # rather than a pretrained tag. Run scripts/download_remote_sensing_checkpoints.py first.
    python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-B-32 --pretrained none --checkpoint checkpoints/remoteclip_vit_b32.pt
    python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-B-32 --pretrained none --checkpoint checkpoints/georsclip_vit_b32.pt

    # SigLIP / SigLIP 2 (any HF model id)
    python eval/model_agnostic_retrieval_eval.py --model siglip --model-id google/siglip2-base-patch16-224

    # DINOv2 -- no text encoder, image-to-image only (auto-selected retrieval mode)
    python eval/model_agnostic_retrieval_eval.py --model dinov2 --model-id facebook/dinov2-base --dataset hirise

    # HiRISE instead of MSL
    python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --dataset hirise --checkpoint checkpoints/hirise_clip_finetuned.pt
"""

import os
import csv
import sys
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from metrics import evaluate_retrieval, print_report  # noqa: E402

K_VALUES = [1, 5, 10, 20]


# -- Encoders -----------------------------------------------------------------
# Every encoder exposes encode_images(list[PIL.Image]) and encode_texts(list[str]),
# both returning L2-normalized float32 numpy arrays, so ranking code downstream
# never needs to know which model family produced the embeddings. SUPPORTS_TEXT
# is False for vision-only models (DINOv2), which restricts them to image2image mode.

class ClipEncoder:
    """OpenAI CLIP (the `clip` package), optionally with fine-tuned weights."""

    SUPPORTS_TEXT = True

    def __init__(self, model_id, checkpoint, device):
        import clip
        self._clip = clip
        self.device = device
        self.model, self.preprocess = clip.load(model_id, device=device)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.model.load_state_dict(state)
            if isinstance(ckpt, dict) and "epoch" in ckpt:
                print(f"  Loaded fine-tuned weights (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.3f})")
            else:
                print("  Loaded weights from", checkpoint)
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
    """
    OpenCLIP models (open_clip_torch) -- covers LAION-pretrained CLIP variants
    (--pretrained laion2b_s34b_b79k, etc.) as well as remote-sensing-adapted
    checkpoints like RemoteCLIP and GeoRSCLIP, which ship as a raw state_dict
    rather than an open_clip pretrained tag: pass --pretrained none plus
    --checkpoint <path> for those.
    """

    SUPPORTS_TEXT = True

    def __init__(self, model_id, pretrained, checkpoint, device):
        import open_clip
        self._open_clip = open_clip
        self.device = device
        pretrained_tag = None if pretrained in (None, "none", "None") else pretrained
        if pretrained_tag is None and not checkpoint:
            raise ValueError("--pretrained none requires --checkpoint <path> (e.g. RemoteCLIP/GeoRSCLIP)")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained_tag
        )
        self.tokenizer = open_clip.get_tokenizer(model_id)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.model.load_state_dict(state)
            print("  Loaded weights from", checkpoint)
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
    """SigLIP / SigLIP 2 via HuggingFace transformers (e.g. google/siglip2-base-patch16-224)."""

    SUPPORTS_TEXT = True

    def __init__(self, model_id, checkpoint, device):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.model.load_state_dict(state)
            print("  Loaded weights from", checkpoint)
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


class DinoV2Encoder:
    """
    DINOv2 (self-supervised ViT, no text tower) -- image-to-image only.
    Answers a different question than the others: is CLIP-style language-image
    training even necessary for the visual side of retrieval, or does a strong
    vision-only backbone do just as well for image-to-image search?
    """

    SUPPORTS_TEXT = False

    def __init__(self, model_id, checkpoint, device):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if checkpoint:
            ckpt = torch.load(checkpoint, map_location=device)
            state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
            self.model.load_state_dict(state)
            print("  Loaded weights from", checkpoint)
        self.model.to(device).eval()

    @torch.no_grad()
    def encode_images(self, images, batch_size=32):
        out = []
        for i in range(0, len(images), batch_size):
            inputs = self.processor(images=images[i:i + batch_size], return_tensors="pt").to(self.device)
            hidden = self.model(**inputs).last_hidden_state
            emb = hidden[:, 0, :]  # the [CLS]-equivalent pooled feature
            emb = emb / emb.norm(dim=-1, keepdim=True)
            out.append(emb.cpu().numpy())
        return np.vstack(out).astype(np.float32)

    def encode_texts(self, texts, batch_size=256):
        raise NotImplementedError("DINOv2 has no text encoder -- only image2image retrieval works")


ENCODER_CLASSES = {"clip": ClipEncoder, "open_clip": OpenClipEncoder, "siglip": SiglipEncoder, "dinov2": DinoV2Encoder}


def build_encoder(args, device):
    if args.model == "clip":
        return ClipEncoder(args.model_id, args.checkpoint, device)
    if args.model == "open_clip":
        return OpenClipEncoder(args.model_id, args.pretrained, args.checkpoint, device)
    if args.model == "siglip":
        return SiglipEncoder(args.model_id, args.checkpoint, device)
    if args.model == "dinov2":
        return DinoV2Encoder(args.model_id, args.checkpoint, device)
    raise ValueError(f"Unknown --model {args.model}")


# -- Dataset loading ------------------------------------------------------------

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


HIRISE_AUG_SUFFIXES = {"r90", "r180", "r270", "fv", "fh", "brt"}


def is_hirise_augmented(filename):
    """True for HiRISE's rotated/flipped/brightness-jittered copies (e.g. *-r90.jpg).

    labels-map-proj-v3.txt lists both the ~10,433 original images and their
    augmented copies together. finetune_clip_hirise.py already excludes these
    when building its corpus; this loader has to match that or retrieval gets
    evaluated against a corpus padded with near-duplicate rotated copies of
    the same image, which makes retrieval look artificially easier.
    """
    stem = Path(filename).stem
    return stem.split("-")[-1] in HIRISE_AUG_SUFFIXES


def load_hirise_dataset(root_dir="data/hirise"):
    labels_file = Path(root_dir) / "labels-map-proj-v3.txt"
    paths, labels = [], []
    with open(labels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            filename = parts[0]
            if is_hirise_augmented(filename):
                continue
            full_path = Path(root_dir) / "map-proj-v3" / filename
            if full_path.exists():
                paths.append(str(full_path))
                labels.append(int(parts[1]))
    return paths, np.array(labels)


DATASET_LOADERS = {"msl": load_msl_dataset, "hirise": load_hirise_dataset}


def load_queries(queries_csv, query_type=None):
    """
    Reads eval/queries/*.csv (class_id, class_name, query_text). If a
    query_type column exists (e.g. once "authentic" scientist-written queries
    get added alongside the current "template" ones) and --query-type was
    passed, filters to matching rows; otherwise returns everything as-is, so
    this keeps working unchanged on today's 3-column CSVs.
    """
    with open(queries_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if query_type and rows and "query_type" in rows[0]:
        rows = [r for r in rows if r["query_type"] == query_type]
    elif query_type:
        print(f"  Note: --query-type={query_type} given but {queries_csv} has no query_type column; using all rows")
    return rows


# -- Image embedding cache ------------------------------------------------------

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


# -- Main ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(ENCODER_CLASSES.keys()))
    parser.add_argument("--model-id", required=True,
                         help="e.g. ViT-B/32, ViT-B-32, google/siglip2-base-patch16-224, facebook/dinov2-base")
    parser.add_argument("--pretrained", default="openai",
                         help="open_clip only: pretrained tag (see open_clip.list_pretrained()), "
                              "or 'none' to load entirely from --checkpoint (RemoteCLIP/GeoRSCLIP)")
    parser.add_argument("--checkpoint", default=None, help="optional fine-tuned/raw .pt to load on top of or instead of the base model")
    parser.add_argument("--dataset", default="msl", choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--dataset-root", default=None, help="defaults to data/msl or data/hirise")
    parser.add_argument("--queries", default=None, help="defaults to eval/queries/msl_queries.csv or eval/queries/hirise_queries.csv")
    parser.add_argument("--query-type", default=None, help="filter queries.csv by query_type column, if present (e.g. 'authentic')")
    parser.add_argument("--retrieval-mode", choices=["text2image", "image2image"], default=None,
                         help="defaults to image2image for --model dinov2, text2image otherwise")
    parser.add_argument("--out", default=None, help="defaults to eval/runs/<model>_<model-id>_<dataset>_<mode>_results.csv")
    parser.add_argument("--cache-dir", default="eval/runs/embedding_cache")
    parser.add_argument("--no-cache", action="store_true", help="recompute image embeddings even if a cache hit exists")
    args = parser.parse_args()

    encoder_cls = ENCODER_CLASSES[args.model]
    mode = args.retrieval_mode or ("image2image" if not encoder_cls.SUPPORTS_TEXT else "text2image")
    if mode == "text2image" and not encoder_cls.SUPPORTS_TEXT:
        raise SystemExit(f"--model {args.model} has no text encoder; use --retrieval-mode image2image")

    if args.dataset_root is None:
        args.dataset_root = "data/msl" if args.dataset == "msl" else "data/hirise"
    if args.queries is None:
        args.queries = "eval/queries/msl_queries.csv" if args.dataset == "msl" else "eval/queries/hirise_queries.csv"
    if args.out is None:
        safe_id = args.model_id.replace("/", "-")
        args.out = f"eval/runs/{args.model}_{safe_id}_{args.dataset}_{mode}_results.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading dataset '{args.dataset}' from {args.dataset_root} ...")
    image_paths, labels = DATASET_LOADERS[args.dataset](args.dataset_root)
    print(f"  {len(image_paths)} images, {int(np.sum(labels >= 0))} labeled")
    if not image_paths:
        raise SystemExit(
            f"No images found under {args.dataset_root}. "
            f"Run scripts/download_data.py --dataset {args.dataset} first."
        )

    print(f"Loading model {args.model}:{args.model_id} ({mode}) ...")
    encoder = build_encoder(args, device)

    corpus_embeddings = get_image_embeddings(encoder, image_paths, args)

    if mode == "image2image":
        query_embeddings = corpus_embeddings
        query_labels = labels.tolist()
        query_texts = None
        query_ids = image_paths
    else:
        queries = load_queries(args.queries, args.query_type)
        print(f"  {len(queries)} queries to evaluate\n")
        query_texts = [row["query_text"] for row in queries]
        query_labels = [int(row["class_id"]) for row in queries]
        query_embeddings = encoder.encode_texts(query_texts)
        query_ids = None

    results = evaluate_retrieval(
        query_embeddings, query_labels, corpus_embeddings, labels.tolist(),
        k_values=K_VALUES,
        exclude_self=(mode == "image2image"),
        query_ids=query_ids,
        corpus_ids=image_paths if mode == "image2image" else None,
        query_texts=query_texts,
    )

    os.makedirs(str(Path(args.out).parent), exist_ok=True)
    per_query = results["per_query"]
    if per_query:
        fieldnames = list(per_query[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_query)

    print_report(results, f"{args.model}:{args.model_id} on {args.dataset} ({mode})")
    print(f"\nPer-query results saved to {args.out}")


if __name__ == "__main__":
    main()
