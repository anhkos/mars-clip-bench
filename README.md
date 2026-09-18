# mars-clip-bench

Benchmarking CLIP, OpenCLIP, and SigLIP (zero-shot and fine-tuned) for
text-to-image retrieval on Mars imagery, split out from the main
[JPS-PDS-LLM](../JPS-PDS-LLM) chatbot repo so the experiment/eval code isn't
tangled up with the production backend, and so datasets/checkpoints never
end up committed to git history.

## Quickstart for teammates

1. **Clone the repo**
   ```bash
   git clone https://github.com/anhkos/mars-clip-bench.git
   cd mars-clip-bench
   ```

2. **Set up a Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   No local GPU? See **[Running in Google Colab](#running-in-google-colab)**
   below for the full notebook-cell walkthrough instead of steps 1-2 here.

3. **Download the datasets** (public Zenodo releases, no auth needed)
   ```bash
   python scripts/download_data.py --dataset all
   ```

4. **Generate the HiRISE train/val/test splits** — deterministic (seed=42), so this
   always reproduces the same split from the raw data, no file-sharing needed
   ```bash
   python finetune/make_hirise_splits.py
   ```

5. **Get the trained checkpoints**
   ```bash
   cp scripts/checkpoint_urls.example.json scripts/checkpoint_urls.json
   # paste the real shareable links into checkpoint_urls.json, then:
   python scripts/download_checkpoints.py
   ```

6. **Reproduce the existing baseline numbers** and confirm your setup matches
   ```bash
   python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --checkpoint checkpoints/clip_finetuned.pt
   python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --dataset hirise --checkpoint checkpoints/hirise_clip_finetuned.pt
   ```
   Compare the printed `SUMMARY` block against `reports/msl_results_report.md` and
   `reports/hirise_results_report.md` — same data + same checkpoint + same seed
   should reproduce the same numbers.

7. **Benchmark a new model** (this is the actual point of this repo)
   ```bash
   python eval/model_agnostic_retrieval_eval.py --model siglip --model-id google/siglip-so400m-patch14-384
   python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-L-14 --pretrained openai
   ```
   Each run writes to `eval/runs/<model>_<model-id>_<dataset>_results.csv` — per-query
   and macro-averaged metrics, directly comparable across models since they all share
   the same query sets and metric functions.

8. **(Optional) Fine-tune a model from scratch** instead of downloading a checkpoint
   ```bash
   python finetune/finetune_clip.py
   python finetune/finetune_clip_hirise.py
   ```

See the sections below for flag-level detail on each step.

## Datasets

Both are public Zenodo releases from Wagstaff et al. (JPL)

| Dataset | Citation | DOI |
|---|---|---|
| MSL (Curiosity rover surface images) | Wagstaff, Lu, Stanboli et al. | [10.5281/zenodo.1049137](https://doi.org/10.5281/zenodo.1049137) |
| HiRISE (orbital landmark images) | Wagstaff et al. | [10.5281/zenodo.2538136](https://doi.org/10.5281/zenodo.2538136) |

## Setup

```bash
pip install -r requirements.txt          # local, with GPU
# or, in Colab:
pip install -r requirements-colab.txt    # skips torch/torchvision — Colab has its own

python scripts/download_data.py --dataset all
python finetune/make_hirise_splits.py    # regenerates data/hirise/{train,val,test}.txt (deterministic, seed=42)
```

If a downloaded ZIP doesn't extract into the expected layout, move files so you end up with:

```
data/msl/calibrated/, data/msl/*-calibrated-shuffled.txt, data/msl/msl_synset_words-indexed.txt
data/hirise/map-proj-v3/, data/hirise/labels-map-proj-v3.txt
```

### Checkpoints

Trained `.pt` checkpoints live outside git (cloud storage), since they're
too large/binary for a clean history:

```bash
cp scripts/checkpoint_urls.example.json scripts/checkpoint_urls.json
# edit checkpoint_urls.json with real shareable links, then:
python scripts/download_checkpoints.py
```

## Running in Google Colab

Everything above works in Colab, with two differences: `torch`/`torchvision`
come preinstalled (don't reinstall them — use `requirements-colab.txt`), and
the runtime's local disk is wiped every time the session disconnects, so
`data/`, `checkpoints/`, and `eval/runs/` should live on Google Drive instead
or you'll re-download/re-encode everything from scratch each session.

> **These are notebook cells, not runnable code in this file.** Go to
> [colab.research.google.com](https://colab.research.google.com), create a
> notebook there, and paste each block below into its own cell. Do **not**
> use a "Run Cell" button that some editors (e.g. VS Code) show above these
> fenced code blocks inside this README — that runs them locally with a
> plain Python kernel, which doesn't understand the `!`/`%` magic syntax
> below and fails with `SyntaxError: invalid syntax` on the `!`. That
> syntax only works inside a real Jupyter/Colab notebook cell.

Paste these as separate cells in a new notebook:

**1. Turn on a GPU runtime** — `Runtime → Change runtime type → T4 GPU` (or
better, if you have Colab Pro) — *before* running anything else. Then verify:

```python
!nvidia-smi
```

**2. Mount Drive and clone the repo**

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content
!git clone https://github.com/anhkos/mars-clip-bench.git
%cd mars-clip-bench
```

**3. Install dependencies** (Colab already has torch — this skips it)

```python
!pip install -q -r requirements-colab.txt
```

**4. Point data/checkpoints/eval-runs at Drive so they survive disconnects**

```python
import os
DRIVE_ROOT = "/content/drive/MyDrive/mars-clip-bench-data"
for sub in ("data", "checkpoints", "eval_runs"):
    os.makedirs(f"{DRIVE_ROOT}/{sub}", exist_ok=True)

!rm -rf data checkpoints eval/runs
!ln -s {DRIVE_ROOT}/data data
!ln -s {DRIVE_ROOT}/checkpoints checkpoints
!ln -s {DRIVE_ROOT}/eval_runs eval/runs
```

After this, `data/`, `checkpoints/`, and `eval/runs/` are symlinks into your
Drive. Downloads and results persist across sessions — re-running this cell
in a fresh session re-links to what's already there instead of re-downloading.

**5. Download data and checkpoints** (same commands as local setup, just `!`-prefixed)

```python
!python scripts/download_data.py --dataset all
!python finetune/make_hirise_splits.py
```

```python
# after copying scripts/checkpoint_urls.example.json to scripts/checkpoint_urls.json
# and filling in real links (either edit it on Drive directly, or upload it):
!python scripts/download_checkpoints.py
```

**6. Fine-tune or benchmark** — identical commands to the local instructions below,
just `!`-prefixed:

```python
!python finetune/finetune_clip.py
!python eval/model_agnostic_retrieval_eval.py --model siglip --model-id google/siglip-so400m-patch14-384
```

**Session-timeout note:** free-tier Colab disconnects idle sessions after
roughly 90 minutes and hard-caps total runtime around 12 hours. Both
fine-tuning scripts already checkpoint to disk after every epoch that
improves val accuracy, so a mid-training disconnect only loses the current
epoch's progress, not the whole run — as long as `checkpoints/` is the Drive
symlink from step 4, not local disk. To resume, you'd currently need to
re-run the finetune script from scratch (it doesn't auto-resume from a
partial checkpoint) — reduce `EPOCHS` in the script if you're working in
short Colab sessions.

## Fine-tuning

```bash
python finetune/finetune_clip.py           # -> checkpoints/clip_finetuned.pt
python finetune/finetune_clip_hirise.py    # -> checkpoints/hirise_clip_finetuned.pt (needs make_hirise_splits.py first)
```

Both are self-contained (no other teammate-provided files needed) since the
raw datasets are public and the HiRISE splits are regenerated deterministically.

## Benchmarking

One script covers every model family, with the exact same metrics
(P@k, R@k, MAP, NDCG@10) used in `reports/`:

```bash
# zero-shot baselines
python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32
python eval/model_agnostic_retrieval_eval.py --model open_clip --model-id ViT-L-14 --pretrained openai
python eval/model_agnostic_retrieval_eval.py --model siglip --model-id google/siglip-so400m-patch14-384

# fine-tuned
python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --checkpoint checkpoints/clip_finetuned.pt

# HiRISE instead of MSL
python eval/model_agnostic_retrieval_eval.py --model clip --model-id ViT-B/32 --dataset hirise --checkpoint checkpoints/hirise_clip_finetuned.pt
```

Results land in `eval/runs/<model>_<model-id>_<dataset>_results.csv`, per-query
and macro-averaged. Image embeddings are cached per model spec under
`eval/runs/embedding_cache/` so reruns are fast and never cross-contaminate
between model families.

## Project structure

```
mars-clip-bench/
├── requirements.txt / requirements-colab.txt
├── .env.example                   # copy to .env; only needed for gated HF models
├── scripts/
│   ├── download_data.py           # pulls both datasets from Zenodo
│   ├── download_checkpoints.py    # pulls .pt files from cloud storage
│   └── checkpoint_urls.example.json
├── data/                          # gitignored — populated by download_data.py
├── checkpoints/                   # gitignored — populated by download_checkpoints.py
├── finetune/
│   ├── finetune_clip.py
│   ├── finetune_clip_hirise.py
│   └── make_hirise_splits.py
├── eval/
│   ├── model_agnostic_retrieval_eval.py
│   ├── queries/                   # committed — small CSV query sets
│   └── runs/                      # gitignored — result CSVs + embedding cache
└── reports/                       # committed — result summaries from the original repo
```
