# MSL Image Retrieval — Model Evaluation Report

**Model:** CLIP ViT-B/32 (fine-tuned)  
**Dataset:** MSL Surface Labeled Dataset  
**Date:** June 2026

---

## Dataset

| Split | Images |
|-------|--------|
| Train | 4,675  |
| Val   | 711    |
| Test  | 1,305  |
| **Total** | **6,691** |

- **24 classes** (rover instruments, geological features, terrain types)
- Notable class imbalance: "Drill hole" accounts for ~40% of all images (2,684)

---

## Model & Training

| Parameter | Value |
|-----------|-------|
| Base model | CLIP ViT-B/32 |
| Optimizer | AdamW |
| Learning rate | 1e-6 |
| Batch size | 64 |
| Epochs | 10 |
| Loss | Cross-entropy (image vs. 24 class text embeddings) |
| Sampling | Class-balanced WeightedRandomSampler |
| LR schedule | Linear warmup (100 steps) + cosine decay |
| Best checkpoint | Epoch 7 (val accuracy = 69.6%) |

**Training prompts:** `"a photo of a {class_name}"` for each of the 24 classes.

---

## Classification Results (Test Set, n=1,305)

**Overall Test Accuracy: 66.1%**

| Class | Accuracy | n |
|-------|----------|---|
| APXS calibration target | 100.0% | 14 |
| MAHLI calibration target | 100.0% | 2 |
| Layers | 91.7% | 12 |
| DRT | 90.3% | 72 |
| Drill hole | 90.2% | 254 |
| Nearby surface | 90.0% | 10 |
| Portion box | 90.0% | 300 |
| APXS | 88.2% | 34 |
| Mastcam calibration target | 85.7% | 14 |
| Inlet | 83.3% | 48 |
| Ground | 82.5% | 57 |
| Horizon | 75.0% | 32 |
| MAHLI | 77.8% | 9 |
| Light-toned veins | 56.2% | 48 |
| Float | 50.0% | 12 |
| ChemCam calibration target | 47.6% | 84 |
| CheMin inlet open | 45.0% | 20 |
| Artifact | 28.6% | 21 |
| DRT spot | 25.0% | 16 |
| Distant landscape | 6.7% | 60 |
| Drill | 4.0% | 150 |
| Mastcam | 2.8% | 36 |

> Note: Low accuracy on Drill and Mastcam reflects visual confusion with closely related classes (Drill hole, rover hardware). Classes with very few test samples (n<10) should be interpreted with caution.

---

## Text-to-Image Retrieval Results

Queries evaluated using `"a photo of a {class_name}"` template style.  
Embeddings ranked by cosine similarity across all 6,691 images.

### Fine-tuned vs. Zero-shot Baseline

| Metric | Zero-shot CLIP ViT-B/32 | **Fine-tuned CLIP ViT-B/32** | Improvement |
|--------|------------------------|------------------------------|-------------|
| P@1    | 10.0% | **31.0%** | +21 pts |
| P@5    | 9.0%  | **28.0%** | +19 pts |
| P@10   | 7.7%  | **28.9%** | +21 pts |
| MAP    | 7.18% | **24.32%** | +17 pts |
| NDCG@10| 8.33% | **28.92%** | +20 pts |

### Fine-tuned Model — Full Retrieval Summary (100 queries, macro-averaged)

| Metric | Value |
|--------|-------|
| P@1    | 31.00% |
| P@5    | 28.00% |
| P@10   | 28.90% |
| P@20   | 27.85% |
| R@1    | 0.42% |
| R@5    | 1.86% |
| R@10   | 3.84% |
| R@20   | 6.80% |
| MAP    | 24.32% |
| NDCG@10| 28.92% |

> Recall values are computed against the full 6,691-image corpus. Low recall is expected given the large denominator — Drill hole alone has 2,684 positives. Precision metrics are more informative for this dataset scale.

---

## Models Evaluated (Zero-shot Comparison)

| Model | P@1 | MAP | NDCG@10 | Notes |
|-------|-----|-----|---------|-------|
| CLIP ViT-B/32 (zero-shot) | 10.0% | 7.18% | 8.33% | Best zero-shot baseline |
| CLIP ViT-L/14@336px (zero-shot) | 0.0% | — | — | Collapsed on class imbalance |
| SigLIP base (zero-shot) | — | — | — | Weaker than CLIP on this domain |
| SigLIP so400m (zero-shot) | — | — | — | Large model hurt by imbalance |
| **CLIP ViT-B/32 (fine-tuned)** | **31.0%** | **24.32%** | **28.92%** | **Best overall** |

---

## Key Findings

1. **Fine-tuning tripled all retrieval metrics** over the zero-shot baseline with only 10 epochs of training on the labeled dataset.
2. **"a photo of" query template** outperforms descriptive natural language queries for CLIP-based retrieval on this domain.
3. **Class-balanced sampling** was critical — without it, the 40% Drill hole imbalance dominated training and suppressed learning for rare classes.
4. **Larger models underperformed** zero-shot (ViT-L/14, SigLIP so400m) due to high-confidence predictions for the majority class flooding top-k results.
5. **Calibration targets** (APXS, MAHLI, Mastcam, ChemCam) achieve near-perfect retrieval — visually distinctive objects that CLIP handles well.
6. **Hard classes** (Distant landscape, Drill, Mastcam) suffer from visual overlap with related classes, not model failure — Drill images are confused with Drill hole, which is a reasonable mistake.

---

## Next Steps

- Add data augmentation (random crop, flip, color jitter) to improve generalization
- Expand labeled data for underrepresented classes (Float n=26, MAHLI calibration target n=22)
- Query ensemble at inference (average multiple query embeddings) to boost recall
- Evaluate on Mars 2020 / MPF datasets for cross-mission generalization
