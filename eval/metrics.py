"""
Shared retrieval evaluation metrics, used by model_agnostic_retrieval_eval.py.

Macro-averages P@k/R@k/MAP/NDCG@10 across CLASSES, not query rows, so a class
with many query-text variants doesn't get more weight than a class with one.
Also reports a majority-class baseline P@1 for context — HiRISE's "other"
class alone is a large share of the corpus, so a model can look deceptively
good just by always ranking "other" images near the top.
"""

from collections import defaultdict
from typing import Optional, Sequence

import numpy as np


def cosine_similarity_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    return q @ c.T


def precision_at_k(ranked_relevant: np.ndarray, k: int) -> float:
    return float(ranked_relevant[:k].mean()) if len(ranked_relevant) else 0.0


def recall_at_k(ranked_relevant: np.ndarray, k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return float(ranked_relevant[:k].sum() / total_relevant)


def average_precision(ranked_relevant: np.ndarray) -> float:
    hits = np.cumsum(ranked_relevant)
    precisions = hits / (np.arange(len(ranked_relevant)) + 1)
    relevant_precisions = precisions[ranked_relevant.astype(bool)]
    return float(relevant_precisions.mean()) if len(relevant_precisions) else 0.0


def ndcg_at_k(ranked_relevant: np.ndarray, k: int) -> float:
    ranked_relevant = ranked_relevant[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(ranked_relevant) + 2))
    dcg = (ranked_relevant * discounts).sum()
    ideal = np.sort(ranked_relevant)[::-1]
    idcg = (ideal * discounts).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_retrieval(query_embeddings: np.ndarray, query_labels: Sequence,
                        corpus_embeddings: np.ndarray, corpus_labels: Sequence,
                        k_values=(1, 5, 10, 20),
                        exclude_self: bool = False,
                        query_ids: Optional[Sequence] = None,
                        corpus_ids: Optional[Sequence] = None,
                        query_texts: Optional[Sequence[str]] = None) -> dict:
    """
    Set exclude_self=True for image-to-image retrieval, so a query image never
    matches itself (pass query_ids/corpus_ids so we know which corpus row
    corresponds to each query).

    Returns per_query (one row per query, for CSV output / spot-checking),
    per_class (each class's own macro-average), macro_avg (averaged again
    across classes), and a majority-class baseline for context.
    """
    sims = cosine_similarity_matrix(query_embeddings, corpus_embeddings)
    corpus_labels_arr = np.array(corpus_labels)

    per_class = defaultdict(lambda: defaultdict(list))
    per_query = []

    for i, qlabel in enumerate(query_labels):
        row = sims[i].copy()
        if exclude_self and query_ids is not None and corpus_ids is not None:
            self_idx = [j for j, cid in enumerate(corpus_ids) if cid == query_ids[i]]
            row[self_idx] = -np.inf
        order = np.argsort(-row)
        ranked_relevant = (corpus_labels_arr[order] == qlabel).astype(float)
        total_relevant = int((corpus_labels_arr == qlabel).sum())

        row_metrics = {"class_label": qlabel, "num_relevant": total_relevant}
        if query_texts is not None:
            row_metrics["query_text"] = query_texts[i]

        for k in k_values:
            p = precision_at_k(ranked_relevant, k)
            r = recall_at_k(ranked_relevant, k, total_relevant)
            row_metrics[f"p@{k}"] = round(p, 4)
            row_metrics[f"r@{k}"] = round(r, 4)
            per_class[qlabel][f"P@{k}"].append(p)
            per_class[qlabel][f"R@{k}"].append(r)

        ap = average_precision(ranked_relevant)
        ndcg10 = ndcg_at_k(ranked_relevant, 10)
        row_metrics["map"] = round(ap, 4)
        row_metrics["ndcg@10"] = round(ndcg10, 4)
        per_class[qlabel]["MAP"].append(ap)
        per_class[qlabel]["NDCG@10"].append(ndcg10)

        per_query.append(row_metrics)

    macro = defaultdict(list)
    for metrics in per_class.values():
        for name, values in metrics.items():
            macro[name].append(float(np.mean(values)))
    macro_avg = {name: float(np.mean(values)) for name, values in macro.items()}

    per_class_avg = {
        cls: {name: float(np.mean(values)) for name, values in metrics.items()}
        for cls, metrics in per_class.items()
    }

    labels, counts = np.unique(corpus_labels_arr, return_counts=True)
    majority_share = float(counts.max() / counts.sum()) if len(counts) else 0.0

    return {
        "macro_avg": macro_avg,
        "per_class": per_class_avg,
        "per_query": per_query,
        "majority_class_baseline_p@1": majority_share,
        "num_queries": len(query_labels),
        "num_corpus": len(corpus_labels),
    }


def print_report(results: dict, model_name: str):
    print(f"\n{'-' * 60}")
    print(f"SUMMARY -- {model_name}")
    print(f"{'-' * 60}")
    print(f"  ({results['num_queries']} queries against {results['num_corpus']} corpus images)")
    for name, value in results["macro_avg"].items():
        print(f"  {name:8s}= {value:.4f}")
    print(
        f"  majority-class baseline P@1 = {results['majority_class_baseline_p@1']:.4f}  "
        f"<- if P@1 above isn't well above this, the model may just be riding class imbalance"
    )
