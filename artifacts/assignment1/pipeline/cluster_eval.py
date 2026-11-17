"""UMAP+HDBSCAN clustering and SToP evaluation (paper / extension protocol)."""
from __future__ import annotations

import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from sklearn.metrics import (
    label_ranking_average_precision_score,
    ndcg_score,
    v_measure_score,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MultiLabelBinarizer
from umap import UMAP

from . import config


def norm_title(t: str) -> str:
    t = unicodedata.normalize("NFKC", str(t)).casefold()
    t = re.sub(r"\s+", " ", t).strip()
    return re.sub(r"[^a-z0-9\s]", "", t)


def load_papers_and_eval():
    papers = pd.read_csv(config.PAPERS_CSV)
    eval_ids = pd.read_csv(config.EVAL653_CSV)
    assert len(papers) == 6517
    assert len(eval_ids) == 653
    return papers, eval_ids


def load_stop_for_eval(eval_df: pd.DataFrame, papers: pd.DataFrame):
    """Return (y_bin, stop_label_sets, k_gt, STOP_CATEGORIES) aligned to eval_df order."""
    stop = pd.read_csv(config.STOP_CSV, usecols=["Labels", "Paper Name"])
    stop["title_norm"] = stop["Paper Name"].map(norm_title)
    stop_labels: dict[str, frozenset] = {}
    for title, group in stop.groupby("title_norm"):
        stop_labels[title] = frozenset(group["Labels"].dropna().unique())
    STOP_CATEGORIES = sorted(stop["Labels"].dropna().unique().tolist())

    papers_idx = papers.set_index("paper_id")
    label_sets = []
    for pid in eval_df["paper_id"]:
        t = norm_title(papers_idx.loc[pid, "title"])
        labs = stop_labels.get(t)
        if labs is None:
            raise KeyError(f"No SToP labels for {pid} title_norm={t}")
        label_sets.append(labs)

    mlb = MultiLabelBinarizer(classes=STOP_CATEGORIES)
    y_bin = mlb.fit_transform(label_sets).astype(np.float64)
    gt_sizes = [len(s) for s in label_sets]
    k_gt = max(1, round(sum(gt_sizes) / len(gt_sizes)))
    print(f"Eval papers={len(eval_df)} categories={len(STOP_CATEGORIES)} "
          f"avg_labels={np.mean(gt_sizes):.2f} -> nDCG@{k_gt}")
    return y_bin, label_sets, k_gt, STOP_CATEGORIES


def cluster_umap_hdbscan(E: np.ndarray, random_state: int = 42):
    t0 = time.time()
    umap_kw = dict(config.UMAP_KW)
    umap_kw["random_state"] = random_state
    emb5 = UMAP(**umap_kw).fit_transform(E)
    labels = HDBSCAN(**config.HDBSCAN_KW).fit_predict(emb5)
    return labels, time.time() - t0


def cluster_hdbscan_precomputed(D: np.ndarray):
    t0 = time.time()
    labels = HDBSCAN(
        min_cluster_size=config.HDBSCAN_KW["min_cluster_size"],
        min_samples=config.HDBSCAN_KW["min_samples"],
        metric="precomputed",
        cluster_selection_method="eom",
    ).fit_predict(D)
    return labels, time.time() - t0


def cluster_agglo_precomputed(D: np.ndarray, n_clusters: int = 80):
    from sklearn.cluster import AgglomerativeClustering
    t0 = time.time()
    labels = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average",
    ).fit_predict(D)
    return labels, time.time() - t0


def labels_to_soft(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    labels = np.asarray(labels)
    valid = sorted({int(c) for c in np.unique(labels) if int(c) >= 0})
    idx = {c: i for i, c in enumerate(valid)}
    mat = np.zeros((len(labels), max(len(valid), 1)), dtype=np.float64)
    for i, c in enumerate(labels):
        c = int(c)
        if c in idx:
            mat[i, idx[c]] = 1.0
    if valid:
        sizes = np.bincount(labels[labels >= 0])
        sizes = sizes[sizes > 0]
        diag = {
            "K": len(valid),
            "noise_frac": float((labels < 0).mean()),
            "n_assigned": int((labels >= 0).sum()),
            "largest": int(sizes.max()),
            "median_size": float(np.median(sizes)),
        }
    else:
        diag = {"K": 0, "noise_frac": 1.0, "n_assigned": 0, "largest": 0, "median_size": 0.0}
    return mat, diag


def soft_cluster_purity(doc_matrix: np.ndarray, true_labels: list) -> float:
    purities = []
    for t in range(doc_matrix.shape[1]):
        weights = doc_matrix[:, t]
        total_w = weights.sum()
        if total_w < 1e-6:
            continue
        label_w: dict = {}
        for w, lbls in zip(weights, true_labels):
            if w < 1e-6:
                continue
            for lbl in lbls:
                label_w[lbl] = label_w.get(lbl, 0.0) + float(w)
        if label_w:
            purities.append(max(label_w.values()) / total_w)
    return float(np.mean(purities)) if purities else float("nan")


def learn_topic_to_label_matrix(doc_topic: np.ndarray, y_bin: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    w = doc_topic.T @ y_bin
    w = w + eps
    w = w / w.sum(axis=1, keepdims=True)
    return w


def pairwise_f1_sampled(y_bin, hard_labels, n_pairs=200_000, random_state=42):
    n = y_bin.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(random_state)
    i_idx = rng.integers(0, n, size=n_pairs)
    j_idx = rng.integers(0, n, size=n_pairs)
    valid = i_idx != j_idx
    i_idx, j_idx = i_idx[valid], j_idx[valid]
    same_gt = (y_bin[i_idx] * y_bin[j_idx]).sum(axis=1) > 0
    same_pred = hard_labels[i_idx] == hard_labels[j_idx]
    tp = np.sum(same_gt & same_pred)
    fp = np.sum(~same_gt & same_pred)
    fn = np.sum(same_gt & ~same_pred)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return float(f1)


def evaluate_on_653(doc_topics_full: np.ndarray, paper_ids_full: list[str],
                    eval_ids: pd.DataFrame, y_bin: np.ndarray,
                    label_sets: list, k_gt: int = 1) -> dict:
    pid_to_i = {p: i for i, p in enumerate(paper_ids_full)}
    rows = [pid_to_i[p] for p in eval_ids["paper_id"]]
    mat = np.clip(np.nan_to_num(doc_topics_full[rows].astype(np.float64)), 0.0, None)
    nz = mat.sum(axis=1) > 0
    pur = soft_cluster_purity(mat, label_sets)

    x, yy = mat[nz], y_bin[nz]
    labelset = np.array(["|".join(sorted(s)) if len(s) else "none" for s in label_sets])
    ls = labelset[nz]

    if x.shape[0] < 10 or x.shape[1] < 1:
        return dict(
            Purity=pur, Coverage=float(nz.mean()),
            LRAP_mean=float("nan"), LRAP_std=float("nan"),
            **{f"nDCG@{k_gt}_mean": float("nan"), f"nDCG@{k_gt}_std": float("nan")},
            V_measure=float("nan"), PairF1=float("nan"),
            n_eval=len(mat), n_eval_assigned=int(nz.sum()),
        )

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=config.RANDOM_STATE)
    lv, dv = [], []
    for tr, te in rkf.split(x):
        P = learn_topic_to_label_matrix(x[tr], yy[tr])
        scores = x[te] @ P
        lv.append(label_ranking_average_precision_score(yy[te], scores))
        dv.append(ndcg_score(yy[te], scores, k=k_gt))
    hard = np.argmax(x, axis=1)
    vm = float(v_measure_score(ls, hard))
    pf = pairwise_f1_sampled(yy.astype(np.int8), hard, n_pairs=config.PAIRF1_PAIRS,
                             random_state=config.RANDOM_STATE)
    return dict(
        Purity=pur,
        Coverage=float(nz.mean()),
        LRAP_mean=float(np.mean(lv)),
        LRAP_std=float(np.std(lv, ddof=1)),
        **{f"nDCG@{k_gt}_mean": float(np.mean(dv)), f"nDCG@{k_gt}_std": float(np.std(dv, ddof=1))},
        V_measure=vm,
        PairF1=pf,
        n_eval=len(mat),
        n_eval_assigned=int(nz.sum()),
        k_gt=k_gt,
    )


def run_vector_clustering(E: np.ndarray, paper_ids: list[str], eval_ids: pd.DataFrame,
                          y_bin, label_sets, k_gt: int, name: str) -> dict:
    labels, runtime = cluster_umap_hdbscan(E)
    soft, diag = labels_to_soft(labels)
    metrics = evaluate_on_653(soft, paper_ids, eval_ids, y_bin, label_sets, k_gt=k_gt)
    out = {
        "method": name,
        "family": "our_method",
        "role": "Our method",
        "runtime_sec": runtime,
        **diag,
        **metrics,
    }
    config.CLUST_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-zA-Z0-9_+\-]+", "_", name)
    pd.DataFrame({"paper_id": paper_ids, "cluster": labels}).to_csv(
        config.CLUST_DIR / f"{safe}.csv", index=False
    )
    print(f"{name}: K={out['K']} cov={out['Coverage']:.3f} "
          f"Pur={out['Purity']:.4f} PF={out['PairF1']:.4f}")
    return out
