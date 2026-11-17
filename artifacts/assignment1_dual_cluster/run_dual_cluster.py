#!/usr/bin/env python3
"""Dual clustering on 6517: cluster titles and abstracts separately, then aggregate.

Aggregation variants:
  - soft_concat: concatenate soft membership bags from both clusterings, evaluate once
  - coassoc: co-association S_ij = 1[same title cl] + 1[same abs cl], then
             UMAP(precomputed) on D=1-S/2 → HDBSCAN
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from umap import UMAP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "assignment1"))

from pipeline import config as base_cfg
from pipeline.cluster_eval import (
    cluster_umap_hdbscan,
    evaluate_on_653,
    labels_to_soft,
    load_papers_and_eval,
    load_stop_for_eval,
)

OUT = ROOT / "artifacts" / "assignment1_dual_cluster"
TF_EMB = ROOT / "artifacts" / "assignment1_title_fusion" / "caches" / "embeddings"
A1_EMB = ROOT / "artifacts" / "assignment1" / "caches" / "embeddings"
CLUST = OUT / "clusters"
RES = OUT / "results"
ENCODERS = ("specter2", "semcse", "qwen")


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def load_emb(encoder: str, kind: str) -> np.ndarray:
    """kind in {title_T0, abstract_R0}."""
    cands = [
        TF_EMB / encoder / f"{kind}.npy",
        A1_EMB / encoder / "R0_full_abstract.npy" if kind == "abstract_R0" else None,
    ]
    for c in cands:
        if c is not None and c.exists():
            E = np.load(c)
            print(f"load {encoder}/{kind} from {c} {E.shape}")
            return E
    raise FileNotFoundError(f"missing embeddings for {encoder}/{kind}")


def cluster_size_stats(labels: np.ndarray) -> dict:
    labs = labels[labels >= 0]
    if len(labs) == 0:
        return {"largest": 0, "median_size": 0.0, "n_singletons": 0}
    sizes = np.bincount(labs)
    sizes = sizes[sizes > 0]
    return {
        "largest": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "n_singletons": int((sizes == 1).sum()),
    }


def eval_row(name, soft, paper_ids, eval_ids, y_bin, label_sets, k_gt, labels=None, extra=None):
    metrics = evaluate_on_653(soft, paper_ids, eval_ids, y_bin, label_sets, k_gt=k_gt)
    ndcg = metrics.get("nDCG@1_mean", metrics.get(f"nDCG@{k_gt}_mean"))
    row = {
        "method": name,
        "fit_n": 6517,
        "eval_n": 653,
        "Purity": metrics["Purity"],
        "LRAP": metrics["LRAP_mean"],
        "nDCG@1": ndcg,
        "V_measure": metrics["V_measure"],
        "PairF1": metrics["PairF1"],
        "Coverage": metrics["Coverage"],
        "n_clusters": soft.shape[1] if labels is None else int(len(set(labels[labels >= 0]))),
        "noise_fraction": float((labels < 0).mean()) if labels is not None else np.nan,
    }
    if labels is not None:
        row.update(cluster_size_stats(labels))
    if extra:
        row.update(extra)
    print(f"{name}: PF={row['PairF1']:.4f} Pur={row['Purity']:.4f} "
          f"K={row['n_clusters']} cov={row['Coverage']:.3f}")
    return row


def soft_concat(soft_t, soft_a):
    """Concatenate soft bags; L1-normalize rows with mass."""
    m = np.hstack([soft_t, soft_a]).astype(np.float64)
    s = m.sum(axis=1, keepdims=True)
    nz = s.ravel() > 0
    m[nz] = m[nz] / s[nz]
    return m


def coassoc_cluster(labels_t, labels_a, batch: int = 512):
    """Co-association: S_ij = I[same title] + I[same abs]; D = 1 - S/2."""
    n = len(labels_t)
    lt = labels_t.astype(np.int32)
    la = labels_a.astype(np.int32)
    # Build dense S in blocks to limit peak mem (~6517^2 * 4 ≈ 170MB float32 OK)
    S = np.zeros((n, n), dtype=np.float32)
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        # title agree (ignore noise -1)
        ti = lt[i0:i1][:, None]
        tj = lt[None, :]
        title_same = (ti == tj) & (ti >= 0) & (tj >= 0)
        ai = la[i0:i1][:, None]
        aj = la[None, :]
        abs_same = (ai == aj) & (ai >= 0) & (aj >= 0)
        S[i0:i1] = title_same.astype(np.float32) + abs_same.astype(np.float32)
    S = 0.5 * (S + S.T)
    np.fill_diagonal(S, 2.0)
    D = (1.0 - S / 2.0).astype(np.float64)
    np.fill_diagonal(D, 0.0)
    t0 = time.time()
    emb5 = UMAP(
        n_neighbors=base_cfg.UMAP_KW["n_neighbors"],
        n_components=base_cfg.UMAP_KW["n_components"],
        min_dist=base_cfg.UMAP_KW["min_dist"],
        metric="precomputed",
        random_state=base_cfg.RANDOM_STATE,
    ).fit_transform(D)
    labels = HDBSCAN(**base_cfg.HDBSCAN_KW).fit_predict(emb5)
    return labels, time.time() - t0


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="specter2,semcse,qwen")
    args = ap.parse_args()

    for d in (OUT, CLUST, RES):
        d.mkdir(parents=True, exist_ok=True)

    papers, eval_ids = load_papers_and_eval()
    paper_ids = papers.paper_id.astype(str).tolist()
    y_bin, label_sets, k_gt, _ = load_stop_for_eval(eval_ids, papers)

    (OUT / "config.json").write_text(json.dumps({
        "fit_n": 6517,
        "eval_n": 653,
        "umap": base_cfg.UMAP_KW,
        "hdbscan": base_cfg.HDBSCAN_KW,
        "aggregations": ["soft_concat", "coassoc"],
        "note": "Cluster titles and abstracts separately, then aggregate",
    }, indent=2, default=str))

    rows = []
    encs = [e.strip() for e in args.encoders.split(",") if e.strip()]

    for enc in encs:
        print(f"\n===== {enc} =====")
        Et = load_emb(enc, "title_T0")
        Ea = load_emb(enc, "abstract_R0")
        assert len(Et) == 6517 and len(Ea) == 6517

        # SPECTER2 embeddings often used unnormalized in prior runs
        Et_in = Et if enc == "specter2" else _l2(Et)
        Ea_in = Ea if enc == "specter2" else _l2(Ea)

        labels_t, rt_t = cluster_umap_hdbscan(Et_in)
        labels_a, rt_a = cluster_umap_hdbscan(Ea_in)
        soft_t, diag_t = labels_to_soft(labels_t)
        soft_a, diag_a = labels_to_soft(labels_a)

        pd.DataFrame({"paper_id": paper_ids, "cluster": labels_t}).to_csv(
            CLUST / f"{enc}__title_only.csv", index=False)
        pd.DataFrame({"paper_id": paper_ids, "cluster": labels_a}).to_csv(
            CLUST / f"{enc}__abstract_only.csv", index=False)

        rows.append(eval_row(
            f"{enc}|title_cluster", soft_t, paper_ids, eval_ids, y_bin, label_sets, k_gt,
            labels=labels_t, extra={"aggregation": "none", "runtime_sec": rt_t,
                                    "K_title": diag_t["K"], "K_abstract": ""}))
        rows.append(eval_row(
            f"{enc}|abstract_cluster", soft_a, paper_ids, eval_ids, y_bin, label_sets, k_gt,
            labels=labels_a, extra={"aggregation": "none", "runtime_sec": rt_a,
                                    "K_title": "", "K_abstract": diag_a["K"]}))

        # Aggregate 1: soft concat
        soft_c = soft_concat(soft_t, soft_a)
        # synthetic hard labels for noise stats: prefer abstract if assigned else title
        hard = np.full(len(paper_ids), -1, dtype=np.int32)
        for i in range(len(paper_ids)):
            if labels_a[i] >= 0:
                hard[i] = labels_a[i]
            elif labels_t[i] >= 0:
                hard[i] = labels_t[i] + 10_000  # offset to keep ids distinct in stats only
        rows.append(eval_row(
            f"{enc}|dual_soft_concat", soft_c, paper_ids, eval_ids, y_bin, label_sets, k_gt,
            labels=None,
            extra={"aggregation": "soft_concat", "runtime_sec": rt_t + rt_a,
                   "K_title": diag_t["K"], "K_abstract": diag_a["K"],
                   "n_clusters": soft_c.shape[1],
                   "noise_fraction": float(((labels_t < 0) & (labels_a < 0)).mean())}))

        # Aggregate 2: co-association recluster
        print(f"{enc}: building co-association…")
        labels_co, rt_co = coassoc_cluster(labels_t, labels_a)
        soft_co, diag_co = labels_to_soft(labels_co)
        pd.DataFrame({"paper_id": paper_ids, "cluster": labels_co}).to_csv(
            CLUST / f"{enc}__coassoc.csv", index=False)
        rows.append(eval_row(
            f"{enc}|dual_coassoc", soft_co, paper_ids, eval_ids, y_bin, label_sets, k_gt,
            labels=labels_co,
            extra={"aggregation": "coassoc", "runtime_sec": rt_t + rt_a + rt_co,
                   "K_title": diag_t["K"], "K_abstract": diag_a["K"]}))

        pd.DataFrame(rows).to_csv(RES / "dual_cluster_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "dual_cluster_results.csv", index=False)
    print("\n=== SUMMARY ===")
    print(df[["method", "aggregation", "PairF1", "Purity", "n_clusters", "noise_fraction"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
