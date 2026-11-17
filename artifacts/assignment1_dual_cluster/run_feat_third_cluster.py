#!/usr/bin/env python3
"""Third clustering from dual views:

Features per paper:
  [L2(title_emb) | L2(abstract_emb) | onehot(title_cluster) | onehot(abstract_cluster)]

Optionally scale the one-hot block so cluster IDs don't dominate / get washed out
by high-dim embeddings. Then UMAP+HDBSCAN once more.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

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


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def load_emb(encoder: str, kind: str) -> np.ndarray:
    cands = [
        TF_EMB / encoder / f"{kind}.npy",
        A1_EMB / encoder / "R0_full_abstract.npy" if kind == "abstract_R0" else None,
    ]
    for c in cands:
        if c is not None and c.exists():
            return np.load(c)
    raise FileNotFoundError(encoder, kind)


def one_hot_clusters(labels: np.ndarray) -> np.ndarray:
    """One-hot including a dedicated noise column for label == -1."""
    labels = np.asarray(labels).astype(np.int32)
    valid = sorted({int(c) for c in np.unique(labels) if c >= 0})
    # columns: noise + valid clusters
    idx = {-1: 0}
    for i, c in enumerate(valid, start=1):
        idx[c] = i
    mat = np.zeros((len(labels), len(valid) + 1), dtype=np.float32)
    for i, c in enumerate(labels):
        mat[i, idx[int(c) if int(c) >= 0 else -1]] = 1.0
    return mat


def build_features(Et, Ea, lt, la, cluster_weight: float) -> np.ndarray:
    Et_n, Ea_n = _l2(Et), _l2(Ea)
    ot, oa = one_hot_clusters(lt), one_hot_clusters(la)
    # scale one-hots so their L2 mass is comparable to embeddings * weight
    # each one-hot row has L2=1; emb rows have L2=1 → weight controls relative influence
    X = np.hstack([Et_n, Ea_n, ot * cluster_weight, oa * cluster_weight]).astype(np.float32)
    return _l2(X)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="specter2,semcse,qwen")
    ap.add_argument("--cluster-weights", default="0.5,1.0,2.0",
                    help="relative weight of one-hot cluster features vs embeddings")
    args = ap.parse_args()

    papers, eval_ids = load_papers_and_eval()
    paper_ids = papers.paper_id.astype(str).tolist()
    y_bin, label_sets, k_gt, _ = load_stop_for_eval(eval_ids, papers)

    prev_path = RES / "dual_cluster_results.csv"
    prev = pd.read_csv(prev_path) if prev_path.exists() else pd.DataFrame()
    # drop older feat_third rows if re-running
    if not prev.empty and "aggregation" in prev.columns:
        prev = prev[~prev.aggregation.astype(str).str.startswith("feat_third")].copy()

    weights = [float(x) for x in args.cluster_weights.split(",") if x.strip()]
    rows = prev.to_dict("records") if not prev.empty else []
    encs = [e.strip() for e in args.encoders.split(",") if e.strip()]

    meta = {
        "method": "feat_third",
        "features": "L2(title_emb)|L2(abs_emb)|w*onehot(title_cl)|w*onehot(abs_cl)",
        "cluster_weights": weights,
        "clustering": "UMAP+HDBSCAN same as Regime A",
    }
    (OUT / "feat_third_config.json").write_text(json.dumps(meta, indent=2))

    for enc in encs:
        print(f"\n===== {enc} feat_third =====")
        Et = load_emb(enc, "title_T0")
        Ea = load_emb(enc, "abstract_R0")
        lt = pd.read_csv(CLUST / f"{enc}__title_only.csv").cluster.to_numpy()
        la = pd.read_csv(CLUST / f"{enc}__abstract_only.csv").cluster.to_numpy()
        assert len(lt) == len(paper_ids) == len(la) == len(Et)

        for w in weights:
            X = build_features(Et, Ea, lt, la, cluster_weight=w)
            print(f"{enc} w={w}: feature dim={X.shape[1]}")
            t0 = time.time()
            labels, _ = cluster_umap_hdbscan(X)
            rt = time.time() - t0
            soft, diag = labels_to_soft(labels)
            metrics = evaluate_on_653(soft, paper_ids, eval_ids, y_bin, label_sets, k_gt=k_gt)
            ndcg = metrics.get("nDCG@1_mean", metrics.get(f"nDCG@{k_gt}_mean"))
            tag = f"{enc}|dual_feat_third_w{w}"
            pd.DataFrame({"paper_id": paper_ids, "cluster": labels}).to_csv(
                CLUST / f"{enc}__feat_third_w{str(w).replace('.', 'p')}.csv", index=False)
            row = {
                "method": tag,
                "aggregation": f"feat_third_w{w}",
                "fit_n": 6517,
                "eval_n": 653,
                "Purity": metrics["Purity"],
                "LRAP": metrics["LRAP_mean"],
                "nDCG@1": ndcg,
                "V_measure": metrics["V_measure"],
                "PairF1": metrics["PairF1"],
                "Coverage": metrics["Coverage"],
                "n_clusters": diag["K"],
                "noise_fraction": diag["noise_frac"],
                "runtime_sec": rt,
                "cluster_weight": w,
                "feature_dim": X.shape[1],
            }
            rows.append(row)
            print(f"{tag}: PF={row['PairF1']:.4f} Pur={row['Purity']:.4f} "
                  f"K={row['n_clusters']} noise={row['noise_fraction']:.3f}")

        pd.DataFrame(rows).to_csv(RES / "dual_cluster_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "dual_cluster_results.csv", index=False)
    print("\n=== feat_third vs baselines (PairF1) ===")
    show = df[df.method.str.contains("abstract_cluster|title_cluster|feat_third|soft_concat|coassoc")]
    print(show.sort_values(["method"])[["method", "aggregation", "PairF1", "Purity", "n_clusters"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
