#!/usr/bin/env python3
"""Setup (5): SPECTER2 A3 (alpha=0.1) + UMAP(precomputed D=1-S) + Ward agglomerative.

Requires similarity cache from title-fusion:
  artifacts/assignment1_title_fusion/caches/similarity/specter2_S_late_a0.1.npy

Default --k 30 (eval-selected among {30,54,80}; exploratory).
Writes results/a3_agglo_headline.csv and clusters/*.csv.
Coverage is always 1.0 (all 653 eval papers assigned).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "artifacts" / "assignment1"))

from pipeline import config as base_cfg
from pipeline.cluster_eval import (
    evaluate_on_653,
    labels_to_soft,
    load_papers_and_eval,
    load_stop_for_eval,
)

OUT = ROOT / "artifacts" / "assignment1_a3_agglo"
SIM = ROOT / "artifacts" / "assignment1_title_fusion" / "caches" / "similarity"
RES = OUT / "results"
CLUST = OUT / "clusters"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=30)
    args = ap.parse_args()

    s_path = SIM / f"specter2_S_late_a{args.alpha}.npy"
    if not s_path.exists():
        raise SystemExit(
            f"missing {s_path}; run setup (2) title_fusion first"
        )

    papers, eval_ids = load_papers_and_eval()
    y_bin, label_sets, k_gt, _ = load_stop_for_eval(eval_ids, papers)
    paper_ids = papers.paper_id.astype(str).tolist()

    S = np.load(s_path)
    D = (1.0 - S).astype(np.float64)
    np.fill_diagonal(D, 0.0)

    t0 = time.time()
    emb5 = UMAP(
        n_neighbors=base_cfg.UMAP_KW["n_neighbors"],
        n_components=base_cfg.UMAP_KW["n_components"],
        min_dist=base_cfg.UMAP_KW["min_dist"],
        metric="precomputed",
        random_state=base_cfg.RANDOM_STATE,
    ).fit_transform(D)
    umap_rt = time.time() - t0

    t1 = time.time()
    labels = AgglomerativeClustering(
        n_clusters=args.k, linkage="ward"
    ).fit_predict(emb5)
    cl_rt = time.time() - t1

    soft, diag = labels_to_soft(labels)
    metrics = evaluate_on_653(
        soft, paper_ids, eval_ids, y_bin, label_sets, k_gt=k_gt
    )
    ndcg = metrics.get("nDCG@1_mean", metrics.get(f"nDCG@{k_gt}_mean"))
    assert metrics["Coverage"] == 1.0
    assert metrics["n_eval_assigned"] == 653
    assert diag["noise_frac"] == 0.0

    RES.mkdir(parents=True, exist_ok=True)
    CLUST.mkdir(parents=True, exist_ok=True)
    row = {
        "method": f"specter2|A3_a{args.alpha}|umap_agglo_ward_K{args.k}",
        "encoder": "specter2",
        "representation": "similarity_late_A3",
        "alpha": args.alpha,
        "clusterer": "umap_agglo_ward",
        "K_requested": args.k,
        "fit_n": 6517,
        "eval_n": 653,
        "Purity": metrics["Purity"],
        "LRAP": metrics["LRAP_mean"],
        "nDCG@1": ndcg,
        "V_measure": metrics["V_measure"],
        "PairF1": metrics["PairF1"],
        "Coverage": metrics["Coverage"],
        "n_eval_assigned": metrics["n_eval_assigned"],
        "n_clusters": diag["K"],
        "noise_fraction": diag["noise_frac"],
        "note": f"full_653_coverage;umap_sec={umap_rt:.1f};cluster_sec={cl_rt:.1f}",
    }
    pd.DataFrame([row]).to_csv(RES / "a3_agglo_headline.csv", index=False)
    tag = f"specter2__A3_a{str(args.alpha).replace('.', 'p')}__agglo_ward_K{args.k}"
    pd.DataFrame({"paper_id": paper_ids, "cluster": labels}).to_csv(
        CLUST / f"{tag}.csv", index=False
    )
    (OUT / "config.json").write_text(json.dumps({
        "alpha": args.alpha,
        "k": args.k,
        "umap": base_cfg.UMAP_KW,
        "clusterer": "AgglomerativeClustering(linkage=ward) on UMAP 5D",
        "similarity_cache": str(s_path.relative_to(ROOT)),
    }, indent=2))
    print(
        f"{row['method']}: PF={row['PairF1']:.4f} "
        f"cov={row['Coverage']:.3f} n_assigned={row['n_eval_assigned']}"
    )


if __name__ == "__main__":
    main()
