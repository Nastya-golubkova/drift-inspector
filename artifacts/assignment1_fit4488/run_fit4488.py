#!/usr/bin/env python3
"""ACC-matched fit-4488 title/abstract fusion (Regime B).

Slices existing 6517 paper-level embeddings to the exact ACC study paper IDs
from artifacts/baseline/acc_clusters_emnlp.csv, then re-clusters with the same
UMAP+HDBSCAN config as Regime A. Does not overwrite Regime A artifacts.
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
from pipeline.encoders import embed_qwen, embed_semcse, embed_specter2

OUT = ROOT / "artifacts" / "assignment1_fit4488"
TF_EMB = ROOT / "artifacts" / "assignment1_title_fusion" / "caches" / "embeddings"
A1_EMB = ROOT / "artifacts" / "assignment1" / "caches" / "embeddings"
ACC_CLUSTERS = ROOT / "artifacts" / "baseline" / "acc_clusters_emnlp.csv"
EMB = OUT / "caches" / "embeddings"
SIM = OUT / "caches" / "similarity"
RES = OUT / "results"
CLUST = OUT / "clusters"
LOGS = OUT / "logs"
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
ENCODERS = ("specter2", "semcse", "qwen")


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def load_fit4488_ids() -> list[str]:
    acc = pd.read_csv(ACC_CLUSTERS)
    ids = sorted(acc.paper_id.astype(str).unique())
    assert len(ids) == 4488, len(ids)
    return ids


def verify_corpus(fit_ids: list[str], eval_ids: pd.DataFrame, papers: pd.DataFrame):
    assert len(fit_ids) == 4488
    assert len(set(fit_ids)) == 4488
    assert eval_ids.paper_id.nunique() == 653
    official = pd.read_csv(base_cfg.EVAL653_CSV)
    assert set(eval_ids.paper_id.astype(str)) == set(official.paper_id.astype(str))
    inter = set(fit_ids) & set(eval_ids.paper_id.astype(str))
    assert len(inter) == 653, len(inter)
    assert set(fit_ids).issubset(set(papers.paper_id.astype(str)))
    print("corpus checks OK: fit=4488, eval=653, intersection=653")


def indices_for(fit_ids: list[str], papers: pd.DataFrame) -> np.ndarray:
    id_to_idx = {str(pid): i for i, pid in enumerate(papers.paper_id.astype(str))}
    return np.array([id_to_idx[pid] for pid in fit_ids], dtype=np.int64)


def encode(encoder: str, texts: list[str], cache: Path) -> np.ndarray:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if encoder == "specter2":
        return embed_specter2(texts, cache, normalize=False)
    if encoder == "semcse":
        return embed_semcse(texts, cache)
    if encoder == "qwen":
        return embed_qwen(texts, cache)
    raise ValueError(encoder)


def load_or_slice_6517(encoder: str, name: str, idx: np.ndarray, texts: list[str] | None) -> np.ndarray:
    """Prefer title-fusion / assignment1 6517 cache, slice to 4488; else encode 4488 texts."""
    candidates = [
        TF_EMB / encoder / f"{name}.npy",
        A1_EMB / encoder / f"{name}.npy",
        A1_EMB / encoder / "R0_full_abstract.npy" if name == "abstract_R0" else None,
    ]
    src = None
    for c in candidates:
        if c is not None and c.exists():
            src = c
            break
    dst = EMB / encoder / f"{name}.npy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src is not None:
        E_full = np.load(src)
        assert len(E_full) >= idx.max() + 1, (src, E_full.shape, idx.max())
        E = E_full[idx]
        np.save(dst, E)
        (dst.with_suffix(".meta.json")).write_text(json.dumps({
            "source": str(src),
            "n_full": int(len(E_full)),
            "n_fit4488": int(len(E)),
            "note": "sliced from 6517 cache to ACC 4488 IDs",
        }, indent=2))
        print(f"slice {encoder}/{name} from {src.name} -> {E.shape}")
        return E
    if texts is None:
        raise FileNotFoundError(f"missing cache for {encoder}/{name} and no texts to encode")
    print(f"encode {encoder}/{name} on 4488 texts (cache miss)")
    return encode(encoder, texts, dst)


def cosine_sim_matrix(E: np.ndarray, batch: int = 1024) -> np.ndarray:
    E = _l2(E)
    n = len(E)
    S = np.empty((n, n), dtype=np.float32)
    Et = E.T
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        S[i0:i1] = E[i0:i1] @ Et
    np.fill_diagonal(S, 1.0)
    S = 0.5 * (S + S.T)
    np.clip(S, -1.0, 1.0, out=S)
    return S


def cluster_umap_precomputed(D: np.ndarray, random_state: int = 42):
    t0 = time.time()
    emb5 = UMAP(
        n_neighbors=base_cfg.UMAP_KW["n_neighbors"],
        n_components=base_cfg.UMAP_KW["n_components"],
        min_dist=base_cfg.UMAP_KW["min_dist"],
        metric="precomputed",
        random_state=random_state,
    ).fit_transform(D)
    labels = HDBSCAN(**base_cfg.HDBSCAN_KW).fit_predict(emb5)
    return labels, time.time() - t0


def cluster_size_stats(labels: np.ndarray) -> dict:
    labs = labels[labels >= 0]
    if len(labs) == 0:
        return {"largest": 0, "median_size": 0.0, "mean_size": 0.0, "n_singletons": 0}
    sizes = np.bincount(labs)
    sizes = sizes[sizes > 0]
    return {
        "largest": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "mean_size": float(sizes.mean()),
        "n_singletons": int((sizes == 1).sum()),
    }


def run_one(name_enc, representation, alpha, E_or_labels, paper_ids, eval_ids,
            y_bin, label_sets, k_gt, *, precomputed_labels=None, runtime=None):
    if precomputed_labels is None:
        labels, runtime = cluster_umap_hdbscan(E_or_labels)
    else:
        labels = precomputed_labels
    soft, diag = labels_to_soft(labels)
    metrics = evaluate_on_653(soft, paper_ids, eval_ids, y_bin, label_sets, k_gt=k_gt)
    sizes = cluster_size_stats(labels)
    ndcg = metrics.get("nDCG@1_mean", metrics.get(f"nDCG@{k_gt}_mean"))
    row = {
        "encoder": name_enc,
        "representation": representation,
        "alpha": alpha if alpha is not None else "",
        "fit_n": 4488,
        "eval_n": 653,
        "Purity": metrics["Purity"],
        "LRAP": metrics["LRAP_mean"],
        "nDCG@1": ndcg,
        "V_measure": metrics["V_measure"],
        "PairF1": metrics["PairF1"],
        "n_clusters": diag["K"],
        "noise_fraction": diag["noise_frac"],
        "Coverage": metrics["Coverage"],
        "runtime_sec": runtime,
        **sizes,
    }
    CLUST.mkdir(parents=True, exist_ok=True)
    tag = f"{name_enc}__{representation}" + (f"__a{alpha}" if alpha is not None and alpha != "" else "")
    tag = tag.replace(".", "p")
    pd.DataFrame({"paper_id": paper_ids, "cluster": labels}).to_csv(CLUST / f"{tag}.csv", index=False)
    print(f"{name_enc} {representation} a={alpha}: K={row['n_clusters']} "
          f"noise={row['noise_fraction']:.3f} PF={row['PairF1']:.4f}")
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="specter2,semcse,qwen")
    args = ap.parse_args()

    for d in (OUT, RES, EMB, SIM, CLUST, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    papers, eval_ids = load_papers_and_eval()
    papers = papers.copy()
    papers["paper_id"] = papers.paper_id.astype(str)
    fit_ids = load_fit4488_ids()
    verify_corpus(fit_ids, eval_ids, papers)
    idx = indices_for(fit_ids, papers)

    # subset papers in ACC order
    papers_fit = papers.iloc[idx].reset_index(drop=True)
    assert papers_fit.paper_id.tolist() == fit_ids

    (OUT / "config.json").write_text(json.dumps({
        "regime": "B_4488",
        "fit_corpus": 4488,
        "fit_id_source": str(ACC_CLUSTERS),
        "eval": 653,
        "alphas": ALPHAS,
        "encoders": list(ENCODERS),
        "umap": base_cfg.UMAP_KW,
        "hdbscan": base_cfg.HDBSCAN_KW,
        "concat_template": "Title: {title} Abstract: {abstract}",
        "random_state": base_cfg.RANDOM_STATE,
        "a3": "UMAP metric=precomputed on D=1-S, then HDBSCAN",
        "note": "embeddings sliced from 6517 caches where available; clustering refit on 4488",
    }, indent=2, default=str))

    y_bin, label_sets, k_gt, _ = load_stop_for_eval(eval_ids, papers)
    titles = papers_fit.title.astype(str).tolist()
    abstracts = papers_fit.abstract.astype(str).tolist()
    concats = [f"Title: {t} Abstract: {a}" for t, a in zip(titles, abstracts)]
    assert papers_fit.title.astype(str).str.strip().ne("").all()
    assert papers_fit.abstract.astype(str).str.strip().ne("").all()

    rows = []
    encs = [e.strip() for e in args.encoders.split(",") if e.strip()]

    for enc in encs:
        print(f"\n===== {enc} fit4488 =====")
        Ea = load_or_slice_6517(enc, "abstract_R0", idx, abstracts)
        Et = load_or_slice_6517(enc, "title_T0", idx, titles)
        Ec = load_or_slice_6517(enc, "text_concat_A1", idx, concats)
        assert Ea.shape[0] == 4488 and Et.shape[0] == 4488 and Ec.shape[0] == 4488

        # R0 — actually recluster on 4488 (do NOT copy 6517 metrics)
        rows.append(run_one(enc, "abstract_R0", None, Ea if enc == "specter2" else _l2(Ea),
                            fit_ids, eval_ids, y_bin, label_sets, k_gt))
        rows.append(run_one(enc, "title_T0", None, Et if enc == "specter2" else _l2(Et),
                            fit_ids, eval_ids, y_bin, label_sets, k_gt))
        rows.append(run_one(enc, "text_concat_A1", None, Ec if enc == "specter2" else _l2(Ec),
                            fit_ids, eval_ids, y_bin, label_sets, k_gt))

        Et_n, Ea_n = _l2(Et), _l2(Ea)
        for a in ALPHAS:
            E = _l2(a * Et_n + (1.0 - a) * Ea_n)
            np.save(EMB / enc / f"embedding_weighted_A2_a{a}.npy", E)
            rows.append(run_one(enc, "embedding_weighted_A2", a, E,
                                fit_ids, eval_ids, y_bin, label_sets, k_gt))

        St_path = SIM / f"{enc}_S_title.npy"
        Sa_path = SIM / f"{enc}_S_abstract.npy"
        if St_path.exists():
            St = np.load(St_path)
            print(f"reuse {St_path.name}")
        else:
            St = cosine_sim_matrix(Et)
            np.save(St_path, St)
        if Sa_path.exists():
            Sa = np.load(Sa_path)
            print(f"reuse {Sa_path.name}")
        else:
            Sa = cosine_sim_matrix(Ea)
            np.save(Sa_path, Sa)
        assert np.allclose(St, St.T, atol=1e-4) and np.allclose(Sa, Sa.T, atol=1e-4)

        for a in ALPHAS:
            S = a * St + (1.0 - a) * Sa
            S = 0.5 * (S + S.T)
            np.fill_diagonal(S, 1.0)
            np.clip(S, -1.0, 1.0, out=S)
            D = (1.0 - S).astype(np.float64)
            np.fill_diagonal(D, 0.0)
            np.save(SIM / f"{enc}_S_late_a{a}.npy", S.astype(np.float32))
            labels, rt = cluster_umap_precomputed(D)
            rows.append(run_one(
                enc, "similarity_late_A3", a, None,
                fit_ids, eval_ids, y_bin, label_sets, k_gt,
                precomputed_labels=labels, runtime=rt,
            ))

        pd.DataFrame(rows).to_csv(RES / "fit4488_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "fit4488_results.csv", index=False)

    best_rows = []
    for enc in df.encoder.unique():
        for rep in df.representation.unique():
            sub = df[(df.encoder == enc) & (df.representation == rep)]
            if sub.empty:
                continue
            i = sub["PairF1"].astype(float).idxmax()
            best_rows.append(sub.loc[i])
    best = pd.DataFrame(best_rows).sort_values(["encoder", "PairF1"], ascending=[True, False])
    best.to_csv(RES / "fit4488_best_per_repr.csv", index=False)
    print("\n=== BEST PER ENCODER×REPR (fit4488) ===")
    print(best[["encoder", "representation", "alpha", "PairF1", "Purity", "n_clusters", "noise_fraction"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
