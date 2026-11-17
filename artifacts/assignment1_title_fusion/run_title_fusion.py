#!/usr/bin/env python3
"""Title–abstract fusion experiments. Reuses Assignment-1 pipeline."""
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
    evaluate_on_653,
    labels_to_soft,
    load_papers_and_eval,
    load_stop_for_eval,
    cluster_umap_hdbscan,
)
from pipeline.encoders import embed_qwen, embed_semcse, embed_specter2

OUT = ROOT / "artifacts" / "assignment1_title_fusion"
EMB = OUT / "caches" / "embeddings"
SIM = OUT / "caches" / "similarity"
RES = OUT / "results"
CLUST = OUT / "clusters"
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5]
ENCODERS = ("specter2", "semcse", "qwen")


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def verify_eval_ids(eval_ids: pd.DataFrame):
    assert len(eval_ids) == 653, len(eval_ids)
    assert eval_ids.paper_id.nunique() == 653
    assert eval_ids.paper_id.duplicated().sum() == 0
    official = pd.read_csv(base_cfg.EVAL653_CSV)
    assert set(eval_ids.paper_id) == set(official.paper_id)
    print("eval ID checks OK: n=653, no dups, matches official file")


def encode(encoder: str, texts: list[str], cache: Path) -> np.ndarray:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if encoder == "specter2":
        return embed_specter2(texts, cache, normalize=False)
    if encoder == "semcse":
        return embed_semcse(texts, cache)
    if encoder == "qwen":
        return embed_qwen(texts, cache)
    raise ValueError(encoder)


def get_abstract_emb(encoder: str, abstracts: list[str]) -> np.ndarray:
    """Reuse Assignment-1 R0 cache (do not recompute)."""
    src = base_cfg.EMB_DIR / encoder / "R0_full_abstract.npy"
    if not src.exists():
        raise FileNotFoundError(src)
    E = np.load(src)
    assert len(E) == len(abstracts), (E.shape, len(abstracts))
    # mirror into this experiment's cache tree for provenance
    dst = EMB / encoder / "abstract_R0.npy"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        np.save(dst, E)
        (dst.with_suffix(".meta.json")).write_text(json.dumps({
            "source": str(src), "n": len(E), "note": "reused R0 abstract embeddings",
        }, indent=2))
    print(f"reuse abstract R0 {encoder} {E.shape}")
    return E


def cosine_sim_matrix(E: np.ndarray, batch: int = 1024) -> np.ndarray:
    E = _l2(E)
    n = len(E)
    S = np.empty((n, n), dtype=np.float32)
    Et = E.T
    for i0 in range(0, n, batch):
        i1 = min(n, i0 + batch)
        S[i0:i1] = E[i0:i1] @ Et
    np.fill_diagonal(S, 1.0)
    # numerical symmetry
    S = 0.5 * (S + S.T)
    np.clip(S, -1.0, 1.0, out=S)
    return S


def cluster_umap_precomputed(D: np.ndarray, random_state: int = 42):
    """UMAP metric=precomputed on distances, then HDBSCAN on embedding."""
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
    # unify nDCG column
    ndcg = metrics.get("nDCG@1_mean", metrics.get(f"nDCG@{k_gt}_mean"))
    row = {
        "encoder": name_enc,
        "representation": representation,
        "alpha": alpha if alpha is not None else "",
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
    pd.DataFrame({"paper_id": paper_ids, "cluster": labels}).to_csv(
        CLUST / f"{tag}.csv", index=False
    )
    print(f"{name_enc} {representation} a={alpha}: K={row['n_clusters']} "
          f"noise={row['noise_fraction']:.3f} PF={row['PairF1']:.4f}")
    return row


def load_r0_reference_rows():
    """Keep existing R0 metrics unchanged from assignment1 stage_ab results."""
    src = ROOT / "artifacts/assignment1/results/stage_ab_results.csv"
    df = pd.read_csv(src)
    rows = []
    mapping = {
        "specter2|R0_full|umap_hdbscan": "specter2",
        "semcse|R0_full|umap_hdbscan": "semcse",
        "qwen|R0_full|umap_hdbscan": "qwen",
    }
    for method, enc in mapping.items():
        r = df[df.method == method]
        if r.empty:
            raise RuntimeError(f"missing R0 reference row for {method}")
        r = r.iloc[0]
        ndcg_col = "nDCG@1_mean" if "nDCG@1_mean" in r.index else None
        rows.append({
            "encoder": enc,
            "representation": "abstract_R0",
            "alpha": "",
            "Purity": r["Purity"],
            "LRAP": r["LRAP_mean"],
            "nDCG@1": r[ndcg_col] if ndcg_col else np.nan,
            "V_measure": r["V_measure"],
            "PairF1": r["PairF1"],
            "n_clusters": int(r["K"]),
            "noise_fraction": float(r["noise_frac"]) if "noise_frac" in r.index else np.nan,
            "Coverage": r["Coverage"],
            "runtime_sec": r.get("runtime_sec", np.nan),
            "largest": r.get("largest", np.nan),
            "median_size": r.get("median_size", np.nan),
            "mean_size": np.nan,
            "n_singletons": np.nan,
            "note": "copied unchanged from assignment1 stage_ab R0",
        })
    return rows


def smoke_test(papers: pd.DataFrame, n: int = 40):
    print("=== SMOKE ===")
    sub = papers.head(n).copy()
    titles = sub.title.astype(str).tolist()
    abstracts = sub.abstract.astype(str).tolist()
    concat = [f"Title: {t} Abstract: {a}" for t, a in zip(titles, abstracts)]
    assert all(c.startswith("Title:") for c in concat)
    Et = embed_specter2(titles, OUT / "caches" / "smoke_title.npy", normalize=False)
    Ea = embed_specter2(abstracts, OUT / "caches" / "smoke_abs.npy", normalize=False)
    Ec = embed_specter2(concat, OUT / "caches" / "smoke_concat.npy", normalize=False)
    assert Et.shape[0] == n and Ea.shape[0] == n and Ec.shape[0] == n
    Et_n, Ea_n = _l2(Et), _l2(Ea)
    fused = _l2(0.3 * Et_n + 0.7 * Ea_n)
    assert fused.shape == Et.shape
    S = cosine_sim_matrix(Et)
    assert S.shape == (n, n)
    assert np.allclose(S, S.T, atol=1e-5)
    assert np.allclose(np.diag(S), 1.0, atol=1e-5)
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)
    assert not np.isnan(D).any()
    print("smoke OK")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--encoders", default="specter2,semcse,qwen")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    EMB.mkdir(parents=True, exist_ok=True)
    SIM.mkdir(parents=True, exist_ok=True)

    papers, eval_ids = load_papers_and_eval()
    verify_eval_ids(eval_ids)
    assert papers.title.astype(str).str.strip().ne("").all()
    assert papers.abstract.astype(str).str.strip().ne("").all()

    (OUT / "config.json").write_text(json.dumps({
        "fit_corpus": 6517,
        "eval": 653,
        "alphas": ALPHAS,
        "encoders": list(ENCODERS),
        "umap": base_cfg.UMAP_KW,
        "hdbscan": base_cfg.HDBSCAN_KW,
        "concat_template": "Title: {title} Abstract: {abstract}",
        "random_state": base_cfg.RANDOM_STATE,
        "a3": "UMAP metric=precomputed on D=1-S, then HDBSCAN",
    }, indent=2, default=str))

    smoke_test(papers)
    if args.smoke_only:
        return

    y_bin, label_sets, k_gt, _ = load_stop_for_eval(eval_ids, papers)
    paper_ids = papers.paper_id.tolist()
    titles = papers.title.astype(str).tolist()
    abstracts = papers.abstract.astype(str).tolist()
    concats = [f"Title: {t} Abstract: {a}" for t, a in zip(titles, abstracts)]

    # token-length audit for A1 (approx whitespace + note SPECTER2 max 512)
    tok_lens = [len(c.split()) for c in concats]
    title_lens = [len(t.split()) for t in titles]
    pd.DataFrame({
        "paper_id": paper_ids,
        "title_ws_tokens": title_lens,
        "concat_ws_tokens": tok_lens,
        "title_likely_kept_if_truncate_512": [True] * len(titles),  # titles first in string
    }).to_csv(RES / "a1_length_audit.csv", index=False)
    print(f"A1 concat ws-tokens: mean={np.mean(tok_lens):.1f} max={max(tok_lens)} "
          f"(title first → not systematically truncated)")

    rows = load_r0_reference_rows()
    encs = [e.strip() for e in args.encoders.split(",") if e.strip()]

    for enc in encs:
        print(f"\n===== {enc} =====")
        Ea = get_abstract_emb(enc, abstracts)
        Et = encode(enc, titles, EMB / enc / "title_T0.npy")
        Ec = encode(enc, concats, EMB / enc / "text_concat_A1.npy")

        # T0
        rows.append(run_one(enc, "title_T0", None, Et if enc == "specter2" else _l2(Et),
                            paper_ids, eval_ids, y_bin, label_sets, k_gt))
        # A1
        rows.append(run_one(enc, "text_concat_A1", None, Ec if enc == "specter2" else _l2(Ec),
                            paper_ids, eval_ids, y_bin, label_sets, k_gt))

        # A2 embedding weighted
        Et_n, Ea_n = _l2(Et), _l2(Ea)
        for a in ALPHAS:
            E = _l2(a * Et_n + (1.0 - a) * Ea_n)
            np.save(EMB / enc / f"embedding_weighted_A2_a{a}.npy", E)
            rows.append(run_one(enc, "embedding_weighted_A2", a, E,
                                paper_ids, eval_ids, y_bin, label_sets, k_gt))

        # A3 late similarity fusion
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
        assert not np.isnan(St).any() and not np.isnan(Sa).any()

        for a in ALPHAS:
            S = a * St + (1.0 - a) * Sa
            S = 0.5 * (S + S.T)
            np.fill_diagonal(S, 1.0)
            np.clip(S, -1.0, 1.0, out=S)
            D = (1.0 - S).astype(np.float64)
            np.fill_diagonal(D, 0.0)
            assert np.allclose(D, D.T, atol=1e-5)
            np.save(SIM / f"{enc}_S_late_a{a}.npy", S.astype(np.float32))
            labels, rt = cluster_umap_precomputed(D)
            rows.append(run_one(
                enc, "similarity_late_A3", a, None,
                paper_ids, eval_ids, y_bin, label_sets, k_gt,
                precomputed_labels=labels, runtime=rt,
            ))

        pd.DataFrame(rows).to_csv(RES / "title_fusion_results.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(RES / "title_fusion_results.csv", index=False)

    # compact best per encoder × representation
    best_rows = []
    for enc in df.encoder.unique():
        for rep in df.representation.unique():
            sub = df[(df.encoder == enc) & (df.representation == rep)]
            if sub.empty:
                continue
            # pick best PairF1; for R0/T0/A1 only one row
            i = sub["PairF1"].astype(float).idxmax()
            best_rows.append(sub.loc[i])
    best = pd.DataFrame(best_rows).sort_values(["encoder", "PairF1"], ascending=[True, False])
    best.to_csv(RES / "title_fusion_best_per_repr.csv", index=False)
    print("\n=== BEST PER ENCODER×REPR ===")
    print(best[["encoder", "representation", "alpha", "PairF1", "Purity", "n_clusters", "noise_fraction"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
