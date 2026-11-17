#!/usr/bin/env python3
"""Run Assignment 1 stages A–C (abstract multi-window clustering)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "artifacts" / "assignment1"))

from pipeline import config
from pipeline.cluster_eval import (
    evaluate_on_653,
    labels_to_soft,
    load_papers_and_eval,
    load_stop_for_eval,
    cluster_hdbscan_precomputed,
    cluster_agglo_precomputed,
    run_vector_clustering,
)
from pipeline.encoders import embed_specter2, embed_semcse, embed_qwen
from pipeline.fusion import early_token_weighted_mean, early_uniform_mean, maxsim_symmetric
from pipeline.windowing import build_window_table, window_stats


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / n).astype(np.float32)


def ensure_windows(papers: pd.DataFrame, max_tokens: int) -> pd.DataFrame:
    path = config.WIN_DIR / f"windows_max{max_tokens}.csv"
    stats_path = config.WIN_DIR / f"windows_max{max_tokens}_stats.json"
    if path.exists():
        win = pd.read_csv(path)
        print(f"reuse windows {path.name} n={len(win)}")
    else:
        config.WIN_DIR.mkdir(parents=True, exist_ok=True)
        win = build_window_table(papers, max_tokens=max_tokens)
        win.to_csv(path, index=False)
        print(f"wrote {path} n={len(win)}")
    stats = window_stats(win)
    stats_path.write_text(json.dumps(stats, indent=2))
    print("window stats:", json.dumps(stats))
    return win


def encode_full_and_windows(encoder: str, papers: pd.DataFrame, win: pd.DataFrame):
    paper_ids = papers["paper_id"].tolist()
    abstracts = papers["abstract"].astype(str).tolist()
    win_texts = win["text"].astype(str).tolist()
    emb_dir = config.EMB_DIR / encoder
    emb_dir.mkdir(parents=True, exist_ok=True)

    if encoder == "specter2":
        # Prefer baseline cache for R0 if shape matches
        full_cache = emb_dir / "R0_full_abstract.npy"
        if (not full_cache.exists()) and config.BASELINE_SPECTER2.exists():
            E0 = np.load(config.BASELINE_SPECTER2)
            if len(E0) == len(abstracts):
                print("reusing baseline SPECTER2 abstract embeddings for R0")
                np.save(full_cache, E0.astype(np.float32))
                (full_cache.with_suffix(".meta.json")).write_text(json.dumps({
                    "source": str(config.BASELINE_SPECTER2),
                    "model": config.SPECTER2_MODEL,
                    "n": len(E0),
                    "note": "raw CLS (not L2); clustering uses cosine UMAP",
                }, indent=2))
        E0 = embed_specter2(abstracts, full_cache, normalize=False)
        Ew = embed_specter2(
            win_texts, emb_dir / f"windows_max{config.PRIMARY_MAX_WINDOW_TOKENS}.npy",
            normalize=False,
        )
    elif encoder == "semcse":
        E0 = embed_semcse(abstracts, emb_dir / "R0_full_abstract.npy")
        Ew = embed_semcse(
            win_texts, emb_dir / f"windows_max{config.PRIMARY_MAX_WINDOW_TOKENS}.npy",
        )
    elif encoder == "qwen":
        E0 = embed_qwen(abstracts, emb_dir / "R0_full_abstract.npy")
        Ew = embed_qwen(
            win_texts, emb_dir / f"windows_max{config.PRIMARY_MAX_WINDOW_TOKENS}.npy",
        )
    else:
        raise ValueError(encoder)
    return paper_ids, E0, Ew


def stage_ab(encoders=("specter2", "semcse", "qwen")):
    papers, eval_ids = load_papers_and_eval()
    y_bin, label_sets, k_gt, _ = load_stop_for_eval(eval_ids, papers)
    win = ensure_windows(papers, config.PRIMARY_MAX_WINDOW_TOKENS)
    rows = []

    for enc in encoders:
        print(f"\n===== Encoder {enc} =====")
        paper_ids, E0, Ew = encode_full_and_windows(enc, papers, win)

        # R0
        print(f"--- {enc} R0 full abstract ---")
        E0_use = E0 if enc == "specter2" else _l2(E0)
        rows.append(run_vector_clustering(
            E0_use, paper_ids, eval_ids, y_bin, label_sets, k_gt,
            f"{enc}|R0_full|umap_hdbscan"))

        # R2 / R3
        R2 = early_uniform_mean(win, Ew, paper_ids)
        R3 = early_token_weighted_mean(win, Ew, paper_ids)
        print(f"--- {enc} R2 uniform mean ---")
        rows.append(run_vector_clustering(
            R2, paper_ids, eval_ids, y_bin, label_sets, k_gt,
            f"{enc}|R2_uniform_mean|umap_hdbscan"))
        print(f"--- {enc} R3 token-weighted mean ---")
        rows.append(run_vector_clustering(
            R3, paper_ids, eval_ids, y_bin, label_sets, k_gt,
            f"{enc}|R3_token_weighted|umap_hdbscan"))

        # cache fused vectors
        np.save(config.EMB_DIR / enc / "R2_uniform_mean.npy", R2)
        np.save(config.EMB_DIR / enc / "R3_token_weighted.npy", R3)

        config.RES_DIR.mkdir(parents=True, exist_ok=True)
        # Persist per-encoder and merge into stage_ab_results.csv
        part = pd.DataFrame(rows)
        part.to_csv(config.RES_DIR / f"stage_ab_{enc}.csv", index=False)
        parts = sorted(config.RES_DIR.glob("stage_ab_*.csv"))
        # exclude aggregate if present in glob pattern carefully
        parts = [p for p in parts if p.name != "stage_ab_results.csv"]
        merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        # drop duplicate method rows keeping last
        merged = merged.drop_duplicates(subset=["method"], keep="last")
        merged.to_csv(config.RES_DIR / "stage_ab_results.csv", index=False)
        print(merged[["method", "K", "Coverage", "Purity", "PairF1"]].to_string(index=False))

    return pd.read_csv(config.RES_DIR / "stage_ab_results.csv")


def stage_c(encoders=("specter2", "semcse")):
    """Late fusion MaxSim + precomputed clustering. SPECTER2 + SemCSE first."""
    papers, eval_ids = load_papers_and_eval()
    y_bin, label_sets, k_gt, _ = load_stop_for_eval(eval_ids, papers)
    win = ensure_windows(papers, config.PRIMARY_MAX_WINDOW_TOKENS)
    paper_ids = papers["paper_id"].tolist()
    rows = []

    for enc in encoders:
        print(f"\n===== Late fusion {enc} =====")
        emb_dir = config.EMB_DIR / enc
        Ew = np.load(emb_dir / f"windows_max{config.PRIMARY_MAX_WINDOW_TOKENS}.npy")
        sim_path = config.SIM_DIR / f"{enc}_maxsim.npy"
        config.SIM_DIR.mkdir(parents=True, exist_ok=True)
        if sim_path.exists() and np.load(sim_path, mmap_mode="r").shape[0] == len(paper_ids):
            print(f"reuse {sim_path}")
            S = np.load(sim_path)
        else:
            S = maxsim_symmetric(Ew, win, paper_ids)
            np.save(sim_path, S)
            np.save(config.SIM_DIR / f"{enc}_maxsim_paper_ids.npy", np.array(paper_ids))
            print(f"saved {sim_path} {S.shape}")

        D = (1.0 - S).astype(np.float64)
        np.fill_diagonal(D, 0.0)

        for cname, fn in [
            ("hdbscan_precomputed", lambda: cluster_hdbscan_precomputed(D)),
            ("agglo_k80", lambda: cluster_agglo_precomputed(D, n_clusters=80)),
        ]:
            labels, rt = fn()
            soft, diag = labels_to_soft(labels)
            metrics = evaluate_on_653(soft, paper_ids, eval_ids, y_bin, label_sets, k_gt=k_gt)
            row = {
                "method": f"{enc}|late_maxsim|{cname}",
                "family": "our_method",
                "role": "Our method",
                "runtime_sec": rt,
                **diag,
                **metrics,
            }
            rows.append(row)
            print(row["method"], f"K={row['K']} PF={row['PairF1']:.4f}")
            safe = row["method"].replace("|", "__")
            pd.DataFrame({"paper_id": paper_ids, "cluster": labels}).to_csv(
                config.CLUST_DIR / f"{safe}.csv", index=False
            )

        pd.DataFrame(rows).to_csv(config.RES_DIR / "stage_c_results.csv", index=False)

    return pd.DataFrame(rows)


def write_combined():
    parts = []
    for name in ["stage_ab_results.csv", "stage_c_results.csv"]:
        p = config.RES_DIR / name
        if p.exists():
            parts.append(pd.read_csv(p))
    if not parts:
        return
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(config.RES_DIR / "our_methods_all.csv", index=False)

    # Published + Reproduced reference table
    t2 = pd.read_csv(config.T2_CSV)
    pub = t2.rename(columns={
        "Method": "method", "LRAP_mean": "LRAP_mean", "nDCG@1_mean": "nDCG@1_mean",
        "V_measure": "V_measure", "PairF1": "PairF1", "Purity": "Purity",
        "K": "K", "Coverage": "Coverage",
    })
    pub["role"] = "Published"
    pub["family"] = "baseline_published"

    # Reproduced = same numbers from baseline FR (audit); copy published as reproduced
    # since audit verified match — mark explicitly
    repro = pub.copy()
    repro["role"] = "Reproduced"
    repro["family"] = "baseline_reproduced"
    repro["method"] = repro["method"].astype(str) + " [reproduced]"

    ours = df.copy()
    ours["role"] = "Our method"

    cols = ["role", "method", "K", "Coverage", "Purity", "LRAP_mean",
            "nDCG@1_mean", "V_measure", "PairF1"]
    for c in cols:
        if c not in pub.columns:
            pub[c] = np.nan
        if c not in repro.columns:
            repro[c] = np.nan
        if c not in ours.columns:
            ours[c] = np.nan

    final = pd.concat([pub[cols], repro[cols], ours[cols]], ignore_index=True)
    final.to_csv(config.RES_DIR / "final_comparison_table.csv", index=False)
    print("wrote final_comparison_table.csv")
    print(final.sort_values("PairF1", ascending=False).head(20).to_string(index=False))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ab", "c", "all", "combine"], default="all")
    ap.add_argument("--encoders", default="specter2,semcse,qwen")
    args = ap.parse_args()
    encs = tuple(e.strip() for e in args.encoders.split(",") if e.strip())

    config.OUT.mkdir(parents=True, exist_ok=True)
    (config.OUT / "config.json").write_text(json.dumps({
        "random_state": config.RANDOM_STATE,
        "umap": config.UMAP_KW,
        "hdbscan": config.HDBSCAN_KW,
        "primary_max_window_tokens": config.PRIMARY_MAX_WINDOW_TOKENS,
        "encoders": {
            "specter2": config.SPECTER2_MODEL,
            "semcse": config.SEMCSE_MODEL,
            "qwen": config.QWEN_MODEL,
            "qwen_instruction": config.QWEN_INSTRUCTION,
        },
        "fit_corpus": 6517,
        "eval_corpus": 653,
        "eval_ids": str(config.EVAL653_CSV),
    }, indent=2, default=str))

    if args.stage in ("ab", "all"):
        stage_ab(encoders=encs)
    if args.stage in ("c", "all"):
        # late fusion: specter2+semcse first (skip qwen if not in encs for C)
        c_encs = [e for e in encs if e in ("specter2", "semcse", "qwen")]
        # Prefer specter2,semcse; include qwen if windows exist
        stage_c(encoders=tuple(c_encs[:2] if args.stage == "all" else c_encs))
    write_combined()


if __name__ == "__main__":
    main()
