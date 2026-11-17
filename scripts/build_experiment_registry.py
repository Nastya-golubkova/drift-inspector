#!/usr/bin/env python3
"""Build artifacts/assignment1/experiment_registry.csv from existing result CSVs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "assignment1" / "experiment_registry.csv"

COLS = [
    "experiment_id", "regime", "role", "fit_corpus", "fit_n", "eval_n", "eval_id_set",
    "encoder", "checkpoint", "representation", "text_sections", "fusion_type", "alpha",
    "window_strategy", "clustering_algorithm", "clustering_parameters",
    "purity", "LRAP", "nDCG_at_1", "V_measure", "PairF1",
    "n_clusters", "noise_fraction", "random_seed", "exploratory_or_fixed",
    "source_artifact", "notes",
]

ENC_CKPT = {
    "specter2": "allenai/specter2_aug2023refresh",
    "semcse": "CLAUSE-Bielefeld/SemCSE_cosine",
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "": "",
}


def _row(**kw):
    base = {c: "" for c in COLS}
    base.update(kw)
    return base


def from_final_comparison():
    path = ROOT / "artifacts/assignment1/results/final_comparison_table.csv"
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        role = r["role"]
        method = str(r["method"])
        # ACC published/reproduced historically fit on 4488; our methods in this file are 6517
        if role in ("Published", "Reproduced"):
            fit_n = 4488 if "ACC" in method else ""
            regime = "A_6517"  # Table 2 presentation; ACC itself is 4488 study corpus
            fit_corpus = "ACC_study_or_original_pipeline"
            if "ACC" in method:
                fit_corpus = "acc_clusters_emnlp_4488"
                regime = "B_4488" if role == "Reproduced" else "published_table2"
            else:
                regime = "published_table2"
            exploratory = "fixed"
        else:
            fit_n = 6517
            regime = "A_6517"
            fit_corpus = "emnlp_full"
            exploratory = "fixed"
        rows.append(_row(
            experiment_id=f"final_cmp_{i}_{role}_{method}"[:120],
            regime=regime,
            role=role,
            fit_corpus=fit_corpus,
            fit_n=fit_n if fit_n != "" else (4488 if "ACC" in method else ""),
            eval_n=653,
            eval_id_set="author_table2_eval_papers",
            encoder="",
            checkpoint="",
            representation=method,
            text_sections="",
            fusion_type="",
            alpha="",
            window_strategy="",
            clustering_algorithm=method,
            clustering_parameters="",
            purity=r.get("Purity", ""),
            LRAP=r.get("LRAP_mean", r.get("LRAP", "")),
            nDCG_at_1=r.get("nDCG@1_mean", r.get("nDCG@1", "")),
            V_measure=r.get("V_measure", ""),
            PairF1=r.get("PairF1", ""),
            n_clusters=r.get("K", ""),
            noise_fraction="",
            random_seed=42,
            exploratory_or_fixed=exploratory,
            source_artifact=str(path.relative_to(ROOT)),
            notes="from final_comparison_table; do not conflate roles",
        ))
    return rows


def from_stage_ab():
    path = ROOT / "artifacts/assignment1/results/stage_ab_results.csv"
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        method = str(r["method"])
        parts = method.split("|")
        enc = parts[0] if parts else ""
        rep = parts[1] if len(parts) > 1 else method
        rows.append(_row(
            experiment_id=f"A_stage_ab_{method}",
            regime="A_6517",
            role="Our method",
            fit_corpus="emnlp_full",
            fit_n=6517,
            eval_n=653,
            eval_id_set="author_table2_eval_papers",
            encoder=enc,
            checkpoint=ENC_CKPT.get(enc, ""),
            representation=rep,
            text_sections="abstract",
            fusion_type="none",
            alpha="",
            window_strategy=rep,
            clustering_algorithm="umap_hdbscan",
            clustering_parameters="umap_nn40_nc5_mcs25_ms5",
            purity=r["Purity"],
            LRAP=r["LRAP_mean"],
            nDCG_at_1=r.get("nDCG@1_mean", ""),
            V_measure=r["V_measure"],
            PairF1=r["PairF1"],
            n_clusters=r["K"],
            noise_fraction=r.get("noise_frac", ""),
            random_seed=42,
            exploratory_or_fixed="fixed",
            source_artifact=str(path.relative_to(ROOT)),
            notes="",
        ))
    return rows


def from_stage_c():
    path = ROOT / "artifacts/assignment1/results/stage_c_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        method = str(r["method"])
        enc = method.split("|")[0] if "|" in method else ""
        rows.append(_row(
            experiment_id=f"A_stage_c_{method}",
            regime="A_6517",
            role="Our method",
            fit_corpus="emnlp_full",
            fit_n=6517,
            eval_n=653,
            eval_id_set="author_table2_eval_papers",
            encoder=enc,
            checkpoint=ENC_CKPT.get(enc, ""),
            representation="late_maxsim",
            text_sections="abstract_windows",
            fusion_type="late_maxsim",
            alpha="",
            window_strategy="maxsim",
            clustering_algorithm=method.split("|")[-1] if "|" in method else "",
            clustering_parameters="",
            purity=r["Purity"],
            LRAP=r["LRAP_mean"],
            nDCG_at_1=r.get("nDCG@1_mean", ""),
            V_measure=r["V_measure"],
            PairF1=r["PairF1"],
            n_clusters=r["K"],
            noise_fraction=r.get("noise_frac", ""),
            random_seed=42,
            exploratory_or_fixed="fixed",
            source_artifact=str(path.relative_to(ROOT)),
            notes="",
        ))
    return rows


def from_title_fusion():
    path = ROOT / "artifacts/assignment1_title_fusion/results/title_fusion_results.csv"
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        enc = r["encoder"]
        rep = r["representation"]
        alpha = r.get("alpha", "")
        if pd.isna(alpha) or alpha == "":
            alpha = ""
            exploratory = "fixed"
        else:
            exploratory = "exploratory"  # alpha grid reported in full; best-α is exploratory
        note = r.get("note", "") if "note" in r.index else ""
        if isinstance(note, float) and pd.isna(note):
            note = ""
        rows.append(_row(
            experiment_id=f"A_tf_{enc}_{rep}_a{alpha}",
            regime="A_6517",
            role="Our method",
            fit_corpus="emnlp_full",
            fit_n=6517,
            eval_n=653,
            eval_id_set="author_table2_eval_papers",
            encoder=enc,
            checkpoint=ENC_CKPT.get(enc, ""),
            representation=rep,
            text_sections="title+abstract" if "title" in str(rep) or "A" in str(rep) else "abstract",
            fusion_type=rep,
            alpha=alpha,
            window_strategy="none",
            clustering_algorithm="umap_hdbscan" if "late" not in str(rep) else "umap_precomputed_hdbscan",
            clustering_parameters="umap_nn40_nc5_mcs25_ms5",
            purity=r["Purity"],
            LRAP=r["LRAP"],
            nDCG_at_1=r["nDCG@1"],
            V_measure=r["V_measure"],
            PairF1=r["PairF1"],
            n_clusters=r["n_clusters"],
            noise_fraction=r["noise_fraction"],
            random_seed=42,
            exploratory_or_fixed=exploratory,
            source_artifact=str(path.relative_to(ROOT)),
            notes=str(note),
        ))
    return rows


def from_fit4488():
    path = ROOT / "artifacts/assignment1_fit4488/results/fit4488_results.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        enc = r["encoder"]
        rep = r["representation"]
        alpha = r.get("alpha", "")
        if pd.isna(alpha) or alpha == "":
            alpha = ""
            exploratory = "fixed"
        else:
            exploratory = "exploratory"
        rows.append(_row(
            experiment_id=f"B_fit4488_{enc}_{rep}_a{alpha}",
            regime="B_4488",
            role="Our method",
            fit_corpus="acc_clusters_emnlp_4488",
            fit_n=4488,
            eval_n=653,
            eval_id_set="author_table2_eval_papers",
            encoder=enc,
            checkpoint=ENC_CKPT.get(enc, ""),
            representation=rep,
            text_sections="title+abstract",
            fusion_type=rep,
            alpha=alpha,
            window_strategy="none",
            clustering_algorithm="umap_hdbscan",
            clustering_parameters="umap_nn40_nc5_mcs25_ms5",
            purity=r["Purity"],
            LRAP=r["LRAP"],
            nDCG_at_1=r["nDCG@1"],
            V_measure=r["V_measure"],
            PairF1=r["PairF1"],
            n_clusters=r["n_clusters"],
            noise_fraction=r["noise_fraction"],
            random_seed=42,
            exploratory_or_fixed=exploratory,
            source_artifact=str(path.relative_to(ROOT)),
            notes="refit clustering on ACC 4488 IDs; embeddings sliced from 6517 caches",
        ))
    return rows


def from_restricted(path: Path, fit_n: int, tag: str):
    if not path.exists():
        return []
    df = pd.read_csv(path)
    # normalize columns
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("pairf1", "pair_f1"):
            colmap["PairF1"] = c
        elif cl == "purity":
            colmap["Purity"] = c
        elif cl in ("lrap", "lrap_mean"):
            colmap["LRAP"] = c
        elif "ndcg" in cl:
            colmap["nDCG"] = c
        elif cl in ("v_measure", "vmeasure"):
            colmap["V"] = c
        elif cl in ("k", "n_clusters"):
            colmap["K"] = c
        elif "noise" in cl:
            colmap["noise"] = c
        elif cl in ("method", "config", "name"):
            colmap["method"] = c
    if "method" not in colmap or "PairF1" not in colmap:
        return []
    rows = []
    for i, r in df.iterrows():
        method = str(r[colmap["method"]])
        exploratory = "exploratory"
        rows.append(_row(
            experiment_id=f"C_{tag}_{method}"[:120],
            regime="C_restricted",
            role="Our method" if method.upper().startswith("OURS") or method.startswith(("A_", "B", "C_", "D_", "E_", "F_")) else (
                "Reproduced" if method in ("ACC", "LDA", "NMF", "BERTopic-SPECTER2", "SentSPECTER") or "SPECTER2" in method else "Our method"
            ),
            fit_corpus=f"restricted_{fit_n}",
            fit_n=fit_n,
            eval_n=fit_n,
            eval_id_set=f"restricted_{fit_n}",
            encoder="specter2" if "SPECTER" in method.upper() or "OURS" in method.upper() or method.startswith(("A_", "B", "D_", "E_", "F_")) else "",
            checkpoint=ENC_CKPT.get("specter2", "") if "SPECTER" in method.upper() or "OURS" in method.upper() else "",
            representation=method,
            text_sections=method,
            fusion_type="",
            alpha="",
            window_strategy="",
            clustering_algorithm="see_source",
            clustering_parameters="often_umap_nn15_hdbscan_mcs15_for_restricted",
            purity=r[colmap["Purity"]] if "Purity" in colmap else "",
            LRAP=r[colmap["LRAP"]] if "LRAP" in colmap else "",
            nDCG_at_1=r[colmap["nDCG"]] if "nDCG" in colmap else "",
            V_measure=r[colmap["V"]] if "V" in colmap else "",
            PairF1=r[colmap["PairF1"]],
            n_clusters=r[colmap["K"]] if "K" in colmap else "",
            noise_fraction=r[colmap["noise"]] if "noise" in colmap else "",
            random_seed="",
            exploratory_or_fixed=exploratory,
            source_artifact=str(path.relative_to(ROOT)),
            notes=f"fit=eval={fit_n}; not comparable to Regime A/B without labeling",
        ))
    return rows


def main():
    rows = []
    rows += from_final_comparison()
    rows += from_stage_ab()
    rows += from_stage_c()
    rows += from_title_fusion()
    rows += from_fit4488()
    rows += from_restricted(ROOT / "artifacts/extension_653/fair653_all_methods.csv", 653, "fair653")
    rows += from_restricted(ROOT / "artifacts/extension_full656/fair656_all_methods.csv", 656, "fair656")
    rows += from_restricted(ROOT / "artifacts/extension/results.csv", 612, "ext612")
    rows += from_restricted(ROOT / "artifacts/intersection/baseline_on_intersection.csv", 612, "inter612")
    rows += from_restricted(ROOT / "artifacts/extension_expanded/results_expanded.csv", 645, "exp645")
    rows += from_restricted(ROOT / "artifacts/extension_sliding/results_full.csv", 612, "slide612")

    df = pd.DataFrame(rows, columns=COLS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT} rows={len(df)}")
    print(df.groupby(["regime", "role"]).size().to_string())


if __name__ == "__main__":
    main()
