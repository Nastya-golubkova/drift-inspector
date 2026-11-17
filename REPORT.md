# Assignment 1 — Research Paper Clustering: Final Report

## Authorship labels (read first)

| Label | Meaning |
|---|---|
| **Published** | Numbers from the Drift Inspector paper Table 2 (LDA, NMF, BERTopic, SentSPECTER, **ACC**, ACC-top30). Never called “ours”. |
| **Ours** | Methods implemented for this assignment: abstract windows (R0/R2/R3), title–abstract fusion (T0/A1/A2/A3), and the ACC-matched fit-4488 refit. |
| **Prior extension** | Restricted-fit / full-text experiments already stored under `artifacts/extension_* (e.g. abs+concl on 653). Not paper methods; historical CSV names may say `OURS_*` — treat that as a filename, not authorship. |

Method map & reproduce commands: [`artifacts/assignment1/README.md`](artifacts/assignment1/README.md).

## 1. Abstract

We evaluate paper-level clustering against SToP topics on the official **653** papers under three fit regimes: full EMNLP (**6517**), ACC study paper IDs (**4488**), and restricted subsets (**612–656**).

**Ours (Regime A, 6517):** SPECTER2 title+abstract concatenation (**A1**) reaches PairF1 **0.549** (fixed). Late fusion (**A3**, α=0.1, α chosen on the 653 labels) reaches **0.557** (exploratory). Both exceed the published Table 2 PairF1 maximum (ACC-top30 = **0.403**).

**Ours (Regime B, 4488):** SPECTER2 abstract-only (**R0**) reaches PairF1 **0.512** (+0.109 vs published ACC-top30). Title-fusion gains from Regime A do **not** transfer (A1=0.425, A3 α=0.1=0.435). ACC clusters **claims**; we cluster **paper** embeddings — matched paper IDs are not an identical pipeline.

**Prior extension (Regime C, fit=eval=653):** abstract+conclusion concat PairF1 **0.508** (`fair653_all_methods.csv`). The ≈0.521 abs+intro+concl figure is **fit=656**, not 653.

## 2. Task and protocol

Recover SToP multi-label structure via clustering. Metrics: Purity, LRAP, nDCG@1, V-measure, PairF1.

| Resource | Path / size |
|---|---|
| Full papers | `data/claims_sources/emnlp_full/papers.csv` — 6517 |
| ACC study papers | `artifacts/baseline/acc_clusters_emnlp.csv` unique IDs — **4488** |
| Eval IDs | `artifacts/extension_full656/author_table2_eval_papers.csv` — **653** |

**Ours A/B clustering:** UMAP (`n_neighbors=40`, `n_components=5`, `min_dist=0`) → HDBSCAN (`min_cluster_size=25`, `min_samples=5`), `random_state=42`. A3: UMAP `metric=precomputed` on `D=1-S`.  
Eval ⊂ 4488: intersection **653**.

**Strict subset checks** (`assert eval_ids.issubset(fit_ids)`), both PASS:

| Regime | Fit N | Eval N | ∩ | Artifact |
|---|---:|---:|---:|---|
| A | 6517 | 653 | 653 | `artifacts/assignment1/audit/eval_subset_of_fit_verification.csv` |
| B | 4488 | 653 | 653 | same |

## 3. Published Table 2 vs project reproduction

### Table 1 — Original published (paper)

Source: `artifacts/assignment1/results/table2_published_original.csv`.

| Method | Purity | LRAP | nDCG@1 | V-measure | PairF1 |
|---|---:|---:|---:|---:|---:|
| LDA | 0.245 | 0.508 | 0.329 | 0.278 | 0.170 |
| NMF | 0.294 | 0.585 | 0.445 | 0.324 | 0.186 |
| BERTopic SPECTER2 | 0.324 | 0.651 | 0.515 | 0.469 | 0.289 |
| BERTopic MPNet | 0.327 | 0.651 | 0.509 | 0.457 | 0.234 |
| SentSPECTER | 0.654 | 0.668 | 0.540 | 0.532 | 0.268 |
| SentSPECTER top30 | 0.472 | 0.637 | 0.512 | 0.471 | 0.365 |
| ACC | 0.689 | 0.674 | 0.580 | 0.571 | 0.361 |
| ACC top30 | 0.618 | 0.694 | 0.596 | 0.552 | **0.403** |

### Table 2 — Reproduction export in this repo

Source: `artifacts/baseline/stop_alignment_table.csv` → `table2_reproduced_from_artifacts.csv`.  
Largest gap vs paper: BERTopic MPNet PairF1 **0.249** vs **0.234**. Details: `artifacts/assignment1/audit/RESULTS_AUDIT.md`.

Do not treat `final_comparison_table.csv` `role=Published` as paper glyphs — that file historically copied the reproduction export into both roles.

## 4. Ours — method definitions

| ID | Definition |
|---|---|
| R0 | Abstract-only embedding |
| R2 / R3 | Mean / token-weighted mean of abstract sliding windows |
| T0 | Title-only |
| A1 | `Title: … Abstract: …` then one encode (no α) |
| A2 | `normalize(α e_title + (1-α) e_abstract)` |
| A3 | `S = α S_title + (1-α) S_abstract`, UMAP on `D=1-S` |

These are **not** reproductions of BERTopic-SPECTER2 or ACC.

## 5. Regimes

| Regime | Fit | Eval | Content |
|---|---:|---:|---|
| A | 6517 | 653 | Ours: windows + title fusion |
| B | 4488 | 653 | Ours: same fusion, ACC paper IDs |
| C | 612–656 | = fit | Prior extension + restricted baselines |

Canonical CSV: `artifacts/assignment1/results/canonical_comparison_table.csv`.

## 6. Ours — Regime A (6517 → 653)

Source: `artifacts/assignment1_title_fusion/results/title_fusion_results.csv` (R0 also in `stage_ab_results.csv`).

| Method | Purity | LRAP | nDCG@1 | V | PairF1 | Status |
|---|---:|---:|---:|---:|---:|---|
| SPECTER2 R0 | 0.666 | 0.695 | 0.632 | 0.630 | 0.525 | Fixed (ours) |
| SPECTER2 A1 | 0.729 | 0.729 | 0.665 | 0.655 | **0.549** | Fixed (ours) |
| SPECTER2 A3 α=0.1 | 0.695 | 0.733 | 0.666 | 0.668 | **0.557** | Exploratory α (ours) |
| SPECTER2 T0 | 0.679 | 0.658 | 0.597 | 0.580 | 0.413 | Fixed (ours) |

Among R0/R2/R3, SPECTER2 R0 remains strongest for SPECTER2; windows help SemCSE more (`stage_ab_results.csv`).

## 7. Ours — Regime B (4488 → 653)

Source: `artifacts/assignment1_fit4488/results/fit4488_results.csv`.

| Method | Purity | LRAP | nDCG@1 | V | PairF1 | Status |
|---|---:|---:|---:|---:|---:|---|
| SPECTER2 R0 | 0.620 | 0.708 | 0.638 | 0.605 | **0.512** | Fixed (ours) |
| SPECTER2 A1 | 0.668 | 0.711 | 0.627 | 0.615 | 0.425 | Fixed (ours) |
| SPECTER2 A3 α=0.1 | 0.647 | 0.698 | 0.614 | 0.617 | 0.435 | Exploratory (ours) |
| Published ACC top30 | 0.618 | 0.694 | 0.596 | 0.552 | 0.403 | Published |
| Published ACC | 0.689 | 0.674 | 0.580 | 0.571 | 0.361 | Published |

R0 vs published ACC-top30: +0.109 PairF1 (~27% relative); also higher on all five metrics. vs full ACC: wins PairF1/LRAP/nDCG/V, **loses Purity**. Title fusion does not help on 4488 under this UMAP+HDBSCAN setup.

## 8. Prior extension — Regime C (sections & windows)

### 8.1 Exact 653: abstract + conclusion and related

Source: `artifacts/extension_653/fair653_all_methods.csv` (fit=eval=**653**).  
Clustering config typically differs from Ours A/B (often UMAP nn≈15).

| Method (CSV name) | Purity | LRAP | nDCG@1 | V | PairF1 | K | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| abs+concl concat | 0.627 | 0.726 | 0.628 | 0.594 | **0.508** | 13 | 0.812 |
| early fusion B1 α=0.75 | 0.652 | 0.728 | 0.643 | 0.604 | 0.501 | 15 | 0.744 |
| mix 0.6/0.2/0.2 | 0.612 | 0.717 | 0.598 | 0.580 | 0.462 | 13 | 0.796 |
| abstract only | 0.561 | 0.666 | 0.565 | 0.548 | 0.446 | 16 | 0.862 |
| title+abs | 0.594 | 0.676 | 0.561 | 0.542 | 0.428 | 14 | 0.832 |
| abs+intro+concl | 0.631 | 0.660 | 0.546 | 0.539 | **0.380** | 13 | 0.894 |
| ACC (restricted) | 0.499 | 0.633 | 0.486 | 0.467 | 0.323 | 23 | 0.891 |

On this restricted protocol, **adding conclusion helps** vs abstract-only (0.508 vs 0.446 PairF1). Adding introduction in a flat concat **hurts** on exact 653 (0.380).

### 8.2 Fit 656 (not 653)

`artifacts/extension_full656/fair656_all_methods.csv`: abs+intro+concl PairF1 ≈ **0.521**. Keep labeled as 656.

### 8.3 Sliding windows & clusterer ablations

- Sliding / window pooling: `artifacts/extension_sliding/results_full.csv` (n=612).
- Clusterer variants (k-means, agglomerative, late fusion): `artifacts/extension/results.csv` (n=612).

## 9. Claims check

| Claim | Verdict |
|---|---|
| Ours A3 0.557 > all published PairF1 | OK if marked exploratory |
| Ours A1 0.549 > published max 0.403 | OK (fixed; Regime A) |
| Ours R0@4488 PairF1 0.512 > ACC-top30 0.403 | OK |
| Ours R0@4488 beats ACC-top30 on all 5 metrics | OK |
| Ours R0@4488 beats full ACC on all 5 | **No** — loses Purity |
| Title fusion always helps | **No** — fails on 4488 |
| Fair claim-level match to ACC | **No** |
| 0.521 abs+intro+concl on 653 | **False** (that is 656) |
| Calling paper ACC “ours” | **Wrong** — fixed in this report |

## 10. Limitations

- Exploratory α on official 653 for A2/A3 “best” rows.
- Regime B matches paper IDs only; ACC is claim-level.
- Prior extension (Regime C) uses different UMAP settings and often CSV-only provenance.
- Incomplete full-text coverage for section experiments.

## 11. Reproducibility (our runners)

```bash
# Ours: windows
uv run python artifacts/assignment1/pipeline/run_workstream_a.py --stage ab --encoders specter2,semcse,qwen

# Ours: title–abstract fusion @6517
uv run python artifacts/assignment1_title_fusion/run_title_fusion.py

# Ours: same @4488 ACC paper IDs
uv run python scripts/run_fit4488.py

# Registry / validation (optional)
uv run python scripts/build_experiment_registry.py
uv run python scripts/validate_experiments.py
```

Prior extension CSVs are **not** regenerated by the commands above.

## 12. Conclusion

| Question | Answer |
|---|---|
| Best **ours** on 6517? | A3 α=0.1 **0.557** (exploratory); fixed A1 **0.549** |
| Best **ours** on 4488? | SPECTER2 R0 **0.512** |
| Best prior extension on 653? | abs+concl **0.508** |
| Beat published PairF1? | Yes for ours A1/A3 (A) and R0 (B) |
| Paper ACC “ours”? | **No** |

Notebook: `notebooks/assignment_template.ipynb`.  
How-to: `artifacts/assignment1/README.md`.
