# Experiment Inventory — Assignment 1 Consolidation

Do not infer protocols from directory names alone. Fit/eval sizes below come from CSVs and verification scripts.

## Regime A — Full EMNLP corpus (fit 6517 → eval 653)

| Family | Source artifact | Representations | Clustering | Notes |
|---|---|---|---|---|
| Published + Reproduced Table 2 | `artifacts/assignment1/results/final_comparison_table.csv`, `artifacts/baseline/stop_alignment_table.csv` | LDA/NMF/BERTopic/SentSPECTER/ACC | original pipelines | ACC historically fit on 4488 claims corpus |
| Abstract windows R0/R2/R3 | `artifacts/assignment1/results/stage_ab_results.csv` | SPECTER2/SemCSE/Qwen full / mean-window / weighted-window | UMAP nn=40 → HDBSCAN mcs=25 | Fit 6517 |
| Late MaxSim | `artifacts/assignment1/results/stage_c_results.csv` | window MaxSim | HDBSCAN precomputed / Agglo K=80 | Fit 6517 |
| Title–abstract fusion | `artifacts/assignment1_title_fusion/results/title_fusion_results.csv` | T0, A1, A2 α, A3 α | same UMAP/HDBSCAN; A3 precomputed | R0 rows **copied** from stage_ab |

**Best exploratory (Regime A):** SPECTER2 `similarity_late_A3` α=0.1 → PairF1 **0.557**. Alpha selected on official 653 labels.

## Regime B — ACC-matched corpus (fit 4488 → eval 653) — NEW

| Item | Path |
|---|---|
| Paper ID source | `artifacts/baseline/acc_clusters_emnlp.csv` (unique `paper_id`, n=4488) |
| Manifest | `artifacts/assignment1_fit4488/corpus_manifest.csv` |
| Verification | `artifacts/assignment1_fit4488/eval_intersection_verification.json` |
| Results | `artifacts/assignment1_fit4488/results/fit4488_results.csv` |

Intersection with official 653 eval IDs: **653** (eval ⊂ fit4488).

## Regime C — Restricted fit (fit = eval on small sets)

| Corpus | Fit=Eval | Source | Key verified PairF1 |
|---|---:|---|---|
| Exact 653 | 653 | `artifacts/extension_653/fair653_all_methods.csv` | abs+concl 0.508; B1 α0.75 0.501; SPECTER2-abs 0.446; ACC 0.323; abs+intro+concl **0.380** |
| Full matched 656 | 656 | `artifacts/extension_full656/fair656_all_methods.csv` | abs+intro+concl **0.521** (do **not** label as 653) |
| Section ∩ 612 | 612 | `artifacts/extension/results.csv`, `artifacts/intersection/baseline_on_intersection.csv` | early fusion / clusterer ablations |
| Expanded 645 | 645 | `artifacts/extension_expanded/results_expanded.csv` | conclusion-proxy mixes |
| Sliding 612 | 612 | `artifacts/extension_sliding/results_full.csv` | windowed sections |

Restricted clustering often uses UMAP nn=15 / HDBSCAN mcs≈15 — **different** from Regime A/B (nn=40 / mcs=25). Never mix regime metrics in one ranking without `fit_n`.

## Shared evaluation

- Official eval IDs: `artifacts/extension_full656/author_table2_eval_papers.csv` (653)
- SToP labels: `data/external/STop_topic_classification_dataset_for_scientific_papers.csv`
- Full paper table: `data/claims_sources/emnlp_full/papers.csv` (6517)
