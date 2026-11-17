# Assignment 1 — Our methods & how to reproduce

Short map of **what is ours**, **what is from the paper**, and **what is prior extension CSV work** already in the repo.

## Who owns which result

| Label | Meaning | Examples |
|---|---|---|
| **Published** | Drift Inspector paper Table 2 | LDA, NMF, BERTopic, SentSPECTER, **ACC**, ACC-top30 |
| **Ours (this work)** | Methods we implemented & ran for this assignment | Abstract windows R0/R2/R3; title–abstract T0/A1/A2/A3; fit-4488 refit |
| **Prior extension** | Older restricted-fit CSVs in `artifacts/extension_*` (not paper; not re-run in the title-fusion/fit4488 pass) | abs+concl on 653, sliding windows, early fusion B1 |

Paper ACC is **never** “ours”. CSV names like `OURS_abs+concl_concat` are historical filenames only.

---

## Method families

### 1. Ours — abstract windows (fit 6517 → eval 653)

Code: `artifacts/assignment1/pipeline/`  
Results: `results/stage_ab_results.csv`, `results/stage_c_results.csv`

| ID | Idea |
|---|---|
| R0 | Full abstract → one embedding |
| R2 | Mean of abstract sliding windows |
| R3 | Token-weighted mean of windows |
| late MaxSim | Window similarity MaxSim + HDBSCAN/Agglo (Stage C; weak / noisy) |

Encoders: SPECTER2, SemCSE, Qwen. Clustering: UMAP nn=40 → HDBSCAN mcs=25.

```bash
uv run python artifacts/assignment1/pipeline/run_workstream_a.py --stage ab --encoders specter2,semcse,qwen
# optional Stage C:
uv run python artifacts/assignment1/pipeline/run_workstream_a.py --stage c --encoders specter2,semcse
```

### 2. Ours — title ↔ abstract fusion (fit 6517 → eval 653)

Code: `artifacts/assignment1_title_fusion/run_title_fusion.py`  
Results: `artifacts/assignment1_title_fusion/results/title_fusion_results.csv`

| ID | Idea |
|---|---|
| T0 | Title only |
| A1 | Text concat `Title: … Abstract: …` (fixed, no α) |
| A2 | Embedding mix `normalize(α e_title + (1-α) e_abstract)`, α∈{0.1…0.5} |
| A3 | Late similarity mix `S = α S_title + (1-α) S_abstract` → UMAP precomputed |

Best fixed: SPECTER2 A1 PairF1 ≈ **0.549**. Best exploratory: SPECTER2 A3 α=0.1 ≈ **0.557** (α chosen on eval labels).

```bash
uv run python artifacts/assignment1_title_fusion/run_title_fusion.py
# or: uv run python scripts/run_fit4488.py  # wrapper for 4488 only
```

### 3. Ours — same fusion on ACC paper IDs (fit 4488 → eval 653)

Code: `artifacts/assignment1_fit4488/run_fit4488.py` (+ `scripts/run_fit4488.py`)  
Results: `artifacts/assignment1_fit4488/results/fit4488_results.csv`  
IDs: unique `paper_id` from `artifacts/baseline/acc_clusters_emnlp.csv`.

Same T0/A1/A2/A3 + R0. Embeddings sliced from 6517 caches; **clustering refit** on 4488.  
Best fixed: SPECTER2 R0 PairF1 ≈ **0.512** (title fusion does not help here).

```bash
uv run python scripts/run_fit4488.py
# or: uv run python artifacts/assignment1_fit4488/run_fit4488.py --encoders specter2,semcse,qwen
```

### 4. Prior extension — sections on restricted 653 (fit=eval=653)

Artifacts only: `artifacts/extension_653/fair653_all_methods.csv`  
(Often UMAP nn≈15 — **not** the same clustering config as families 1–3.)

| Historical CSV name | Meaning | PairF1 |
|---|---|---:|
| `OURS_abs+concl_concat` | Abstract + conclusion text concat | **0.508** |
| `OURS_B1_a0.75` | Early embedding fusion abs+concl α=0.75 | 0.501 |
| `OURS_mix_0.6/0.2/0.2` | Weighted abs/intro/concl | 0.462 |
| `OURS_abstract` / `SPECTER2-abs+HDBSCAN` | Abstract only | 0.446 |
| `OURS_title+abs` | Title + abstract | 0.428 |
| `OURS_abs+intro+concl_concat` | Abs+intro+concl | 0.380 |
| `ACC` | ACC on this restricted set | 0.323 |

**Do not** call the 656 abs+intro+concl result (PairF1≈0.521) a 653 result — see `artifacts/extension_full656/`.

### 5. Prior extension — sliding windows (mostly abstract; fit=eval=612)

`artifacts/extension_sliding/results_full.csv` — window mean / weighted mean / mixes.

### 6. Prior extension — clustering ablations (fit=eval=612)

`artifacts/extension/results.csv` — same embeddings, vary clusterer (UMAP+HDBSCAN / k-means / agglomerative; late fusion + spectral/agglo).

---

## Analysis (no GPU)

```bash
uv run python scripts/build_experiment_registry.py
uv run python scripts/validate_experiments.py
```

Written report: repo-root [`REPORT.md`](../../REPORT.md)  
Submission notebook: [`notebooks/assignment_template.ipynb`](../../notebooks/assignment_template.ipynb)  
Audit: [`audit/RESULTS_AUDIT.md`](audit/RESULTS_AUDIT.md)

## Layout (our runners)

```
artifacts/assignment1/           # windows pipeline + registry/audit
artifacts/assignment1_title_fusion/
artifacts/assignment1_fit4488/
artifacts/assignment1_dual_cluster/  # exploratory dual / feat_third
scripts/run_fit4488.py
scripts/build_experiment_registry.py
scripts/validate_experiments.py
notebooks/assignment_template.ipynb
```
