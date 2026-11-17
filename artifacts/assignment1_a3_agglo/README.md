# Setup (5) — A3 + UMAP + agglomerative ward

```bash
uv run python artifacts/assignment1_a3_agglo/run_a3_agglo.py --alpha 0.1 --k 30
```

Requires setup (2) similarity cache:
`artifacts/assignment1_title_fusion/caches/similarity/specter2_S_late_a0.1.npy`

Pipeline: `D=1-S` -> UMAP(precomputed) -> Ward on UMAP 5D. Coverage = 1.0 (all 653 eval papers).
K=30 is eval-selected among {30,54,80} (exploratory).

Headline CSV: `results/a3_agglo_headline.csv`
