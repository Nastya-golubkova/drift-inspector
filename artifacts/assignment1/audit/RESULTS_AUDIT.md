# Results Audit — Methodological Consistency

Audit date: consolidation pass after Regime B fit4488.  
Scope: documentation and provenance only — **no new experiments**.

**Authorship:** “Ours” = assignment runners (windows + title fusion + fit4488). Paper ACC/Table 2 = **Published** only. `extension_*` abs+concl / sliding = **Prior extension** (historical `OURS_*` CSV names are filenames).

## 1. Critical finding: Table 2 mislabeling

### Issue

`artifacts/assignment1/results/final_comparison_table.csv` and the previous `REPORT.md` presented values from `artifacts/baseline/stop_alignment_table.csv` under the label **Published**. Those values are the **baseline_comparison.ipynb export** (reproduction / project artifact), not the paper’s rounded Table 2 glyphs.

`final_comparison_table.csv` further duplicates the same numbers under both `role=Published` and `role=Reproduced`, so “reproduction” appeared trivially identical to “published” without establishing a separate paper reference.

### Resolution

- **Table 1 (published):** `artifacts/assignment1/results/table2_published_original.csv` — paper values from `baseline_reproduction_audit.csv` `published_*` columns.
- **Table 2 (reproduced):** `artifacts/assignment1/results/table2_reproduced_from_artifacts.csv` — `stop_alignment_table.csv`.
- Deltas: `artifacts/assignment1/audit/table2_published_vs_reproduced_deltas.csv`.
- Historical CSVs (`stop_alignment_table.csv`, `final_comparison_table.csv`) **left unchanged** (no overwrite of artifacts).

### Notable discrepancies (reproduced − published)

| Method | Metric | Published | Reproduced | \|Δ\| |
|---|---|---:|---:|---:|
| BERTopic MPNet | PairF1 | 0.234 | 0.249 | 0.015 |
| LDA | V-measure | 0.278 | 0.270 | 0.008 |
| BERTopic SPECTER2 | Purity | 0.324 | 0.318 | 0.006 |
| BERTopic MPNet | nDCG@1 | 0.509 | 0.516 | 0.007 |

Most other cells differ by ≤0.003 (rounding). BERTopic MPNet PairF1 is beyond typical rounding; treat the artifact as a verified **export of the project run**, not as the paper glyph.

### Reproduction verification status

From `baseline_reproduction_audit.csv`: methods are marked `reproduced` with `executed_in_repo=yes` via `baseline_comparison.ipynb` outputs matching the stop_alignment export. Assignment1 session did **not** re-run those baselines; provenance is “artifact matches prior notebook export,” not a fresh 2026 re-execution log in this consolidation pass.

## 2. Regime separation checks

| Check | Result |
|---|---|
| 4488 IDs from `acc_clusters_emnlp.csv` | Pass — n=4488 unique |
| 653 ∩ 4488 | Pass — 653 |
| A3 α labeled exploratory | Pass — corrected in REPORT |
| A1 labeled fixed (no α) | Pass |
| 656 abs+intro+concl not filed as 653 | Pass — separate `C_656` row in canonical table |
| Regime C UMAP ≠ A/B noted | Pass |

## 3. Metric consistency spot-checks

Verified against source CSVs (not estimates):

| Result | Source | Five metrics present |
|---|---|---|
| SPECTER2 A3 α0.1 @6517 | title_fusion_results.csv | Yes — PF 0.557 |
| SPECTER2 A1 @6517 | title_fusion_results.csv | Yes — PF 0.549 |
| SPECTER2 R0 @4488 | fit4488_results.csv | Yes — PF 0.512; P 0.620; LRAP 0.708; nDCG 0.638; V 0.605 |
| fair653 abs+concl | fair653_all_methods.csv | Yes — PF 0.508 |
| fair656 abs+intro+concl | fair656_all_methods.csv | Yes — PF 0.521 |

## 4. Claims that were overstated / corrected

1. **“Published Table 2 (from project artifacts)”** with artifact numbers → replaced by separate published vs reproduced tables.
2. **Regime B “outperforms ACC-top30”** without five-metric check → verified: R0 beats published ACC-top30 on all five; still loses **Purity** to full ACC.
3. **Title fusion as general improvement** → constrained: helps on 6517; does not transfer to 4488.
4. **Fair comparison with ACC** → constrained: paper-level vs claim-level.

## 5. Unsupported or unresolved items

| Item | Status |
|---|---|
| Fresh re-execution of LDA/NMF/BERTopic/ACC in this audit session | Not done (by design — no new experiments) |
| Why BERTopic MPNet PairF1 differs by 0.015 from paper | Unresolved; export kept as Reproduced |
| Full logs for every Regime C CSV row | Incomplete; many CSV-only |
| Whether `final_comparison` `Published` role should be rewritten | Deferred — corrected presentation in REPORT; raw file preserved |

## 6. Inventory / registry

- `experiment_registry.csv`: 226 rows built from existing CSVs.
- Ambiguity: Table2 rows appear both as published reference and as reproduced exports; registry tags regimes `published_table2` / `B_4488` / `A_6517` / `C_restricted`.
- Not every registry row has a regenerating script in-repo; do not claim bit-reproducible re-runs for CSV-only Regime C entries.

## 7. Deliverables produced by this audit

1. Corrected `REPORT.md`
2. `canonical_comparison_table.csv`
3. `table2_published_original.csv` + `table2_reproduced_from_artifacts.csv`
4. This audit document + delta CSV
5. Updated `notebooks/final_experimental_analysis.ipynb` (Table 2 separation)
6. Unsupported-claims list (Section 5 and REPORT §10)
