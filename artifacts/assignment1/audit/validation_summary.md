# Validation summary

## Protected artifact hashes

- `artifacts/assignment1/results/final_comparison_table.csv` sha256=`d225bd96fa2b60e6…` present
- `artifacts/assignment1/results/stage_ab_results.csv` sha256=`e04f137125471cd0…` present
- `artifacts/baseline/stop_alignment_table.csv` sha256=`a90d2d154e7779d4…` present
- `artifacts/assignment1_title_fusion/results/title_fusion_results.csv` sha256=`5e1c77118f2fb2bd…` present

- Published ACC-top30 PairF1=0.403326 (expected ~0.403)

## Metric finiteness

- OK `artifacts/assignment1_title_fusion/results/title_fusion_results.csv` (39 rows)
- OK `artifacts/assignment1/results/stage_ab_results.csv` (9 rows)
- OK `artifacts/assignment1_fit4488/results/fit4488_results.csv` (39 rows)
- OK `artifacts/extension_653/fair653_all_methods.csv` (13 rows)

## Regime B fit4488 protocol

- OK manifest n=4488, eval flagged=653
- OK artifacts/assignment1_fit4488/eval_intersection_verification.json
- OK all rows fit_n=4488 eval_n=653 (n_rows=39)
- cluster assignment files: 39

## Registry

- OK registry rows=226

regime            role      
A_6517            Our method    65
B_4488            Our method    39
                  Reproduced     2
C_restricted      Our method    88
                  Reproduced    18
published_table2  Published      8
                  Reproduced     6

## Summary counts

- verified OK: 13
- failures: 0
- missing/pending: 0
