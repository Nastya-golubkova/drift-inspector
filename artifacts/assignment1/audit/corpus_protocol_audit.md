# Corpus protocol audit (Assignment 1)

## Corpora in this repository

| Corpus | N | Path / definition | Role in paper |
|---|---:|---|---|
| `emnlp_full` abstracts | **6517** | `data/claims_sources/emnlp_full/papers.csv` | Fit corpus for LDA, NMF, BERTopic×2, SentSPECTER |
| Balanced study papers | **4488** | years 2020–2025 × 748; `artifacts/baseline/acc_clusters_emnlp.csv` | Fit corpus for ACC (claims) |
| ACC claims (qwen) | **16576** | `data/claims/openrouter_claims.csv` filtered to study | ACC embedding + clustering |
| Table 2 eval | **653** | `artifacts/extension_full656/author_table2_eval_papers.csv` (= EMNLP 2020 ∩ SToP) | SToP metrics only |
| Section full-text | **653** | SToP `Whole Extract` matched by title | full-text section experiments only; **not** Assignment 1 (abstract methods) |

Checksum of sorted 653 `paper_id`s (sha256):  
`b23dc926cb35a29b0caba0b41838806edd7de6f37bb6fc0822e2c112eaaf4dc4`

## Original Table 2 protocol (correct)

1. **Fit** topic model / clustering on the large corpus (6517 or 4488/claims).
2. Build paper-level soft topic bags for all fit documents.
3. **Evaluate** alignment metrics only on the **653** SToP-matched papers.
4. Purity on all 653; LRAP / nDCG@1 / V / PairF1 on nonzero-topic subset (Coverage).
5. Evaluator: RepeatedKFold 5×5 seed 42; PairF1 200k pairs seed 42; `k_gt=1`.

Do **not** refit UMAP/HDBSCAN on the 653 alone and call that Table 2.

## Incorrect protocol used in fair653

`artifacts/extension_653/fair653_all_methods.csv` fitted **every** method (including ACC and OURS) only on the 653 papers (paper-scale UMAP `n_neighbors=15`).  

| Method | Table 2 PairF1 (large fit) | fair653 PairF1 (653 fit) |
|---|---:|---:|
| ACC | 0.361 | 0.323 |
| ACC-top30 | 0.403 | (not run; K&lt;30) |
| OURS abs+concl | n/a | 0.508 |

Comparing OURS@fair653 to published ACC is **invalid**.

## ACC: why 0.361 vs 0.323

- **0.361**: clusters from 16576 claims → K=80; 653 inherit soft bags (Coverage 0.839).
- **0.323**: recluster 2356 claims of those 653 papers → K=23; different geometry.

Harness on author clusters reproduces PairF1 **0.3607** (`artifacts/audit/extractor_ablation_STOP_FINDINGS.md`).

## Assignment 1 (abstract methods) implication

New abstract multi-window methods **must** fit on **6517** abstracts and evaluate on **653**, matching abstract baselines — never mix full-text sections into the main grade path.
