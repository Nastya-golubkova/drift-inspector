#!/usr/bin/env python3
"""Validate experiment artifacts; write audit/validation_summary.md."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "artifacts/assignment1/audit/validation_summary.md"

PROTECTED = [
    ROOT / "artifacts/assignment1/results/final_comparison_table.csv",
    ROOT / "artifacts/assignment1/results/stage_ab_results.csv",
    ROOT / "artifacts/baseline/stop_alignment_table.csv",
    ROOT / "artifacts/assignment1_title_fusion/results/title_fusion_results.csv",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_metrics_finite(path: Path, metric_cols: list[str]) -> list[str]:
    issues = []
    if not path.exists():
        return [f"MISSING {path}"]
    df = pd.read_csv(path)
    for c in metric_cols:
        if c not in df.columns:
            issues.append(f"{path.name}: missing column {c}")
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.isna().any():
            issues.append(f"{path.name}: NaN in {c} ({int(s.isna().sum())} rows)")
        if np.isinf(s.fillna(0)).any():
            issues.append(f"{path.name}: Inf in {c}")
    return issues


def main():
    lines = ["# Validation summary", ""]
    ok = []
    bad = []
    missing = []

    # Protected hashes
    lines.append("## Protected artifact hashes")
    lines.append("")
    for p in PROTECTED:
        if p.exists():
            digest = sha256(p)
            ok.append(str(p.relative_to(ROOT)))
            lines.append(f"- `{p.relative_to(ROOT)}` sha256=`{digest[:16]}…` present")
        else:
            missing.append(str(p))
            lines.append(f"- MISSING `{p}`")
    lines.append("")

    # Published Table 2 spot-check
    t2 = ROOT / "artifacts/assignment1/results/final_comparison_table.csv"
    if t2.exists():
        df = pd.read_csv(t2)
        pub = df[df.role == "Published"]
        acc = pub[pub.method.astype(str).str.contains("ACC-top30")]
        if not acc.empty:
            pf = float(acc.iloc[0]["PairF1"])
            if abs(pf - 0.403326) < 1e-5 or abs(pf - 0.403) < 1e-3:
                ok.append("Table2 ACC-top30 PairF1 unchanged ~0.403")
                lines.append(f"- Published ACC-top30 PairF1={pf:.6f} (expected ~0.403)")
            else:
                bad.append(f"ACC-top30 PairF1 drifted: {pf}")
                lines.append(f"- ERROR ACC-top30 PairF1={pf}")
    lines.append("")

    # Metric finiteness
    lines.append("## Metric finiteness")
    lines.append("")
    checks = [
        (ROOT / "artifacts/assignment1_title_fusion/results/title_fusion_results.csv",
         ["Purity", "LRAP", "nDCG@1", "V_measure", "PairF1"]),
        (ROOT / "artifacts/assignment1/results/stage_ab_results.csv",
         ["Purity", "LRAP_mean", "V_measure", "PairF1"]),
        (ROOT / "artifacts/assignment1_fit4488/results/fit4488_results.csv",
         ["Purity", "LRAP", "nDCG@1", "V_measure", "PairF1", "fit_n", "eval_n"]),
        (ROOT / "artifacts/extension_653/fair653_all_methods.csv",
         ["PairF1", "Purity"]),
    ]
    for path, cols in checks:
        issues = check_metrics_finite(path, cols)
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            lines.append(f"- pending/missing `{path.relative_to(ROOT)}`")
        elif issues:
            bad.extend(issues)
            for iss in issues:
                lines.append(f"- FAIL {iss}")
        else:
            ok.append(str(path.relative_to(ROOT)))
            lines.append(f"- OK `{path.relative_to(ROOT)}` ({len(pd.read_csv(path))} rows)")
    lines.append("")

    # fit4488 protocol
    lines.append("## Regime B fit4488 protocol")
    lines.append("")
    man = ROOT / "artifacts/assignment1_fit4488/corpus_manifest.csv"
    ver = ROOT / "artifacts/assignment1_fit4488/eval_intersection_verification.json"
    res = ROOT / "artifacts/assignment1_fit4488/results/fit4488_results.csv"
    if man.exists():
        m = pd.read_csv(man)
        if len(m) == 4488 and m.paper_id.nunique() == 4488:
            ok.append("fit4488 manifest")
            lines.append(f"- OK manifest n={len(m)}, eval flagged={int(m.in_eval653.sum())}")
        else:
            bad.append("manifest size")
            lines.append(f"- FAIL manifest n={len(m)}")
    else:
        missing.append("corpus_manifest.csv")
    if ver.exists():
        ok.append("eval verification json")
        lines.append(f"- OK {ver.relative_to(ROOT)}")
    if res.exists():
        df = pd.read_csv(res)
        if (df.fit_n == 4488).all() and (df.eval_n == 653).all():
            ok.append("fit4488 fit_n/eval_n columns")
            lines.append(f"- OK all rows fit_n=4488 eval_n=653 (n_rows={len(df)})")
        else:
            bad.append("fit4488 fit_n mismatch")
            lines.append("- FAIL fit_n/eval_n not uniform")
        # cluster CSVs exist for each row
        clust = ROOT / "artifacts/assignment1_fit4488/clusters"
        n_clust = len(list(clust.glob("*.csv"))) if clust.exists() else 0
        lines.append(f"- cluster assignment files: {n_clust}")
    else:
        lines.append("- fit4488 results pending")
    lines.append("")

    # Registry
    reg = ROOT / "artifacts/assignment1/experiment_registry.csv"
    lines.append("## Registry")
    lines.append("")
    if reg.exists():
        rdf = pd.read_csv(reg)
        lines.append(f"- OK registry rows={len(rdf)}")
        lines.append("")
        lines.append(rdf.groupby(["regime", "role"]).size().to_string())
        ok.append("experiment_registry.csv")
    else:
        missing.append("experiment_registry.csv")
        lines.append("- registry not built yet")
    lines.append("")

    lines.append("## Summary counts")
    lines.append("")
    lines.append(f"- verified OK: {len(ok)}")
    lines.append(f"- failures: {len(bad)}")
    lines.append(f"- missing/pending: {len(missing)}")
    if bad:
        lines.append("")
        lines.append("### Failures")
        for b in bad:
            lines.append(f"- {b}")
    if missing:
        lines.append("")
        lines.append("### Missing / pending")
        for m in missing:
            lines.append(f"- {m}")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines) + "\n")
    print(SUMMARY.read_text())
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
