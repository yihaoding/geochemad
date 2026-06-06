#!/usr/bin/env python3
"""
Rank the 24 attention-score variants by their *mean* AUC across all areas
(no per-area cherry-pick). Produces:

  1. A ranking of variants by cross-area mean AUC.
  2. A per-area table for the single best unified variant, comparable to T1.
  3. The full per-area × per-variant matrix.

Outputs (under reports/):
  icdm_attn_unified_ranking.csv
  icdm_attn_unified_per_area.csv
  icdm_attn_unified_matrix.csv
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(os.environ.get("GEOCHEM_OUT_ROOT", "outputs"))
ALIGNED = ROOT / "outputs_icdm_aligned_input"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

AREAS = [
    ("sed1",  "area1_sed1_au"),
    ("sed2",  "area2_sed2_cu"),
    ("rock1", "area3_rock1_w"),
    ("rock2", "area4_rock2_au"),
    ("rock3", "area5_rock3_cu"),
    ("soil1", "area6_soil1_au"),
    ("soil2", "area7_soil2_au"),
    ("soil3", "area8_soil3_ni"),
    ("sed3",  "area15_sed3_au"),
    ("sed4",  "area16_sed4_cu"),
]


def main():
    # 1) read every (area × variant) AUC into a long table
    rows = []
    for short, area_id in AREAS:
        j_attn = ALIGNED / short / "attn" / "attn_metrics.json"
        if not j_attn.exists():
            print(f"[skip] {short}: no attn metrics")
            continue
        d = json.loads(j_attn.read_text())
        for variant, m in d.items():
            rows.append({
                "area":     short,
                "variant":  variant,
                "AUC":      m.get("AUC_mean", np.nan),
                "AUC_std":  m.get("AUC_std",  np.nan),
                "AP":       m.get("AP_mean",  np.nan),
                "PR_AUC":   m.get("PR_AUC_mean", np.nan),
            })
    df = pd.DataFrame(rows)

    # 2) wide matrix: rows = variant, cols = area, values = AUC
    M = df.pivot(index="variant", columns="area", values="AUC")
    # reorder area columns to match AREAS list
    M = M[[s for s, _ in AREAS if s in M.columns]]
    M["mean_AUC"]   = M.mean(axis=1)
    M["median_AUC"] = M.iloc[:, :-1].median(axis=1)
    M["min_AUC"]    = M.iloc[:, :-2].min(axis=1)
    # rank by mean
    M = M.sort_values("mean_AUC", ascending=False)

    out_mat = REPORTS / "icdm_attn_unified_matrix.csv"
    M.to_csv(out_mat, float_format="%.4f")
    print(f"→ {out_mat}\n")

    # 3) ranking table
    rank = M[["mean_AUC", "median_AUC", "min_AUC"]].copy()
    out_rank = REPORTS / "icdm_attn_unified_ranking.csv"
    rank.to_csv(out_rank, float_format="%.4f")
    print("=== top 10 unified attention variants (by mean AUC across 10 areas) ===")
    print(rank.head(10).to_string(float_format="%.4f"))

    # 4) per-area table for the single best unified variant
    best_variant = rank.index[0]
    print(f"\n>>> best unified variant: {best_variant}  "
          f"(mean={rank.loc[best_variant, 'mean_AUC']:.4f}, "
          f"min={rank.loc[best_variant, 'min_AUC']:.4f})\n")

    # baseline T1 per area
    t1_rows = []
    for short, area_id in AREAS:
        t1_j = ALIGNED / short / "t1" / "metrics.json"
        attn_j = ALIGNED / short / "attn" / "attn_metrics.json"
        t1 = json.loads(t1_j.read_text()) if t1_j.exists() else {}
        attn = json.loads(attn_j.read_text()) if attn_j.exists() else {}
        b = attn.get(best_variant, {})
        t1_rows.append({
            "area":     short,
            "area_id":  area_id,
            "T1_AUC":   t1.get("AUC_mean", np.nan),
            "T1_std":   t1.get("AUC_std",  np.nan),
            "attn_AUC": b.get("AUC_mean",  np.nan),
            "attn_std": b.get("AUC_std",   np.nan),
            "delta":    (b.get("AUC_mean", np.nan) -
                         t1.get("AUC_mean", np.nan)),
        })
    pa = pd.DataFrame(t1_rows)
    pa.loc[len(pa)] = {
        "area":     "MEAN",
        "area_id":  "",
        "T1_AUC":   pa["T1_AUC"].mean(),
        "T1_std":   pa["T1_std"].mean(),
        "attn_AUC": pa["attn_AUC"].mean(),
        "attn_std": pa["attn_std"].mean(),
        "delta":    pa["delta"].mean(),
    }
    out_pa = REPORTS / "icdm_attn_unified_per_area.csv"
    pa.to_csv(out_pa, index=False, float_format="%.4f")
    print(f"=== unified variant '{best_variant}' per area vs baseline T1 ===")
    print(pa.to_string(index=False, float_format="%.4f"))
    print(f"\n→ {out_pa}")


if __name__ == "__main__":
    main()
