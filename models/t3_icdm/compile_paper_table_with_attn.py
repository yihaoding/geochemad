#!/usr/bin/env python3
"""
Build one unified 20-run AUC table that adds all attention-score variants
to the existing ICDM Table I (auc_main): T1 (Vanilla Transformer) and
T2 (GeoChemFormer) values are taken VERBATIM from the published LaTeX
(CLR + LLM-selected features). Attention values come from
outputs_icdm_aligned_input/<short>/attn/attn_metrics.json — same protocol
(CLR + LLM features + 20-run negative sampling).

Outputs:
  reports/icdm_table1_with_attn.csv          (auc±std per cell)
  reports/icdm_table1_with_attn_auc_only.csv (auc only — easy to import)
  reports/icdm_table1_with_attn.tex          (LaTeX snippet for the paper)
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT    = Path(os.environ.get("GEOCHEM_OUT_ROOT", "outputs"))
ALIGNED = ROOT / "outputs_icdm_aligned_input"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ─── 1. Published Table I (only T2 = GeoChemFormer, per user) ────────────────
# Verbatim from the LaTeX the user pasted. T1 (Vanilla Transformer) omitted.
PAPER = {
    "sed1":  {"T2": (0.864, 0.026)},
    "sed2":  {"T2": (0.688, 0.057)},
    "rock1": {"T2": (0.698, 0.113)},
    "rock2": {"T2": (0.508, 0.068)},  # rock2 N=223 — flagged in paper
    "rock3": {"T2": (0.765, 0.042)},
    "soil1": {"T2": (0.552, 0.085)},
    "soil2": {"T2": (0.650, 0.059)},
    "soil3": {"T2": (0.493, 0.072)},
    "sed3":  {"T2": (0.571, 0.047)},
    "sed4":  {"T2": (0.640, 0.037)},
}
AREAS = list(PAPER.keys())

SCALES = ("local", "mid", "glob")
SCORES = ("entropy", "max", "mean_dist", "top1_dist")
ATTN_VARIANTS = [f"attn_{s}_{sc}{sg}"
                 for s in SCALES for sc in SCORES for sg in ("", "_neg")]


def _read_attn(short: str, variant: str):
    p = ALIGNED / short / "attn" / "attn_metrics.json"
    if not p.exists():
        return (np.nan, np.nan)
    d = json.loads(p.read_text())
    m = d.get(variant)
    if m is None:
        return (np.nan, np.nan)
    return (m.get("AUC_mean", np.nan), m.get("AUC_std", np.nan))


def main():
    methods = ["T2"] + ATTN_VARIANTS
    auc_mat = pd.DataFrame(index=AREAS, columns=methods, dtype=float)
    std_mat = pd.DataFrame(index=AREAS, columns=methods, dtype=float)
    cell_mat = pd.DataFrame(index=AREAS, columns=methods, dtype=object)

    for area in AREAS:
        for m in ("T2",):
            auc, std = PAPER[area][m]
            auc_mat.loc[area, m] = auc
            std_mat.loc[area, m] = std
            cell_mat.loc[area, m] = f"{auc:.3f}±{std:.3f}"
        for v in ATTN_VARIANTS:
            auc, std = _read_attn(area, v)
            auc_mat.loc[area, v] = auc
            std_mat.loc[area, v] = std
            cell_mat.loc[area, v] = (f"{auc:.3f}±{std:.3f}"
                                     if np.isfinite(auc) else "—")

    # MEAN row across all 10 areas
    auc_mat.loc["MEAN_10"] = auc_mat.mean(axis=0, skipna=True)
    std_mat.loc["MEAN_10"] = std_mat.mean(axis=0, skipna=True)
    # also mean excluding rock2 (paper convention — N=223 outlier)
    no_rock = [a for a in AREAS if a != "rock2"]
    auc_mat.loc["MEAN_no_rock2"] = auc_mat.loc[no_rock].mean(axis=0, skipna=True)
    std_mat.loc["MEAN_no_rock2"] = std_mat.loc[no_rock].mean(axis=0, skipna=True)
    for row in ("MEAN_10", "MEAN_no_rock2"):
        cell_mat.loc[row] = [
            (f"{auc_mat.loc[row,c]:.3f}±{std_mat.loc[row,c]:.3f}"
             if np.isfinite(auc_mat.loc[row,c]) else "—")
            for c in methods
        ]

    # ── write CSVs (areas as cols, methods as rows — easier paper layout) ──
    auc_T = auc_mat.T
    cell_T = cell_mat.T

    out_cells = REPORTS / "icdm_table1_with_attn.csv"
    cell_T.to_csv(out_cells)
    out_auc = REPORTS / "icdm_table1_with_attn_auc_only.csv"
    auc_T.to_csv(out_auc, float_format="%.4f")

    # console: full table
    print("=== Full AUC±std table (rows = method, cols = area + means) ===")
    print(cell_T.to_string())

    # ── ranking: which methods beat T2 on MEAN_10? ──────────────────────────
    rank = auc_T[["MEAN_10", "MEAN_no_rock2"]].copy()
    rank["beats_T2_on_MEAN_10"] = rank["MEAN_10"] > auc_mat.loc["MEAN_10", "T2"]
    rank = rank.sort_values("MEAN_10", ascending=False)
    out_rank = REPORTS / "icdm_table1_methods_ranked.csv"
    rank.to_csv(out_rank, float_format="%.4f")
    print("\n=== top 10 methods by MEAN_10 AUC ===")
    print(rank.head(10).to_string(float_format="%.4f"))

    # ── LaTeX snippet (only top-K attn variants, to keep paper table sane) ──
    K = 4   # top 4 attn variants by MEAN_10
    top_attn = [m for m in rank.index if m.startswith("attn_")][:K]
    paper_methods = ["T2"] + top_attn
    lines = [
        "% AUTOGEN: compile_paper_table_with_attn.py",
        f"% Top-{K} attention variants by MEAN_10 AUC appended to T1/T2 baselines.",
        "\\begin{tabular}{l|" + "c" * len(paper_methods) + "}",
        "\\toprule",
        "Area & " + " & ".join(paper_methods).replace("_", "\\_") + " \\\\",
        "\\midrule",
    ]
    for area in AREAS + ["MEAN_10", "MEAN_no_rock2"]:
        row_cells = [area.replace("_", "\\_")]
        # find best in this row to bold
        vals = {m: auc_mat.loc[area, m] for m in paper_methods}
        best = max(vals, key=lambda k: vals[k] if np.isfinite(vals[k]) else -1)
        for m in paper_methods:
            auc = auc_mat.loc[area, m]
            std = std_mat.loc[area, m]
            if not np.isfinite(auc):
                row_cells.append("---")
            else:
                c = f"{auc:.3f}_{{\\pm{std:.3f}}}"
                if m == best:
                    c = "\\mathbf{" + c + "}"
                row_cells.append("$" + c + "$")
        lines.append(" & ".join(row_cells) + " \\\\")
        if area == "soil3": lines.append("\\midrule")
        if area == "sed4":  lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]

    out_tex = REPORTS / "icdm_table1_with_attn.tex"
    out_tex.write_text("\n".join(lines))

    print(f"\n→ {out_cells}")
    print(f"→ {out_auc}")
    print(f"→ {out_rank}")
    print(f"→ {out_tex}  (top-{K} attn appended to T1/T2)")


if __name__ == "__main__":
    main()
