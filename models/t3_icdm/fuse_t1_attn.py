#!/usr/bin/env python3
"""
Fuse T1 supervised anomaly score with one unified attention score using
rank-mean. Re-evaluates with 20-run negative sampling. Produces a per-area
comparison: T1 only vs attn only vs (T1 + attn) rank fusion.

Default unified variant = attn_local_mean_dist (top-ranked global mean).
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from evaluate import evaluate, _standardize_xy  # noqa: E402
from area_config import AREAS as AREA_CFGS      # noqa: E402

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


def fuse_rank(s_a: np.ndarray, s_b: np.ndarray) -> np.ndarray:
    return (rankdata(s_a) + rankdata(s_b)) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="attn_local_mean_dist")
    p.add_argument("--neg_runs", type=int, default=20)
    args = p.parse_args()

    cfg_by_id = {a["area_id"]: a for a in AREA_CFGS}
    rows = []
    for short, area_id in AREAS:
        t1_csv  = ALIGNED / short / "t1" / "anomaly_scores.csv"
        atn_csv = ALIGNED / short / "attn" / f"{args.variant}.csv"
        if not (t1_csv.exists() and atn_csv.exists()):
            print(f"[skip] {short}: missing csv (t1={t1_csv.exists()}, attn={atn_csv.exists()})")
            continue

        cfg = cfg_by_id[area_id]; h = cfg["hparams"]
        dep_df = _standardize_xy(pd.read_csv(cfg["deposits"]))

        t1_df  = _standardize_xy(pd.read_csv(t1_csv))
        atn_df = _standardize_xy(pd.read_csv(atn_csv))

        # alignment: both files have rows in the same data-load order (load_area
        # is deterministic). Sanity-check by comparing x,y.
        assert np.allclose(t1_df[["x","y"]].to_numpy(),
                            atn_df[["x","y"]].to_numpy(), atol=1e-3), \
               f"x/y mismatch between t1 and attn for {short}"
        xy   = t1_df[["x","y"]].to_numpy()
        dep  = dep_df[["x","y"]].to_numpy()
        s_t1 = t1_df["anomaly_score"].to_numpy()
        s_at = atn_df["anomaly_score"].to_numpy()
        s_fu = fuse_rank(s_t1, s_at)

        def _eval(scores, tag):
            return evaluate(xy, scores, dep,
                            radius_km=h["radius_km"], min_dist=h["min_dist"],
                            neg_runs=args.neg_runs,
                            label=f"fuse/{short}/{tag}")

        m_t1 = _eval(s_t1, "t1")
        m_at = _eval(s_at, "attn")
        m_fu = _eval(s_fu, "fuse")

        rows.append({
            "area":      short,
            "T1_AUC":    m_t1["AUC_mean"],
            "T1_std":    m_t1["AUC_std"],
            "attn_AUC":  m_at["AUC_mean"],
            "attn_std":  m_at["AUC_std"],
            "fuse_AUC":  m_fu["AUC_mean"],
            "fuse_std":  m_fu["AUC_std"],
            "delta_fuse_vs_T1": m_fu["AUC_mean"] - m_t1["AUC_mean"],
        })

    df = pd.DataFrame(rows)
    mean_row = {c: df[c].mean() if df[c].dtype.kind == "f" else "MEAN"
                for c in df.columns}
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    out = REPORTS / f"icdm_fuse_t1_{args.variant}.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    print(f"\n=== T1 vs attn ({args.variant}) vs rank-fuse — per area ===")
    print(df.to_string(index=False, float_format="%.4f"))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
