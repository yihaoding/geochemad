#!/usr/bin/env python3
"""
Run 20-run negative-sampling evaluation on existing anomaly_scores.csv files
for the ICDM aligned_input runs. Writes metrics.json next to the CSV.

Usage:
    python eval_anomaly_scores.py [--out_root OUTPUT_ROOT]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from evaluate import evaluate, _standardize_xy  # noqa: E402
from area_config import AREAS                   # noqa: E402

AREA_KEY_MAP = {
    "sed1":  "area1_sed1_au",
    "sed2":  "area2_sed2_cu",
    "rock1": "area3_rock1_w",
    "rock2": "area4_rock2_au",
    "rock3": "area5_rock3_cu",
    "soil1": "area6_soil1_au",
    "soil2": "area7_soil2_au",
    "soil3": "area8_soil3_ni",
    "sed3":  "area15_sed3_au",
    "sed4":  "area16_sed4_cu",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="outputs_icdm_aligned_input")
    p.add_argument("--neg_runs", type=int, default=20)
    p.add_argument("--label", default="icdm")
    args = p.parse_args()

    area_cfg = {a["area_id"]: a for a in AREAS}

    for short, full in AREA_KEY_MAP.items():
        csv = Path(args.out_root) / short / "t1" / "anomaly_scores.csv"
        if not csv.exists():
            print(f"[skip] {short}: no {csv}")
            continue

        cfg = area_cfg[full]
        h = cfg["hparams"]
        dep_csv = cfg["deposits"]

        samp_df = _standardize_xy(pd.read_csv(csv))
        dep_df  = _standardize_xy(pd.read_csv(dep_csv))

        metrics = evaluate(
            samp_df[["x", "y"]].to_numpy(),
            samp_df["anomaly_score"].to_numpy(),
            dep_df[["x", "y"]].to_numpy(),
            radius_km=h["radius_km"],
            min_dist=h["min_dist"],
            neg_runs=args.neg_runs,
            label=f"{args.label}/{short}",
        )

        out = csv.parent / "metrics.json"
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[ok] {short}  AUC={metrics['AUC_mean']:.4f}±{metrics['AUC_std']:.3f}  → {out}")


if __name__ == "__main__":
    main()
