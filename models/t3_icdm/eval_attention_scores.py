#!/usr/bin/env python3
"""
Eval all attn_{scale}_{score}.csv files produced by extract_attention_v3.py
for one area, using the same 20-run negative-sampling protocol as
eval_anomaly_scores.py. Writes one JSON summarising every score variant.

Usage:
    python eval_attention_scores.py \
        --attn_dir <.../attn> \
        --area_id area1_sed1_au \
        --label icdm_attn
"""
import argparse, json, os, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from evaluate import evaluate, _standardize_xy  # noqa: E402
from area_config import AREAS                   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn_dir", required=True,
                   help="Directory containing attn_*.csv files (from extract_attention_v3.py).")
    p.add_argument("--area_id",  required=True,
                   help="e.g. area1_sed1_au; used to look up deposits + hparams.")
    p.add_argument("--neg_runs", type=int, default=20)
    p.add_argument("--label",    default="icdm_attn")
    p.add_argument("--out_json", default=None,
                   help="Defaults to <attn_dir>/attn_metrics.json")
    args = p.parse_args()

    cfg = next(a for a in AREAS if a["area_id"] == args.area_id)
    h = cfg["hparams"]
    dep_df = _standardize_xy(pd.read_csv(cfg["deposits"]))

    attn_dir = Path(args.attn_dir)
    out_json = Path(args.out_json) if args.out_json else attn_dir / "attn_metrics.json"

    results = {}
    csvs = sorted(attn_dir.glob("attn_*.csv"))
    if not csvs:
        raise SystemExit(f"No attn_*.csv files in {attn_dir}")

    for csv in csvs:
        samp_df = _standardize_xy(pd.read_csv(csv))
        metrics = evaluate(
            samp_df[["x", "y"]].to_numpy(),
            samp_df["anomaly_score"].to_numpy(),
            dep_df[["x", "y"]].to_numpy(),
            radius_km = h["radius_km"],
            min_dist  = h["min_dist"],
            neg_runs  = args.neg_runs,
            label     = f"{args.label}/{args.area_id}/{csv.stem}",
        )
        results[csv.stem] = metrics
        print(f"[ok] {csv.stem:>30s}  AUC={metrics['AUC_mean']:.4f}±{metrics['AUC_std']:.3f}")

    # also try sign-flipped variants — for some score types higher-may-mean-normal
    # (e.g. high attention concentration = strong match = NOT anomalous).
    for csv in csvs:
        samp_df = _standardize_xy(pd.read_csv(csv))
        flipped = -samp_df["anomaly_score"].to_numpy()
        metrics = evaluate(
            samp_df[["x", "y"]].to_numpy(), flipped,
            dep_df[["x", "y"]].to_numpy(),
            radius_km=h["radius_km"], min_dist=h["min_dist"],
            neg_runs=args.neg_runs,
            label=f"{args.label}/{args.area_id}/{csv.stem}_neg",
        )
        results[csv.stem + "_neg"] = metrics
        print(f"[ok] {csv.stem + '_neg':>30s}  AUC={metrics['AUC_mean']:.4f}±{metrics['AUC_std']:.3f}")

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    # rank by AUC_mean for a quick console summary
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["AUC_mean"])
    print(f"\n=== top 5 for {args.area_id} ===")
    for name, m in ranked[:5]:
        print(f"  {name:>34s}  AUC={m['AUC_mean']:.4f}±{m['AUC_std']:.3f}")
    print(f"→ {out_json}")


if __name__ == "__main__":
    main()
