#!/usr/bin/env python3
"""
evaluate.py  –  Unified spatial evaluation for any anomaly scoring method.

Computes the same metrics as the GeoTransformer pipeline:

  --- Discrimination ---
    AUC_mean  ± std   (ROC-AUC over NEG_RUNS negative draws)
    AP_mean   ± std   (Average Precision)
    PR_AUC_mean ± std (PR-curve AUC)

  --- Spatial Targeting ---
    SR_5pct    fraction of deposits whose nearest sample is in top 5% by score
    Recall_50  fraction of deposits whose nearest sample is in top-50 by score

  --- Spatial Consistency ---
    DTD_km     mean distance (km) from top-1% anomalous samples to nearest deposit
    Density    mean anomaly score within radius_km of deposits

Usage
-----
  python evaluate.py \\
      --scores_csv  outputs/area1/anomaly_scores.csv \\
      --deposits    /path/to/area1_site.csv \\
      --radius_km   2.0 \\
      --min_dist    0.5 \\
      --neg_runs    20

Input CSV columns
-----------------
  scores_csv  must have: x (or X), y (or Y), anomaly_score
  deposits    must have: x/X/lon/longitude and y/Y/lat/latitude  (any casing)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
)
from sklearn.neighbors import KDTree


# ── Coordinate helpers ────────────────────────────────────────────────────────

def _standardize_xy(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with canonical 'x' and 'y' float columns, or raise."""
    col_map = {c.lower(): c for c in df.columns}
    for xkey in ("x", "lon", "longitude", "easting"):
        if xkey in col_map:
            df = df.rename(columns={col_map[xkey]: "x"})
            break
    for ykey in ("y", "lat", "latitude", "northing"):
        if ykey in col_map:
            df = df.rename(columns={col_map[ykey]: "y"})
            break
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Cannot find x/y columns. Available: {list(df.columns)}")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    return df.dropna(subset=["x", "y"])


def _pairwise_dist_km(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Pairwise distance matrix (km) between (N,2) and (M,2) coordinate arrays.
    Tries Haversine first (lat/lon); falls back to Euclidean / 1000 when
    values are clearly projected (|x| > 180 or |y| > 90).
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    if A.ndim == 1:
        A = A[None, :]
    if B.ndim == 1:
        B = B[None, :]

    latlon = (np.abs(A[:, 0]).max() <= 180) and (np.abs(A[:, 1]).max() <= 90)
    if latlon:
        Alon = np.radians(A[:, 0])[:, None]
        Alat = np.radians(A[:, 1])[:, None]
        Blon = np.radians(B[:, 0])[None, :]
        Blat = np.radians(B[:, 1])[None, :]
        dlon = Blon - Alon
        dlat = Blat - Alat
        a = np.sin(dlat / 2) ** 2 + np.cos(Alat) * np.cos(Blat) * np.sin(dlon / 2) ** 2
        return 6371.0 * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    else:
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=2)) / 1000.0


# ── Metric helpers (identical logic to new_geo_transformer.py) ────────────────

def _map_deposit_scores(dep_xy, samp_xy, samp_scores, radius_km=2.0):
    """Score for each deposit = score of its nearest sample (nn fallback)."""
    tree = KDTree(samp_xy)
    _, idx = tree.query(dep_xy, k=1)
    return samp_scores[idx.ravel()]


def _sample_negatives(samp_xy, dep_xy, samp_scores, k, min_km, rng):
    """Sample k negative scores at least min_km from every deposit."""
    D = _pairwise_dist_km(samp_xy, dep_xy)
    min_d = D.min(axis=1)
    eligible = np.where(min_d >= min_km)[0]
    if len(eligible) == 0:
        print(f"  ⚠️  No negatives >= {min_km} km from deposits; using all samples.")
        eligible = np.arange(len(samp_scores))
    return samp_scores[rng.choice(eligible, size=k, replace=True)]


def _sr_at_pct(samp_scores, dep_nn_idx, top_pct=0.05):
    n_top   = max(1, int(len(samp_scores) * top_pct))
    top_set = set(np.argsort(-samp_scores)[:n_top])
    n_dep   = len(dep_nn_idx)
    return float("nan") if n_dep == 0 else sum(1 for i in dep_nn_idx if i in top_set) / n_dep


def _recall_k(samp_scores, dep_nn_idx, k=50):
    top_set = set(np.argsort(-samp_scores)[:k])
    n_dep   = len(dep_nn_idx)
    return float("nan") if n_dep == 0 else sum(1 for i in dep_nn_idx if i in top_set) / n_dep


def _dtd_km(samp_xy, samp_scores, dep_xy, top_pct=0.01):
    n_top   = max(1, int(len(samp_scores) * top_pct))
    top_idx = np.argsort(-samp_scores)[:n_top]
    D = _pairwise_dist_km(samp_xy[top_idx], dep_xy)
    return float(D.min(axis=1).mean())


def _density_score(samp_xy, samp_scores, dep_xy, radius_km=2.0):
    vals = []
    for d in dep_xy:
        D      = _pairwise_dist_km(samp_xy, d[None, :]).ravel()
        within = np.where(D <= radius_km)[0]
        if len(within) > 0:
            vals.append(samp_scores[within].mean())
    return float(np.mean(vals)) if vals else float("nan")


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(samp_xy, samp_scores, dep_xy, radius_km, min_dist, neg_runs, label=""):
    rng = np.random.default_rng(42)

    pos_scores = _map_deposit_scores(dep_xy, samp_xy, samp_scores, radius_km)
    k = len(pos_scores)

    tree = KDTree(samp_xy)
    _, dep_nn_idx = tree.query(dep_xy, k=1)
    dep_nn_idx = dep_nn_idx.ravel()

    aucs, aps, pr_aucs = [], [], []
    for _ in range(neg_runs):
        neg_scores = _sample_negatives(samp_xy, dep_xy, samp_scores, k, min_dist, rng)
        y_true  = np.array([1] * k + [0] * len(neg_scores))
        y_score = np.concatenate([pos_scores, neg_scores])
        aucs.append(roc_auc_score(y_true, y_score))
        aps.append(average_precision_score(y_true, y_score))
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_aucs.append(float(np.trapz(prec[::-1], rec[::-1])))

    aucs    = np.array(aucs)
    aps     = np.array(aps)
    pr_aucs = np.array(pr_aucs)

    sr_5pct   = _sr_at_pct(samp_scores, dep_nn_idx, top_pct=0.05)
    recall_50 = _recall_k(samp_scores, dep_nn_idx, k=50)
    dtd_km    = _dtd_km(samp_xy, samp_scores, dep_xy, top_pct=0.01)
    density   = _density_score(samp_xy, samp_scores, dep_xy, radius_km=radius_km)

    results = {
        "AUC_mean":    float(aucs.mean()),    "AUC_std":    float(aucs.std()),
        "AP_mean":     float(aps.mean()),     "AP_std":     float(aps.std()),
        "PR_AUC_mean": float(pr_aucs.mean()), "PR_AUC_std": float(pr_aucs.std()),
        "SR_5pct":     float(sr_5pct),
        "Recall_50":   float(recall_50),
        "DTD_km":      float(dtd_km),
        "Density":     float(density),
    }

    title = f"  {label}" if label else ""
    print(f"\n{'='*50}{title}")
    print(f"  N_samples={len(samp_scores)}  N_deposits={len(dep_xy)}")
    print(f"  neg_runs={neg_runs}  radius_km={radius_km}  min_dist={min_dist} km")
    print(f"{'='*50}")
    groups = {
        "Discrimination":      [("AUC_mean",    "AUC_std"),
                                ("AP_mean",     "AP_std"),
                                ("PR_AUC_mean", "PR_AUC_std")],
        "Spatial Targeting":   [("SR_5pct",   None),
                                ("Recall_50", None)],
        "Spatial Consistency": [("DTD_km",  None),
                                ("Density", None)],
    }
    for grp, keys in groups.items():
        print(f"\n--- {grp} ---")
        for mean_k, std_k in keys:
            v = results[mean_k]
            if std_k:
                print(f"  {mean_k:14s}: {v:.4f} ± {results[std_k]:.4f}")
            else:
                print(f"  {mean_k:14s}: {v:.4f}")
    print(f"{'='*50}\n")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Unified spatial evaluation for geochemical anomaly scores")
    p.add_argument("--scores_csv",  required=True,
                   help="CSV with columns: x/X, y/Y, anomaly_score")
    p.add_argument("--deposits",    required=True,
                   help="Deposit locations CSV (any x/y column naming)")
    p.add_argument("--radius_km",   type=float, default=2.0,
                   help="Deposit neighbourhood radius for Density metric (km)")
    p.add_argument("--min_dist",    type=float, default=0.5,
                   help="Minimum distance from deposits for negative sampling (km)")
    p.add_argument("--neg_runs",    type=int,   default=20,
                   help="Number of negative-sampling rounds for AUC/AP stats")
    p.add_argument("--out",         default=None,
                   help="Optional path to save results JSON")
    p.add_argument("--label",       default="",
                   help="Label shown in printed header (e.g. model name / area)")
    return p.parse_args()


def main():
    args = get_args()

    scores_df = pd.read_csv(args.scores_csv)
    scores_df = _standardize_xy(scores_df)
    if "anomaly_score" not in scores_df.columns:
        sys.exit("ERROR: scores_csv must contain an 'anomaly_score' column.")
    scores_df = scores_df.dropna(subset=["anomaly_score"])

    dep_df = pd.read_csv(args.deposits)
    dep_df = _standardize_xy(dep_df)

    samp_xy     = scores_df[["x", "y"]].to_numpy(dtype=float)
    samp_scores = scores_df["anomaly_score"].to_numpy(dtype=float)
    dep_xy      = dep_df[["x", "y"]].to_numpy(dtype=float)

    results = evaluate(
        samp_xy, samp_scores, dep_xy,
        radius_km=args.radius_km,
        min_dist=args.min_dist,
        neg_runs=args.neg_runs,
        label=args.label,
    )

    if args.out:
        import json
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved → {args.out}")


if __name__ == "__main__":
    main()
