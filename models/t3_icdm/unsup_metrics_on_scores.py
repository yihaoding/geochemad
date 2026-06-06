#!/usr/bin/env python3
"""
Compute unsupervised selection metrics (Moran's I, p99/p50, kurtosis, skewness)
on the anomaly_scores.csv of every ICDM variant, for every area available.

Lets us answer: under a FIXED unsupervised criterion (e.g. moran or p99/p50),
which backbone variant wins per area? Should match how VAE-family / T1
checkpoints were picked elsewhere in the project.

Output: prints a wide table area x variant x metric, plus winners.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurt, skew as _skew
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared"))
from area_config import AREAS  # noqa: E402

ROOT = Path(os.environ.get("GEOCHEM_OUT_ROOT", "outputs"))

VARIANTS = [
    "outputs_icdm_v3",
    "outputs_icdm_v3_aligned",
    "outputs_icdm_v3p1",
    "outputs_icdm_v3p1_clr",
    "outputs_icdm_v3p1_clr_fs",
    "outputs_icdm_v3p2",
    "outputs_icdm_v3p2_clr",
    "outputs_icdm_v3p2_clr_fs",
    "outputs_icdm_aligned_d03",
    "outputs_icdm_aligned_input",
]

SHORT2FULL = {
    "sed1":  "area1_sed1_au",
    "sed2":  "area2_sed2_cu",
    "sed3":  "area15_sed3_au",
    "sed4":  "area16_sed4_cu",
    "rock1": "area3_rock1_w",
    "rock2": "area4_rock2_au",
    "rock3": "area5_rock3_cu",
    "soil1": "area6_soil1_au",
    "soil2": "area7_soil2_au",
    "soil3": "area8_soil3_ni",
}
MORAN_K = 8


def morans_i(scores, coords, k=MORAN_K):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    idx = idx[:, 1:]
    z = scores - scores.mean()
    lag = z[idx].mean(axis=1)
    den = (z * z).sum()
    return float((z * lag).sum() / den) if den > 0 else 0.0


def unsup_metrics(scores, coords):
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if s.size < 10:
        return dict(moran=np.nan, p99_p50=np.nan, kurtosis=np.nan, skewness=np.nan)
    p99 = float(np.percentile(s, 99))
    p50 = float(np.median(s))
    return dict(
        moran    = morans_i(np.asarray(scores, float), coords),
        p99_p50  = p99 / max(p50, 1e-12),
        kurtosis = float(_kurt(s, fisher=True, bias=False)),
        skewness = float(_skew(s, bias=False)),
    )


def main():
    rows = []
    for v in VARIANTS:
        vdir = ROOT / v
        if not vdir.exists():
            continue
        for short in SHORT2FULL:
            csv = vdir / short / "t1" / "anomaly_scores.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(csv)
            x = df.get("x", df.get("X")).to_numpy(float)
            y = df.get("y", df.get("Y")).to_numpy(float)
            coords = np.stack([x, y], axis=1)
            s = df["anomaly_score"].to_numpy(float)
            m = unsup_metrics(s, coords)
            m.update(variant=v.replace("outputs_icdm_", ""), area=short)
            rows.append(m)
    df = pd.DataFrame(rows)
    cols = ["variant", "area", "moran", "p99_p50", "kurtosis", "skewness"]
    df = df[cols]
    pd.set_option("display.float_format", "{:.4f}".format)

    # Wide: area x variant for each metric
    for metric in ("moran", "p99_p50"):
        piv = df.pivot(index="variant", columns="area", values=metric)
        cols = [c for c in ["sed1","sed2","sed3","sed4","rock1","rock2",
                            "rock3","soil1","soil2","soil3"] if c in piv.columns]
        piv = piv[cols]
        print(f"\n=== {metric} (HIGHER = more spatially clustered / heavy-tailed) ===")
        print(piv.to_string(float_format="%.4f"))
        # Winners per area
        winners = piv.idxmax(axis=0)
        print("  winner:", dict(winners))

    out_csv = ROOT / "reports/tables/icdm_variant_unsup_metrics.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    main()
