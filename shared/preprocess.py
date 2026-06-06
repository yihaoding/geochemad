"""
preprocess.py  –  Unified geochemical data loading and preprocessing.

Public API
----------
  load_area(area_cfg, transform, val_frac, seed)
      → df_train, df_val, df_all, feature_cols, raw_cols, scaler

  save_scores(df_all, scores, out_dir)
      → path to anomaly_scores.csv

  run_evaluate(scores_csv, deposits_csv, out_dir, ...)
      → metrics dict  +  writes metrics.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Coordinate / metadata columns to exclude from element features
EXCLUDE_COLS = {
    "X", "Y", "SAMPLEID", "SAMPLETYPE", "WAMEX_A_NO", "COMPSAMPID",
    "Longitude", "Latitude",
}


# ── Core transforms ───────────────────────────────────────────────────────────

def half_dl(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replace <=0 / NaN values with half the column's minimum positive value."""
    df = df.copy()
    for c in cols:
        arr = df[c].to_numpy(dtype=float)
        bad = ~np.isfinite(arr) | (arr <= 0)
        if bad.any():
            pos = arr[arr > 0]
            arr[bad] = pos.min() / 2 if len(pos) else 1e-6
            df[c] = arr
    return df


def clr_transform(X: np.ndarray) -> np.ndarray:
    """Centered Log-Ratio.  X must be strictly positive → returns [N, D]."""
    lX = np.log(np.clip(X, 1e-10, None)).astype(np.float64)
    return (lX - lX.mean(axis=1, keepdims=True)).astype(np.float32)


def _helmert(D: int) -> np.ndarray:
    """Helmert sub-matrix V (D x D-1) for isometric log-ratio basis."""
    V = np.zeros((D, D - 1))
    for k in range(D - 1):
        c = 1.0 / np.sqrt((k + 1) * (k + 2))
        V[:k + 1, k] = c
        V[k + 1, k]  = -(k + 1) * c
    return V


def ilr_transform(X: np.ndarray) -> np.ndarray:
    """Isometric Log-Ratio.  X strictly positive → returns [N, D-1] in
    orthonormal log-ratio space (drops one dim vs CLR)."""
    D  = X.shape[1]
    if D < 2:
        # Cannot ILR with 1 element; fall back to CLR-like log-centred.
        return clr_transform(X)
    V  = _helmert(D)
    lX = np.log(np.clip(X, 1e-10, None)).astype(np.float64)
    return (lX @ V).astype(np.float32)


# ── Main loading function ─────────────────────────────────────────────────────

def load_area(area_cfg: dict,
              transform: str = "clr",
              val_frac: float = 0.05,
              seed: int = 42):
    """
    Load and preprocess one area.

    Parameters
    ----------
    area_cfg  : dict from AREAS in area_config.py
    transform : "clr" (default) | "ilr" | "none"
    val_frac  : fraction held out as validation for checkpoint selection
    seed      : RNG seed (controls the val split)

    Returns
    -------
    df_train     : 95% of data, feature cols replaced with scaled values
    df_val       : 5% of data, same
    df_all       : full data, same (for scoring all samples at inference)
    feature_cols : element column names (same as raw_cols after CLR)
    raw_cols     : original element column names before transform
    scaler       : fitted StandardScaler (call scaler.transform() on new data)
    """
    df = pd.read_csv(area_cfg["csv_path"], low_memory=False).dropna(subset=["X", "Y"])

    # ── Determine which columns are element features ──────────────────────────
    if area_cfg.get("input_elements"):
        raw_cols = [c for c in area_cfg["input_elements"] if c in df.columns]
        missing  = set(area_cfg["input_elements"]) - set(raw_cols)
        if missing:
            print(f"[warn] input_elements not in CSV: {sorted(missing)}")
    else:
        raw_cols = [
            c for c in df.columns
            if c not in EXCLUDE_COLS and df[c].dtype != object
        ]

    # Ensure target_element is included
    tgt = area_cfg.get("target_element")
    if tgt and tgt not in raw_cols and tgt in df.columns:
        raw_cols = [tgt] + raw_cols

    df = half_dl(df, raw_cols)

    # ── Val split (random, seed-fixed) ────────────────────────────────────────
    rng       = np.random.default_rng(seed)
    n         = len(df)
    val_idx   = rng.choice(n, size=max(1, int(n * val_frac)), replace=False)
    train_idx = np.setdiff1d(np.arange(n), val_idx)

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val   = df.iloc[val_idx  ].reset_index(drop=True)

    X_tr = df_train[raw_cols].to_numpy(dtype=np.float32)
    X_va = df_val  [raw_cols].to_numpy(dtype=np.float32)
    X_al = df      [raw_cols].to_numpy(dtype=np.float32)

    if transform == "clr":
        X_tr = clr_transform(X_tr)
        X_va = clr_transform(X_va)
        X_al = clr_transform(X_al)
    elif transform == "ilr":
        X_tr = ilr_transform(X_tr)
        X_va = ilr_transform(X_va)
        X_al = ilr_transform(X_al)

    scaler   = StandardScaler().fit(X_tr)
    X_tr_s   = scaler.transform(X_tr).astype(np.float32)
    X_va_s   = scaler.transform(X_va).astype(np.float32)
    X_al_s   = scaler.transform(X_al).astype(np.float32)

    # ILR drops one dimension; rename feature cols accordingly.
    if transform == "ilr":
        feature_cols = [f"ilr_{i}" for i in range(X_tr_s.shape[1])]
        # Drop original raw_cols from frames; new ilr_X cols replace them.
        df_train = df_train.drop(columns=[c for c in raw_cols if c in df_train.columns])
        df_val   = df_val  .drop(columns=[c for c in raw_cols if c in df_val.columns])
        df_all_base = df.drop(columns=[c for c in raw_cols if c in df.columns]).copy()
    else:
        feature_cols = list(raw_cols)  # column names unchanged
        df_all_base = df.copy()

    for i, c in enumerate(feature_cols):
        df_train[c] = X_tr_s[:, i]
        df_val  [c] = X_va_s[:, i]

    df_all = df_all_base
    for i, c in enumerate(feature_cols):
        df_all[c] = X_al_s[:, i]

    print(f"[load] {area_cfg['area_id']}  N={n}  "
          f"train={len(train_idx)}  val={len(val_idx)}  "
          f"features={len(feature_cols)}  transform={transform}")

    return df_train, df_val, df_all, feature_cols, raw_cols, scaler


# ── Output helpers ────────────────────────────────────────────────────────────

def save_scores(df_all: pd.DataFrame,
                scores: np.ndarray,
                out_dir: str,
                x_col: str = "X",
                y_col: str = "Y",
                fname: str = "anomaly_scores.csv") -> str:
    """Write anomaly scores CSV (columns: x, y, anomaly_score) and return path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    pd.DataFrame({
        "x":             df_all[x_col].to_numpy(),
        "y":             df_all[y_col].to_numpy(),
        "anomaly_score": scores,
    }).to_csv(path, index=False)
    return path


def run_evaluate(scores_csv: str,
                 deposits_csv: str,
                 out_dir: str,
                 radius_km: float = 2.0,
                 min_dist: float  = 0.5,
                 neg_runs: int    = 20,
                 label: str       = "",
                 metrics_fname: str = "metrics.json") -> dict:
    """
    Run spatial evaluation via shared/evaluate.py and save metrics JSON.
    Returns the metrics dict.
    """
    _shared = os.path.dirname(__file__)
    if _shared not in sys.path:
        sys.path.insert(0, _shared)
    from evaluate import evaluate, _standardize_xy   # noqa: E402

    scores_df = pd.read_csv(scores_csv)
    scores_df = _standardize_xy(scores_df)
    dep_df    = pd.read_csv(deposits_csv)
    dep_df    = _standardize_xy(dep_df)

    samp_xy     = scores_df[["x", "y"]].to_numpy(dtype=float)
    samp_scores = scores_df["anomaly_score"].to_numpy(dtype=float)
    dep_xy      = dep_df[["x", "y"]].to_numpy(dtype=float)

    results = evaluate(samp_xy, samp_scores, dep_xy,
                       radius_km=radius_km, min_dist=min_dist,
                       neg_runs=neg_runs, label=label)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, metrics_fname), "w") as f:
        json.dump(results, f, indent=2)
    return results
