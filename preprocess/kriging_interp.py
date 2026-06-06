#!/usr/bin/env python3
"""
kriging_interp.py  –  Kriging interpolation baseline for geochemical data.
==========================================================================
Evaluates multiple kriging methods and variogram models on a spatially
stratified held-out set (default: 2000 points).

Kriging methods tried
---------------------
  Ordinary Kriging  (OK)  – no drift term, estimates the mean from data
  Universal Kriging (UK)  – linear spatial drift: uk_linear
                            quadratic drift:       uk_quadratic

Variogram models tuned
----------------------
  linear | power | gaussian | spherical | exponential

Outputs
-------
  <out_dir>/kriging_results_<element>.csv   per-element metric table
  <out_dir>/kriging_summary.csv             mean across elements per config
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

# ── optional pykrige ──────────────────────────────────────────────────────────
try:
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
    _PYKRIGE = True
except ImportError:
    _PYKRIGE = False

EXCLUDE_COLS = {"X", "Y", "LONGITUDE", "LATITUDE", "SAMPLE_ID",
                "LAB_ID", "SITE_ID", "DATASET_CODE", "SURVEY_ID",
                "SITE_CODE", "SPECIMEN_ID", "PROJECT_ID"}

SHARED_ELEMENTS = [
    "As", "Sb", "Ag", "W",  "Mo", "Bi",
    "Sn", "Nb", "Ta",
    "Au", "Pb", "Zn",
    "Co", "Cu", "Fe", "Mg", "Ca", "Ni",
]

# ── Kriging configurations to grid-search ─────────────────────────────────────
VARIOGRAM_MODELS = ["spherical", "exponential", "gaussian", "linear", "power"]

KRIGING_CONFIGS = []
for vmodel in VARIOGRAM_MODELS:
    KRIGING_CONFIGS.append({"method": "OK", "variogram_model": vmodel,
                             "drift_terms": None})
for vmodel in VARIOGRAM_MODELS:
    KRIGING_CONFIGS.append({"method": "UK_linear",    "variogram_model": vmodel,
                             "drift_terms": ["regional_linear"]})
for vmodel in VARIOGRAM_MODELS:
    KRIGING_CONFIGS.append({"method": "UK_quadratic", "variogram_model": vmodel,
                             "drift_terms": ["regional_linear", "specified"]})


# ─── Helpers ──────────────────────────────────────────────────────────────────

def resolve_shared_cols(df_cols: list, shared_names: list) -> list:
    result, seen = [], set()
    for name in shared_names:
        if name in df_cols and name not in seen:
            result.append(name); seen.add(name); continue
        matches = [c for c in df_cols if c.startswith(name + "_")]
        if matches and matches[0] not in seen:
            result.append(matches[0]); seen.add(matches[0])
        elif not matches:
            print(f"[warn] shared element '{name}' not found in CSV")
    return result


def half_detection_limit(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replace < prefixed detection-limit values with half the detection limit."""
    df = df.copy()
    for col in cols:
        if df[col].dtype == object:
            mask = df[col].astype(str).str.startswith("<")
            df.loc[mask, col] = (
                pd.to_numeric(df.loc[mask, col].astype(str).str[1:],
                              errors="coerce") / 2.0
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def spatial_stratified_split(df: pd.DataFrame, val_size: int,
                              grid_n: int = 20, seed: int = 42):
    rng = np.random.default_rng(seed)
    x   = df["X"].to_numpy(dtype=float)
    y   = df["Y"].to_numpy(dtype=float)
    n   = len(df)
    val_size = min(val_size, n - max(10, 64))

    x_bins = np.linspace(x.min(), x.max() + 1e-6, grid_n + 1)
    y_bins = np.linspace(y.min(), y.max() + 1e-6, grid_n + 1)
    cx = np.clip(np.digitize(x, x_bins) - 1, 0, grid_n - 1)
    cy = np.clip(np.digitize(y, y_bins) - 1, 0, grid_n - 1)
    cell_id = cx * grid_n + cy

    unique_cells, counts = np.unique(cell_id, return_counts=True)
    val_idx = []
    for cell, count in zip(unique_cells, counts):
        cell_pts = np.where(cell_id == cell)[0]
        n_pick   = max(1, round(val_size * count / n))
        n_pick   = min(n_pick, len(cell_pts))
        val_idx.extend(rng.choice(cell_pts, size=n_pick, replace=False).tolist())

    val_idx = np.array(val_idx, dtype=int)
    if len(val_idx) > val_size:
        val_idx = rng.choice(val_idx, size=val_size, replace=False)
    elif len(val_idx) < val_size:
        remaining = np.setdiff1d(np.arange(n), val_idx)
        extra     = min(val_size - len(val_idx), len(remaining))
        val_idx   = np.concatenate(
            [val_idx, rng.choice(remaining, size=extra, replace=False)])

    train_idx = np.setdiff1d(np.arange(n), val_idx)
    return train_idx, val_idx


def interp_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rmse   = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae    = float(np.mean(np.abs(yt - yp)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2     = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    r      = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float("nan")
    return rmse, mae, float(r2), r


def run_kriging(x_train, y_train, z_train,
                x_val, y_val,
                cfg: dict,
                n_closest: int = 64,
                enable_statistics: bool = False):
    """
    Run one kriging configuration on a single element.

    Parameters
    ----------
    x_train, y_train : 1-D arrays of training coordinates
    z_train          : 1-D array of training values
    x_val, y_val     : 1-D arrays of query coordinates
    cfg              : dict with keys: method, variogram_model, drift_terms
    n_closest        : number of nearest training points used per prediction

    Returns
    -------
    z_pred : 1-D array of predicted values at validation points
    """
    vmodel = cfg["variogram_model"]
    method = cfg["method"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if method == "OK":
                krig = OrdinaryKriging(
                    x_train, y_train, z_train,
                    variogram_model=vmodel,
                    enable_plotting=False,
                    verbose=False,
                    enable_statistics=enable_statistics,
                )
                z_pred, _ = krig.execute(
                    "points", x_val, y_val,
                    n_closest_points=n_closest,
                    backend="vectorized",
                )
            else:
                drift_terms = cfg["drift_terms"]
                krig = UniversalKriging(
                    x_train, y_train, z_train,
                    variogram_model=vmodel,
                    drift_terms=drift_terms,
                    enable_plotting=False,
                    verbose=False,
                )
                z_pred, _ = krig.execute(
                    "points", x_val, y_val,
                    n_closest_points=n_closest,
                    backend="vectorized",
                )
            return np.asarray(z_pred)
        except Exception as exc:
            print(f"    [skip] {method}/{vmodel} failed: {exc}")
            return np.full(len(x_val), np.nan)


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="Kriging interpolation baseline for geochemical data")
    p.add_argument("--csv_path",         required=True)
    p.add_argument("--shared_elements",  default=None,
                   help="Comma-separated base element names. Default: built-in 18.")
    p.add_argument("--val_size",         type=int, default=2000,
                   help="Number of held-out validation points (default: 2000)")
    p.add_argument("--val_grid_n",       type=int, default=20,
                   help="Grid size for spatial stratified split")
    p.add_argument("--n_closest",        type=int, default=64,
                   help="Number of nearest training points used per kriging prediction")
    p.add_argument("--variogram_models", default=None,
                   help="Comma-separated subset of variogram models to try. "
                        "Default: all (spherical,exponential,gaussian,linear,power)")
    p.add_argument("--methods",          default=None,
                   help="Comma-separated subset of methods to try. "
                        "Default: all (OK,UK_linear,UK_quadratic)")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--out_dir",          default="outputs/kriging")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not _PYKRIGE:
        print("[error] pykrige not installed. Run: pip install pykrige")
        sys.exit(1)

    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"[kriging] Loading {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=["X", "Y"])
    numeric_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and df[c].dtype != object]

    # ── Resolve shared element columns ────────────────────────────────────────
    shared_names = (
        [s.strip() for s in args.shared_elements.split(",")]
        if args.shared_elements else SHARED_ELEMENTS
    )
    shared_cols = resolve_shared_cols(numeric_cols, shared_names)
    print(f"[kriging] {len(shared_cols)} element cols: {shared_cols}")

    df = half_detection_limit(df, shared_cols)

    # ── Spatial stratified split ──────────────────────────────────────────────
    train_idx, val_idx = spatial_stratified_split(
        df, val_size=args.val_size, grid_n=args.val_grid_n, seed=args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    print(f"[kriging] split → train={len(train_df)}  val={len(val_df)}")

    x_train = train_df["X"].to_numpy(dtype=np.float64)
    y_train = train_df["Y"].to_numpy(dtype=np.float64)
    x_val   = val_df["X"].to_numpy(dtype=np.float64)
    y_val   = val_df["Y"].to_numpy(dtype=np.float64)

    # ── Filter configs by user selection ─────────────────────────────────────
    configs = list(KRIGING_CONFIGS)
    if args.variogram_models:
        keep_vm = set(args.variogram_models.split(","))
        configs = [c for c in configs if c["variogram_model"] in keep_vm]
    if args.methods:
        keep_m = set(args.methods.split(","))
        configs = [c for c in configs if c["method"] in keep_m]

    print(f"[kriging] Testing {len(configs)} configurations × "
          f"{len(shared_cols)} elements")
    print(f"[kriging] n_closest={args.n_closest}")
    print()

    # ── Per-element grid search ───────────────────────────────────────────────
    all_summary_rows = []

    for col in shared_cols:
        z_train_raw = train_df[col].to_numpy(dtype=np.float64)
        z_val_raw   = val_df[col].to_numpy(dtype=np.float64)

        # log-transform for heavy-tailed geochemical data
        pos_mask = z_train_raw > 0
        if pos_mask.sum() < 10:
            print(f"  [{col}] too few positive values — skipping")
            continue

        z_min   = z_train_raw[pos_mask].min()
        z_train = np.where(pos_mask, np.log1p(z_train_raw), np.log1p(z_min / 2))
        z_val_log = np.where(z_val_raw > 0,
                             np.log1p(z_val_raw),
                             np.log1p(z_min / 2))

        print(f"  ── Element: {col} (n_train_valid={pos_mask.sum()}) "
              f"{'─' * max(0, 40 - len(col))}")

        elem_rows = []
        best_rmse = float("inf")
        best_cfg  = None

        for cfg in configs:
            tag = f"{cfg['method']}/{cfg['variogram_model']}"
            z_pred_log = run_kriging(
                x_train[pos_mask], y_train[pos_mask], z_train[pos_mask],
                x_val, y_val,
                cfg=cfg,
                n_closest=min(args.n_closest, pos_mask.sum()),
            )
            # back-transform
            z_pred = np.expm1(z_pred_log)
            z_true = z_val_raw

            rmse, mae, r2, r = interp_metrics(z_true, z_pred)
            row = {
                "element":         col,
                "method":          cfg["method"],
                "variogram_model": cfg["variogram_model"],
                "rmse":            rmse,
                "mae":             mae,
                "r2":              r2,
                "r":               r,
                "n_val":           int(np.isfinite(z_true).sum()),
            }
            elem_rows.append(row)
            all_summary_rows.append(row)

            status = f"RMSE={rmse:8.4f}  MAE={mae:8.4f}  R²={r2:7.4f}  r={r:7.4f}"
            marker = " *" if (np.isfinite(rmse) and rmse < best_rmse) else ""
            print(f"    {tag:<35} {status}{marker}")
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_cfg  = tag

        if best_cfg:
            print(f"    >> Best: {best_cfg}  RMSE={best_rmse:.4f}")

        # Save per-element results
        elem_df = pd.DataFrame(elem_rows).sort_values("rmse")
        elem_path = os.path.join(args.out_dir, f"kriging_results_{col}.csv")
        elem_df.to_csv(elem_path, index=False)
        print()

    # ── Summary across all elements ───────────────────────────────────────────
    if not all_summary_rows:
        print("[kriging] No results produced.")
        return

    full_df = pd.DataFrame(all_summary_rows)

    summary = (
        full_df.groupby(["method", "variogram_model"])
        .agg(
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae",  "mean"),
            mean_r2=("r2",   "mean"),
            mean_r=("r",    "mean"),
            n_elements=("element", "nunique"),
        )
        .reset_index()
        .sort_values("mean_rmse")
    )

    summary_path = os.path.join(args.out_dir, "kriging_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("══ Overall summary (sorted by mean RMSE across all elements) ════════")
    print(f"  {'Method':<15} {'Variogram':<15} {'RMSE':>8} {'MAE':>8} "
          f"{'R²':>8} {'r':>8}  N_elem")
    for _, row in summary.iterrows():
        print(f"  {row['method']:<15} {row['variogram_model']:<15} "
              f"{row['mean_rmse']:8.4f} {row['mean_mae']:8.4f} "
              f"{row['mean_r2']:8.4f} {row['mean_r']:8.4f}  "
              f"{int(row['n_elements'])}")

    best = summary.iloc[0]
    print(f"\n  >> Best overall: {best['method']}/{best['variogram_model']}  "
          f"mean RMSE={best['mean_rmse']:.4f}")
    print(f"\n[kriging] Results saved to: {args.out_dir}/")
    print(f"[kriging] Summary: {summary_path}")


if __name__ == "__main__":
    main()
