#!/usr/bin/env python3
"""
all_interp.py  –  Comprehensive spatial interpolation baseline for geochemical data.
=====================================================================================
Preprocessing follows pretrain_multi.py exactly so results are directly comparable
to the GeoTransformerMulti baseline:

  1. half_detection_limit  – replace non-positive / NaN with ½ × min_positive
                             (same function as geo_transformer.py)
  2. CLR + StandardScaler  – Centered Log-Ratio across ALL elements jointly,
                             then StandardScaler (fit on train only)
  3. Interpolation         – in the CLR+scaled space (same as pretrain target)
  4. Metrics               – RMSE / MAE / R² / r in CLR+scaled space

Methods
-------
  IDW          – Inverse Distance Weighting
                   power × {1, 2, 3}  ×  n_neighbors × {16, 32, 64}
  RBF          – Radial Basis Function (scipy RBFInterpolator)
                   kernel × {linear, thin_plate_spline, cubic}
                   (training points capped at --rbf_max_train; default 5000)
  KNN          – K-Nearest Neighbour regression (distance-weighted)
                   k × {1, 3, 5, 10}
  OK           – Ordinary Kriging
                   variogram × {spherical, exponential, gaussian, linear, power}
                   n_closest × {32, 64, 128}
                   (training points capped at --kriging_max_train; default 10000)
  UK_linear    – Universal Kriging (linear drift)
  UK_quadratic – Universal Kriging (quadratic drift)

Memory strategy
---------------
  * All transforms are computed jointly once, before the element loop.
  * Elements are processed one at a time; results are written to disk after
    each element; intermediate arrays are deleted and gc.collect() is called.
  * RBF and Kriging have separate training-point caps to prevent OOM.
  * --resume skips elements whose results_<col>.csv already exist.

Outputs
-------
  <out_dir>/results_<col>.csv   per-element metric table (CLR+scaled space)
  <out_dir>/summary.csv         mean across elements per config
"""

import argparse
import contextlib
import gc
import os
import signal
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

try:
    from scipy.interpolate import RBFInterpolator
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from pykrige.ok import OrdinaryKriging
    from pykrige.uk import UniversalKriging
    _PYKRIGE = True
except ImportError:
    _PYKRIGE = False


# ── Constants ─────────────────────────────────────────────────────────────────

EXCLUDE_COLS = {"X", "Y", "LONGITUDE", "LATITUDE", "SAMPLE_ID",
                "LAB_ID", "SITE_ID", "DATASET_CODE", "SURVEY_ID",
                "SITE_CODE", "SPECIMEN_ID", "PROJECT_ID",
                "SAMPLEID", "SAMPLETYPE", "WAMEX_A_NO", "COMPSAMPID"}

SHARED_ELEMENTS = [
    "As", "Sb", "Ag", "W",  "Mo", "Bi",
    "Sn", "Nb", "Ta",
    "Au", "Pb", "Zn",
    "Co", "Cu", "Fe", "Mg", "Ca", "Ni",
]

VARIOGRAM_MODELS = ["spherical", "exponential"]   # gaussian/power numerically unstable; linear redundant
N_CLOSEST_LIST   = [16, 32]                        # nc64/128 consistently worse: variogram range >> neighbourhood
KRIGING_METHODS  = ["OK", "UK_linear", "UK_quadratic"]

IDW_POWERS    = [1, 2, 3]
IDW_NEIGHBORS = [16, 32, 64]

RBF_KERNELS = ["linear", "thin_plate_spline", "cubic"]

KNN_K_LIST = [1, 3, 5, 10]

COKRIGING_VARIOGRAMS  = ["spherical", "exponential", "gaussian"]
COKRIGING_N_CLOSEST   = [32]


# ── Worker globals (set in parent before Pool fork; inherited via COW) ────────
# Large arrays are never pickled — forked workers share them copy-on-write.
_W_x_train = _W_y_train = _W_x_val = _W_y_val = None
_W_nn_dist = _W_nn_idx  = None
_W_X_train_scaled = _W_X_val_scaled = None
_W_rbf_max = _W_krig_max = _W_timeout = None


def _run_config_worker(packed):
    """Worker: run one config; all large arrays come from module globals (fork-shared)."""
    cfg, feat_idx, seed = packed
    rng      = np.random.default_rng(seed)
    z_train  = _W_X_train_scaled[:, feat_idx].astype(np.float64)
    z_val    = _W_X_val_scaled[:,   feat_idx].astype(np.float64)
    timed_out = False
    try:
        with _time_limit(_W_timeout):
            if cfg["family"] == "CoKriging":
                z_pred = run_cokriging(
                    _W_x_train, _W_y_train, _W_X_train_scaled, feat_idx,
                    _W_x_val, _W_y_val,
                    kriging_max_train=_W_krig_max, rng=rng,
                    nn_dist=_W_nn_dist, nn_idx=_W_nn_idx,
                    **cfg["params"],
                )
            else:
                z_pred = predict(
                    cfg, _W_x_train, _W_y_train, z_train, _W_x_val, _W_y_val,
                    rbf_max_train=_W_rbf_max, kriging_max_train=_W_krig_max,
                    rng=rng, nn_dist=_W_nn_dist, nn_idx=_W_nn_idx,
                )
    except _TimeoutError:
        timed_out = True
        z_pred = np.full(len(_W_x_val), np.nan)
    except Exception as exc:
        print(f"    [error] {cfg['tag']}: {exc}", flush=True)
        z_pred = np.full(len(_W_x_val), np.nan)
    rmse, mae, r2, r = interp_metrics(z_val, z_pred)
    return cfg["tag"], cfg["family"], float(rmse), float(mae), float(r2), float(r), timed_out


# ── Timeout infrastructure (SIGALRM, Linux only) ──────────────────────────────

class _TimeoutError(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _TimeoutError()

@contextlib.contextmanager
def _time_limit(seconds: int):
    """Raise _TimeoutError if the body takes longer than `seconds`."""
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── Preprocessing (matches geo_transformer.py / pretrain_multi.py) ────────────

def half_detection_limit(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replace non-positive / NaN values with ½ × minimum positive per column."""
    df = df.copy()
    for c in cols:
        arr = df[c].to_numpy(dtype=float)
        bad = ~np.isfinite(arr) | (arr <= 0)
        if bad.any():
            min_pos = arr[arr > 0].min() if (arr > 0).any() else 1e-6
            arr[bad] = min_pos / 2.0
            df[c] = arr
    return df


def clr_transform(X: np.ndarray) -> np.ndarray:
    """Centered Log-Ratio: log(x) - row_mean(log(x)).  X must be > 0."""
    log_X = np.log(np.clip(X, 1e-10, None)).astype(np.float64)
    return (log_X - log_X.mean(axis=1, keepdims=True)).astype(np.float32)


def fit_clr_scaler(X_train_raw: np.ndarray):
    """Fit CLR + StandardScaler on training data. Returns (scaler, X_train_scaled)."""
    X_clr   = clr_transform(X_train_raw)
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_clr)
    return scaler, X_scaled.astype(np.float32)


def apply_clr_scaler(X_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Apply CLR then the fitted scaler to new data."""
    X_clr = clr_transform(X_raw)
    return scaler.transform(X_clr).astype(np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def resolve_shared_cols(df_cols, shared_names):
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


def spatial_stratified_split(df, val_size, grid_n=20, seed=42):
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


def interp_metrics(y_true, y_pred):
    """Metrics in CLR+scaled space (same units as pretrain evaluation)."""
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


# ── Method implementations ────────────────────────────────────────────────────

def run_idw(x_train, y_train, z_train, x_val, y_val, power=2, n_neighbors=32):
    try:
        coords_train = np.column_stack([x_train, y_train])
        coords_val   = np.column_stack([x_val,   y_val])
        tree = KDTree(coords_train)
        k    = min(n_neighbors, len(x_train))
        dist, idx = tree.query(coords_val, k=k)
        dist = np.where(dist == 0, 1e-10, dist)
        weights = 1.0 / dist ** power
        weights /= weights.sum(axis=1, keepdims=True)
        return (weights * z_train[idx]).sum(axis=1).astype(np.float64)
    except Exception as exc:
        print(f"    [skip] IDW p={power} k={n_neighbors}: {exc}")
        return np.full(len(x_val), np.nan)


def run_rbf(x_train, y_train, z_train, x_val, y_val,
            kernel="thin_plate_spline", rbf_max_train=5000, rng=None):
    """RBF via scipy. Training points subsampled to rbf_max_train to limit N×N matrix."""
    if not _SCIPY:
        return np.full(len(x_val), np.nan)
    try:
        n = len(x_train)
        if n > rbf_max_train:
            if rng is None:
                rng = np.random.default_rng(0)
            idx = rng.choice(n, size=rbf_max_train, replace=False)
            x_tr, y_tr, z_tr = x_train[idx], y_train[idx], z_train[idx]
        else:
            x_tr, y_tr, z_tr = x_train, y_train, z_train

        coords_train = np.column_stack([x_tr, y_tr])
        coords_val   = np.column_stack([x_val, y_val])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # smoothing adds Tikhonov regularisation to the RBF system;
            # without it, global kernels on spatially heterogeneous data
            # produce astronomically ill-conditioned matrices.
            rbf    = RBFInterpolator(coords_train, z_tr, kernel=kernel,
                                     smoothing=1.0)
            z_pred = rbf(coords_val)
        return np.asarray(z_pred, dtype=np.float64)
    except Exception as exc:
        print(f"    [skip] RBF kernel={kernel}: {exc}")
        return np.full(len(x_val), np.nan)


def run_knn(x_train, y_train, z_train, x_val, y_val, k=5):
    try:
        coords_train = np.column_stack([x_train, y_train])
        coords_val   = np.column_stack([x_val,   y_val])
        knn = KNeighborsRegressor(
            n_neighbors=min(k, len(x_train)),
            weights="distance", algorithm="auto", n_jobs=1,
        )
        knn.fit(coords_train, z_train)
        return np.asarray(knn.predict(coords_val), dtype=np.float64)
    except Exception as exc:
        print(f"    [skip] KNN k={k}: {exc}")
        return np.full(len(x_val), np.nan)


def _spatial_subsample_idx(x, y, n_target, rng):
    """
    Grid-based spatial subsample: returns the selected indices.
    Picks one random point per cell of an √n_target × √n_target grid,
    ensuring spatial coverage across the full domain. Random sampling
    over-represents clustered survey areas and leaves sparse regions empty,
    causing the fitted variogram range to reflect survey logistics rather
    than true spatial correlation structure.
    """
    n = len(x)
    if n <= n_target:
        return np.arange(n, dtype=int)
    grid_n = max(2, int(np.ceil(np.sqrt(n_target))))
    x_bins = np.linspace(x.min() - 1e-6, x.max() + 1e-6, grid_n + 1)
    y_bins = np.linspace(y.min() - 1e-6, y.max() + 1e-6, grid_n + 1)
    cx = np.clip(np.digitize(x, x_bins) - 1, 0, grid_n - 1)
    cy = np.clip(np.digitize(y, y_bins) - 1, 0, grid_n - 1)
    cell_id = cx * grid_n + cy
    idx = []
    for c in np.unique(cell_id):
        pts = np.where(cell_id == c)[0]
        idx.append(int(rng.choice(pts)))
    idx = np.array(idx, dtype=int)
    if len(idx) > n_target:
        idx = rng.choice(idx, size=n_target, replace=False)
    return idx


def _spatial_subsample_for_kriging(x, y, z, n_target, rng):
    """Convenience wrapper: returns (x, y, z) subsets."""
    idx = _spatial_subsample_idx(x, y, n_target, rng)
    return x[idx], y[idx], z[idx]


def _solve_ok_local(x_train, y_train, z_train, vfunc, vparams,
                    nn_dist, nn_idx):
    """Solve the Ordinary Kriging system for each val point using pre-queried
    full-data neighbors.  nn_dist/nn_idx shape: (n_val, nc)."""
    n_val, nc = nn_idx.shape
    z_pred = np.empty(n_val)
    for i in range(n_val):
        ni  = nn_idx[i]
        d_v = nn_dist[i]                        # (nc,) distances to val point
        x_n = x_train[ni]; y_n = y_train[ni]

        dx  = x_n[:, None] - x_n[None, :]
        dy  = y_n[:, None] - y_n[None, :]
        d_m = np.sqrt(dx * dx + dy * dy)        # (nc, nc) pairwise distances

        gam_m = vfunc(vparams, d_m.ravel()).reshape(nc, nc)
        np.fill_diagonal(gam_m, 0.0)            # γ(0) = 0 by definition
        gam_v = vfunc(vparams, d_v)             # (nc,)

        # Augmented OK system  [gam_m 1; 1ᵀ 0] [w; μ] = [gam_v; 1]
        # Small Tikhonov term (1e-6) prevents singular systems when many
        # neighbours fall within the nugget range (common in dense surveys).
        A = np.zeros((nc + 1, nc + 1))
        A[:nc, :nc] = gam_m + 1e-6 * np.eye(nc)
        A[:nc,  nc] = 1.0
        A[nc,  :nc] = 1.0
        b = np.zeros(nc + 1)
        b[:nc] = gam_v
        b[nc]  = 1.0

        try:
            w = np.linalg.solve(A, b)[:nc]
            z_pred[i] = float(w @ z_train[ni])
        except np.linalg.LinAlgError:
            z_pred[i] = np.nan
    return z_pred


def run_kriging(x_train, y_train, z_train, x_val, y_val,
                method="OK", variogram_model="spherical",
                drift_terms=None, n_closest=64,
                kriging_max_train=10000, rng=None,
                nn_dist=None, nn_idx=None):
    """Local Kriging: variogram fitted on a random subsample; predictions use
    the true n_closest neighbors from the FULL training set (nn_dist/nn_idx,
    pre-queried by the caller).  Falls back to pykrige execute() for UK."""
    if not _PYKRIGE:
        return np.full(len(x_val), np.nan)
    try:
        n = len(x_train)
        if rng is None:
            rng = np.random.default_rng(0)
        # Spatial grid subsample ensures variogram is fitted from a
        # spatially representative sample (not random, which over-represents
        # clustered survey areas and underestimates near-range structure).
        x_tr, y_tr, z_tr = _spatial_subsample_for_kriging(
            x_train, y_train, z_train, kriging_max_train, rng)

        nc = min(n_closest, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "OK":
                krig = OrdinaryKriging(
                    x_tr, y_tr, z_tr,
                    variogram_model=variogram_model,
                    nlags=10,    # default 6 is too coarse for geochemical data
                    weight=True, # weight by pair count per lag bin
                    enable_plotting=False,
                    verbose=False,
                )
                # Local OK: solve manually using full-data neighbors
                if nn_dist is not None and nn_idx is not None:
                    nd = nn_dist[:, :nc]
                    ni = nn_idx[:, :nc]
                else:
                    # Fallback: build KDTree locally (slower but correct)
                    tree = KDTree(np.column_stack([x_train, y_train]))
                    nd, ni = tree.query(np.column_stack([x_val, y_val]), k=nc)
                z_pred = _solve_ok_local(
                    x_train, y_train, z_train,
                    krig.variogram_function, krig.variogram_model_parameters,
                    nd, ni,
                )
            else:
                # Universal Kriging: fall back to pykrige execute (complex drift)
                krig = UniversalKriging(
                    x_tr, y_tr, z_tr,
                    variogram_model=variogram_model,
                    drift_terms=drift_terms,
                    enable_plotting=False,
                    verbose=False,
                )
                z_pred, _ = krig.execute(
                    "points", x_val, y_val,
                    n_closest_points=min(nc, len(x_tr)),
                    backend="loop",
                )
        return np.asarray(z_pred, dtype=np.float64)
    except Exception as exc:
        print(f"    [skip] {method}/{variogram_model}/nc={n_closest}: {exc}")
        return np.full(len(x_val), np.nan)


def run_cokriging(x_train, y_train, X_train_scaled, target_idx,
                  x_val, y_val, variogram_model="spherical", n_closest=64,
                  kriging_max_train=10000, rng=None,
                  nn_dist=None, nn_idx=None):
    """
    Regression Kriging (Co-Kriging variant).

    Step 1 — trend: Ridge regression on the element most correlated with the
             target (by |Pearson r| in training data).  Secondary values at
             val locations are estimated via IDW (k=32, power=2) from FULL data.
    Step 2 — residuals: Local OrdinaryKriging on the full-training residuals
             using pre-queried full-data neighbors (nn_dist/nn_idx).
    Prediction = trend(secondary_val) + kriging(residuals, coords_val).
    """
    if not _PYKRIGE:
        return np.full(len(x_val), np.nan)
    try:
        from sklearn.linear_model import Ridge

        z_train_full = X_train_scaled[:, target_idx].astype(np.float64)
        n_feats = X_train_scaled.shape[1]
        n       = len(x_train)

        # Find best secondary element by |Pearson r| in training data
        sec_cols = [j for j in range(n_feats) if j != target_idx]
        corrs    = np.array([
            abs(np.corrcoef(z_train_full, X_train_scaled[:, j])[0, 1])
            for j in sec_cols
        ])
        best_sec = sec_cols[int(np.nanargmax(corrs))]
        sec_full = X_train_scaled[:, best_sec].astype(np.float64)

        # Estimate secondary variable at val locations via IDW from FULL training set
        coords_full = np.column_stack([x_train.astype(np.float32), y_train.astype(np.float32)])
        coords_va   = np.column_stack([x_val.astype(np.float32),   y_val.astype(np.float32)])
        tree_full   = KDTree(coords_full)
        k_idw       = min(32, n)
        dist_f, kidx_f = tree_full.query(coords_va, k=k_idw)
        dist_f  = np.where(dist_f == 0, 1e-10, dist_f)
        w_f     = 1.0 / dist_f ** 2
        w_f    /= w_f.sum(axis=1, keepdims=True)
        sec_va  = (w_f * sec_full[kidx_f]).sum(axis=1)

        # Spatial subsample for variogram fitting (O(N²) cost in pykrige).
        if rng is None:
            rng = np.random.default_rng(0)
        sub_idx = _spatial_subsample_idx(x_train, y_train, kriging_max_train, rng)
        x_tr    = x_train[sub_idx]; y_tr  = y_train[sub_idx]
        z_tr    = z_train_full[sub_idx]; sec_tr = sec_full[sub_idx]

        # Step 1: Ridge regression on secondary variable
        reg = Ridge(alpha=1.0)
        reg.fit(sec_tr.reshape(-1, 1), z_tr)
        # Residuals computed on ALL training points (not just subsample)
        resid_full = z_train_full - reg.predict(sec_full.reshape(-1, 1))
        resid_tr   = z_tr - reg.predict(sec_tr.reshape(-1, 1))
        trend_va   = reg.predict(sec_va.reshape(-1, 1))

        # Step 2: Local OK on full-training residuals
        nc = min(n_closest, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            krig = OrdinaryKriging(
                x_tr, y_tr, resid_tr,
                variogram_model=variogram_model,
                nlags=10,
                weight=True,
                enable_plotting=False,
                verbose=False,
            )
        if nn_dist is not None and nn_idx is not None:
            nd = nn_dist[:, :nc]
            ni = nn_idx[:, :nc]
        else:
            tree = KDTree(np.column_stack([x_train, y_train]))
            nd, ni = tree.query(np.column_stack([x_val, y_val]), k=nc)

        resid_pred = _solve_ok_local(
            x_train, y_train, resid_full,
            krig.variogram_function, krig.variogram_model_parameters,
            nd, ni,
        )
        return (trend_va + resid_pred).astype(np.float64)
    except Exception as exc:
        print(f"    [skip] CoKriging/{variogram_model}/nc={n_closest}: {exc}")
        return np.full(len(x_val), np.nan)


# ── Config builders ───────────────────────────────────────────────────────────

def build_configs(args):
    configs = []

    for power in IDW_POWERS:
        for nn in IDW_NEIGHBORS:
            configs.append({
                "family": "IDW",
                "tag":    f"IDW_p{power}_k{nn}",
                "params": {"power": power, "n_neighbors": nn},
            })

    if _SCIPY:
        for kernel in RBF_KERNELS:
            configs.append({
                "family": "RBF",
                "tag":    f"RBF_{kernel}",
                "params": {"kernel": kernel},
            })
    else:
        print("[warn] scipy not found — RBF methods skipped")

    for k in KNN_K_LIST:
        configs.append({
            "family": "KNN",
            "tag":    f"KNN_k{k}",
            "params": {"k": k},
        })

    if _PYKRIGE:
        for kmethod in KRIGING_METHODS:
            drift = (None if kmethod == "OK"
                     else ["regional_linear"] if kmethod == "UK_linear"
                     else ["regional_linear", "specified"])
            for vmodel in VARIOGRAM_MODELS:
                for nc in N_CLOSEST_LIST:
                    configs.append({
                        "family": kmethod,
                        "tag":    f"{kmethod}_{vmodel}_nc{nc}",
                        "params": {
                            "method":          kmethod,
                            "variogram_model": vmodel,
                            "drift_terms":     drift,
                            "n_closest":       nc,
                        },
                    })
        for vmodel in COKRIGING_VARIOGRAMS:
            for nc in COKRIGING_N_CLOSEST:
                configs.append({
                    "family": "CoKriging",
                    "tag":    f"CoKriging_{vmodel}_nc{nc}",
                    "params": {
                        "variogram_model": vmodel,
                        "n_closest":       nc,
                    },
                })
    else:
        print("[warn] pykrige not found — Kriging / CoKriging methods skipped")

    if args.families:
        keep = set(f.strip() for f in args.families.split(","))
        configs = [c for c in configs if c["family"] in keep]

    return configs


def predict(cfg, x_tr, y_tr, z_tr, x_va, y_va,
            rbf_max_train, kriging_max_train, rng,
            nn_dist=None, nn_idx=None):
    fam = cfg["family"]
    p   = cfg["params"]
    if fam == "IDW":
        return run_idw(x_tr, y_tr, z_tr, x_va, y_va, **p)
    if fam == "RBF":
        return run_rbf(x_tr, y_tr, z_tr, x_va, y_va,
                       rbf_max_train=rbf_max_train, rng=rng, **p)
    if fam == "KNN":
        return run_knn(x_tr, y_tr, z_tr, x_va, y_va, **p)
    # Kriging families — pass pre-queried full-data neighbors
    return run_kriging(x_tr, y_tr, z_tr, x_va, y_va,
                       kriging_max_train=kriging_max_train, rng=rng,
                       nn_dist=nn_dist, nn_idx=nn_idx, **p)


# ── Argument parsing ──────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="All interpolation methods with CLR preprocessing")
    csv_group = p.add_mutually_exclusive_group(required=True)
    csv_group.add_argument("--csv_path",   default=None,
                           help="Single geochemical CSV (original behaviour).")
    csv_group.add_argument("--csv_paths",  default=None,
                           help="Comma-separated list of geochemical CSVs. "
                                "Each area is evaluated independently; results "
                                "are aggregated into a combined summary.")
    p.add_argument("--shared_elements",    default=None,
                   help="Comma-separated base element names. Default: built-in 18.")
    p.add_argument("--val_size",           type=int, default=2000)
    p.add_argument("--val_grid_n",         type=int, default=20)
    p.add_argument("--seed",               type=int, default=42)
    p.add_argument("--transform",          default="clr", choices=["clr", "none"],
                   help="Feature transform: 'clr' (default, matches pretrain) "
                        "or 'none' (raw values, StandardScaler only)")
    p.add_argument("--families",           default=None,
                   help="Comma-separated subset: IDW, RBF, KNN, OK, "
                        "UK_linear, UK_quadratic. Default: all available.")
    p.add_argument("--rbf_max_train",      type=int, default=5000,
                   help="Max training points for RBFInterpolator (default: 5000)")
    p.add_argument("--kriging_max_train",  type=int, default=20000,
                   help="Max training points for Kriging variogram fitting "
                        "(default: 20000). Spatially subsampled if larger.")
    p.add_argument("--resume",             action="store_true",
                   help="Skip elements whose results_<col>.csv already exist.")
    p.add_argument("--n_workers",          type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or os.cpu_count(),
                   help="Parallel workers for config loop (default: SLURM_CPUS_PER_TASK "
                        "or os.cpu_count()).")
    p.add_argument("--timeout",            type=int, default=300,
                   help="Per-config timeout in seconds (default: 300). "
                        "Configs that exceed this are skipped with NaN results.")
    p.add_argument("--out_dir",            default="outputs/all_interp")
    return p.parse_args()


# ── Per-area runner ────────────────────────────────────────────────────────────

def run_one_area(csv_path: str, args, out_dir: str) -> list:
    """
    Run all interpolation methods on a single area CSV.
    Returns a list of result dicts (one per element × config).
    Saves per-element CSVs to out_dir.
    """
    rng = np.random.default_rng(args.seed)

    print(f"\n[interp] ══ Area: {os.path.basename(csv_path)} ══", flush=True)
    df = pd.read_csv(csv_path).dropna(subset=["X", "Y"])
    numeric_cols = [c for c in df.columns
                    if c not in EXCLUDE_COLS and df[c].dtype != object]

    shared_names = (
        [s.strip() for s in args.shared_elements.split(",")]
        if args.shared_elements else SHARED_ELEMENTS
    )
    shared_cols = resolve_shared_cols(numeric_cols, shared_names)
    print(f"[interp] {len(shared_cols)} element cols: {shared_cols}", flush=True)

    print("[interp] Applying half-detection-limit ...", flush=True)
    df = half_detection_limit(df, shared_cols)

    train_idx, val_idx = spatial_stratified_split(
        df, val_size=args.val_size, grid_n=args.val_grid_n, seed=args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    print(f"[interp] split → train={len(train_df)}  val={len(val_df)}", flush=True)

    X_train_raw = train_df[shared_cols].to_numpy(dtype=np.float32)
    X_val_raw   = val_df[shared_cols].to_numpy(dtype=np.float32)

    if args.transform == "clr":
        print("[interp] Applying CLR + StandardScaler (fit on train) ...", flush=True)
        scaler, X_train_scaled = fit_clr_scaler(X_train_raw)
        X_val_scaled = apply_clr_scaler(X_val_raw, scaler)
    else:
        print("[interp] Applying StandardScaler only (no CLR) ...", flush=True)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val_scaled   = scaler.transform(X_val_raw).astype(np.float32)

    x_train = train_df["X"].to_numpy(dtype=np.float64)
    y_train = train_df["Y"].to_numpy(dtype=np.float64)
    x_val   = val_df["X"].to_numpy(dtype=np.float64)
    y_val   = val_df["Y"].to_numpy(dtype=np.float64)

    del df, train_df, val_df, X_train_raw, X_val_raw
    gc.collect()

    configs = build_configs(args)
    print(f"[interp] {len(configs)} configs × {len(shared_cols)} elements  "
          f"({args.transform.upper()}+StandardScaler space)", flush=True)

    # Pre-build KDTree on full training coords once; reused by all Kriging configs.
    nn_dist_pre = nn_idx_pre = None
    kriging_families = {"OK", "UK_linear", "UK_quadratic", "CoKriging"}
    if _PYKRIGE and any(c["family"] in kriging_families for c in configs):
        _max_nc = max(N_CLOSEST_LIST)
        print(f"[interp] Pre-building KDTree (n={len(x_train)}) + "
              f"querying {_max_nc} neighbors for {len(x_val)} val points ...",
              flush=True)
        _tree = KDTree(np.column_stack([x_train, y_train]))
        nn_dist_pre, nn_idx_pre = _tree.query(
            np.column_stack([x_val, y_val]),
            k=min(_max_nc, len(x_train)),
        )
        del _tree
        gc.collect()
        print("[interp] KDTree pre-query done.", flush=True)

    # ── Set fork-shared globals before spawning Pool ──────────────────────────
    # Must use `global` so forked workers inherit from __main__, not a re-import.
    global _W_x_train, _W_y_train, _W_x_val, _W_y_val
    global _W_nn_dist, _W_nn_idx
    global _W_X_train_scaled, _W_X_val_scaled
    global _W_rbf_max, _W_krig_max, _W_timeout
    _W_x_train        = x_train
    _W_y_train        = y_train
    _W_x_val          = x_val
    _W_y_val          = y_val
    _W_nn_dist        = nn_dist_pre
    _W_nn_idx         = nn_idx_pre
    _W_X_train_scaled = X_train_scaled
    _W_X_val_scaled   = X_val_scaled
    _W_rbf_max        = args.rbf_max_train
    _W_krig_max       = args.kriging_max_train
    _W_timeout        = args.timeout

    n_workers = min(args.n_workers, len(configs))
    print(f"[interp] Parallel workers: {n_workers}", flush=True)

    all_rows = []

    with Pool(n_workers) as pool:
        for col_idx, col in enumerate(shared_cols, 1):
            out_path = os.path.join(out_dir, f"results_{col}.csv")

            if args.resume and os.path.exists(out_path):
                print(f"  [{col_idx}/{len(shared_cols)}] {col} — skipping (resume)",
                      flush=True)
                all_rows.extend(pd.read_csv(out_path).to_dict("records"))
                continue

            feat_idx = shared_cols.index(col)
            z_val_mean = float(X_val_scaled[:, feat_idx].mean())
            n_train    = X_train_scaled.shape[0]
            print(f"  ══ [{col_idx}/{len(shared_cols)}] {col}  "
                  f"(n_train={n_train}, val_mean={z_val_mean:.3f}) ══",
                  flush=True)

            tasks = [
                (cfg, feat_idx, args.seed + col_idx * 1000 + i)
                for i, cfg in enumerate(configs)
            ]
            results = pool.map(_run_config_worker, tasks)

            elem_rows = []
            best_rmse, best_tag = float("inf"), None
            n_val = int(np.isfinite(X_val_scaled[:, feat_idx]).sum())

            for (tag, family, rmse, mae, r2, r, timed_out), cfg in zip(results, configs):
                if timed_out:
                    print(f"    [timeout] {tag} > {args.timeout}s — skipped",
                          flush=True)
                marker = " *" if (np.isfinite(rmse) and rmse < best_rmse) else ""
                print(f"    {tag:<45} RMSE={rmse:7.4f} MAE={mae:7.4f} "
                      f"R²={r2:7.4f} r={r:7.4f}{marker}", flush=True)
                if np.isfinite(rmse) and rmse < best_rmse:
                    best_rmse, best_tag = rmse, tag
                elem_rows.append({
                    "area": os.path.basename(csv_path),
                    "element": col, "family": family, "config": tag,
                    "rmse": rmse, "mae": mae, "r2": r2, "r": r,
                    "n_val": n_val,
                })

            if best_tag:
                print(f"    >> Best: {best_tag}  RMSE={best_rmse:.4f}", flush=True)

            pd.DataFrame(elem_rows).sort_values("rmse").to_csv(out_path, index=False)
            all_rows.extend(elem_rows)
            print(f"    Saved → {out_path}\n", flush=True)
            gc.collect()

    return all_rows


def _print_summary(full_df: pd.DataFrame, summary_path: str, label: str):
    summary = (
        full_df.groupby(["family", "config"])
        .agg(mean_rmse=("rmse", "mean"), mean_mae=("mae", "mean"),
             mean_r2=("r2", "mean"), mean_r=("r", "mean"),
             n_elements=("element", "nunique"))
        .reset_index()
        .sort_values("mean_rmse")
    )
    summary.to_csv(summary_path, index=False)
    print(f"\n══ {label} (sorted by mean RMSE) ══════════════════════════")
    print(f"  {'Family':<14} {'Config':<45} {'RMSE':>7} {'MAE':>7} "
          f"{'R²':>7} {'r':>7}  N")
    for _, row in summary.iterrows():
        print(f"  {row['family']:<14} {row['config']:<45} "
              f"{row['mean_rmse']:7.4f} {row['mean_mae']:7.4f} "
              f"{row['mean_r2']:7.4f} {row['mean_r']:7.4f}  "
              f"{int(row['n_elements'])}")
    best = summary.iloc[0]
    print(f"\n  >> Best overall: {best['config']}  "
          f"mean RMSE={best['mean_rmse']:.4f}", flush=True)
    print(f"[interp] Summary: {summary_path}", flush=True)
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    csv_list = (
        [p.strip() for p in args.csv_paths.split(",")]
        if args.csv_paths else [args.csv_path]
    )

    all_rows = []

    if len(csv_list) == 1:
        # Single-area: original behaviour, results in out_dir directly
        rows = run_one_area(csv_list[0], args, args.out_dir)
        all_rows.extend(rows)
        if rows:
            full_df = pd.DataFrame(rows)
            _print_summary(full_df, os.path.join(args.out_dir, "summary.csv"),
                           "Overall summary (CLR+scaled space)")
    else:
        # Multi-area: each area gets its own sub-directory
        print(f"[interp] Multi-area mode: {len(csv_list)} area(s)", flush=True)
        for csv_path in csv_list:
            area_tag = os.path.splitext(os.path.basename(csv_path))[0]
            area_dir = os.path.join(args.out_dir, area_tag)
            os.makedirs(area_dir, exist_ok=True)
            rows = run_one_area(csv_path, args, area_dir)
            all_rows.extend(rows)

        if not all_rows:
            print("[interp] No results produced.")
            return

        full_df = pd.DataFrame(all_rows)

        # Per-area summaries
        for area_tag, grp in full_df.groupby("area"):
            area_dir = os.path.join(args.out_dir,
                                    os.path.splitext(area_tag)[0])
            _print_summary(grp, os.path.join(area_dir, "summary.csv"),
                           f"Summary for {area_tag}")

        # Combined summary across all areas
        _print_summary(full_df,
                       os.path.join(args.out_dir, "summary_all_areas.csv"),
                       "Combined summary — all areas")

    print(f"\n[interp] Results saved to: {args.out_dir}/", flush=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)
    main()
