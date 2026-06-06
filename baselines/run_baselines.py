#!/usr/bin/env python3
"""
run_baselines.py  –  Classical statistical baselines for geochemical anomaly detection.

Methods
-------
  zscore  : per-sample anomaly score = max |z-score| across CLR-transformed features
  mahal   : Mahalanobis distance from training centroid in feature space
  knn     : mean Euclidean distance to K nearest neighbors in feature space

Usage
-----
  # One area, one method
  python run_baselines.py --areas area9_dh1_au --method zscore

  # All areas, all methods
  python run_baselines.py --areas all --method all

  # Multiple areas
  python run_baselines.py --areas area9_dh1_au,area10_dh2_ag --method all
"""

import argparse
import os
import sys

import numpy as np

# ── Shared imports ────────────────────────────────────────────────────────────
_SCRIPTS = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SCRIPTS)
from shared.area_config import resolve_areas
from shared.preprocess  import load_area, save_scores, run_evaluate


# ── Method implementations ────────────────────────────────────────────────────

def score_zscore(X_all: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Max absolute z-score across features per sample."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    std[std < 1e-10] = 1e-10
    return np.abs((X_all - mean) / std).max(axis=1)


def score_mahal(X_all: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Mahalanobis distance from training centroid."""
    from scipy.spatial.distance import cdist
    mean = X_train.mean(axis=0, keepdims=True)
    cov  = np.cov(X_train.T)
    # Regularise covariance (avoid singular matrix)
    cov += np.eye(cov.shape[0]) * 1e-6
    try:
        VI = np.linalg.inv(cov)
        d  = cdist(X_all, mean, metric="mahalanobis", VI=VI).ravel()
    except np.linalg.LinAlgError:
        # Fall back to Euclidean if covariance is still singular
        print("  [warn] Mahalanobis: singular covariance, falling back to Euclidean")
        d = np.linalg.norm(X_all - mean, axis=1)
    return d


def score_knn(X_all: np.ndarray, X_train: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean distance to K nearest training neighbors in feature space."""
    from sklearn.neighbors import KDTree
    tree = KDTree(X_train)
    dist, _ = tree.query(X_all, k=min(k, len(X_train)))
    return dist.mean(axis=1)


def score_ocsvm(X_all: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """One-Class SVM with RBF kernel; score = -decision_function (higher = more anomalous)."""
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    clf.fit(X_train)
    # decision_function: higher = more inlier → negate so higher = more anomaly
    return -clf.decision_function(X_all).ravel()


def score_isoforest(X_all: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Isolation Forest; score = -decision_function (higher = more anomalous)."""
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(X_train)
    return -clf.decision_function(X_all).ravel()


# ── Per-area runner ───────────────────────────────────────────────────────────

METHODS = {
    "zscore":   score_zscore,
    "mahal":    score_mahal,
    "knn":      score_knn,
    "ocsvm":    score_ocsvm,
    "isoforest":score_isoforest,
}


def run_area(area_cfg: dict, methods: list, out_root: str,
             transform: str = "clr",
             radius_km: float = 2.0, min_dist: float = 0.5, neg_runs: int = 20,
             out_suffix: str = ""):
    area_id = area_cfg["area_id"]
    print(f"\n{'='*60}\n  {area_id}\n{'='*60}")

    df_train, df_val, df_all, feature_cols, raw_cols, scaler = load_area(
        area_cfg, transform=transform)

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32)
    X_all   = df_all  [feature_cols].to_numpy(dtype=np.float32)

    for method in methods:
        print(f"\n[{method}] scoring {len(X_all)} samples …")
        fn = METHODS[method]

        if method == "knn":
            scores = fn(X_all, X_train, k=5)
        else:
            scores = fn(X_all, X_train)

        out_dir   = os.path.join(out_root, area_id, method + out_suffix)
        scores_csv = save_scores(df_all, scores, out_dir)
        print(f"  saved → {scores_csv}")

        metrics = run_evaluate(
            scores_csv     = scores_csv,
            deposits_csv   = area_cfg["deposits"],
            out_dir        = out_dir,
            radius_km      = radius_km,
            min_dist       = min_dist,
            neg_runs       = neg_runs,
            label          = f"{area_id}/{method}",
        )
        print(f"  AUC={metrics['AUC_mean']:.4f}  AP={metrics['AP_mean']:.4f}  "
              f"SR@5%={metrics['SR_5pct']:.4f}  DTD={metrics['DTD_km']:.2f}km")


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="Classical baselines for GAD")
    p.add_argument("--areas",     default="all",
                   help="'all', single area_id, or comma-separated area_ids")
    p.add_argument("--method",    default="all",
                   choices=list(METHODS) + ["all"])
    p.add_argument("--out_root",  default="outputs")
    p.add_argument("--transform", default="clr", choices=["clr", "ilr", "none"])
    p.add_argument("--out_suffix", default="",
                   help="optional suffix appended to method subdir (e.g. '_ilr')")
    p.add_argument("--input_elements", default=None,
                   help="comma-separated override of area_cfg['input_elements'] "
                        "(used for FS ablation: PCA-15, Manual, etc.)")
    p.add_argument("--radius_km", type=float, default=2.0)
    p.add_argument("--min_dist",  type=float, default=0.5)
    p.add_argument("--neg_runs",  type=int,   default=20)
    return p.parse_args()


def main():
    args    = get_args()
    areas   = resolve_areas(args.areas)
    methods = list(METHODS) if args.method == "all" else [args.method]

    print(f"Running {methods} on {len(areas)} area(s)")
    for area_cfg in areas:
        if args.input_elements:
            # override per-area input_elements (e.g. PCA-15)
            area_cfg = dict(area_cfg)  # shallow copy so we don't mutate the global
            area_cfg["input_elements"] = [c.strip() for c in args.input_elements.split(",") if c.strip()]
        run_area(area_cfg, methods, args.out_root,
                 transform=args.transform,
                 radius_km=args.radius_km,
                 min_dist=args.min_dist,
                 neg_runs=args.neg_runs,
                 out_suffix=args.out_suffix)

    print("\nAll done.")


if __name__ == "__main__":
    main()
