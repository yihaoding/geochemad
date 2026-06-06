#!/usr/bin/env python3
"""
run_interpolation.py  –  IDW spatial interpolation → PNG patches for VAE training.

Pipeline
--------
  1. Load area CSV → apply half-DL → CLR → StandardScaler
  2. Detect/use resolution: round(median_NN_distance) metres per pixel
  3. IDW-interpolate each element to a regular grid
  4. Save each element grid as PNG tiles (16×16 patches)
  5. Build deposit pixel-label map and save dataset_split.pkl

Patch size = average sample spacing in metres so that each pixel ≈ one sample.
16×16 patches give the model a neighbourhood context of ~16 sample spacings.

Outputs
-------
  <out_root>/<area_id>/vae_patches/
    patches_<elem>/patches/patch_RR_CC.png   (one folder per element)
    dataset_split.pkl                         (shared across elements)
    metadata.json                             (grid parameters for inverse mapping)

Usage
-----
  python run_interpolation.py --areas area9_dh1_au
  python run_interpolation.py --areas area9_dh1_au,area10_dh2_ag --resolution_m 100
  python run_interpolation.py --areas all
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SCRIPTS)
from shared.area_config   import resolve_areas
from shared.feature_select import FS_CHOICES, FS_SUFFIX, select_raw_cols
from shared.preprocess     import half_dl, clr_transform, EXCLUDE_COLS
from shared.evaluate       import _standardize_xy


PATCH_SIZE   = 16
IDW_K        = 8
IDW_POWER    = 2
DEPOSIT_R_M  = 2000   # within 2 km → label 1
DEPOSIT_BUF_M = 5000  # 2–5 km → label -1 (ambiguous)


# ── Coordinate helpers ────────────────────────────────────────────────────────

def lonlat_to_m(lons, lats, mean_lat_rad):
    """Equirectangular approximation. Good for areas < 500 km wide."""
    xm = lons * np.cos(mean_lat_rad) * 111000.0
    ym = lats * 111000.0
    return xm.astype(np.float32), ym.astype(np.float32)


def _sanitize(name: str) -> str:
    """Make element column names filesystem-safe."""
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")


# ── Main interpolation ────────────────────────────────────────────────────────

def interpolate_area(area_cfg: dict, out_root: str,
                     resolution_m: float = None,
                     patch_size: int = PATCH_SIZE,
                     fs: str = "none",
                     pca_top_k: int = 15):
    area_id = area_cfg["area_id"]
    print(f"\n{'='*60}\n  Interpolate  {area_id}  (fs={fs})\n{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(area_cfg["csv_path"], low_memory=False).dropna(subset=["X", "Y"])

    def _auto_detect():
        return [c for c in df.columns
                if c not in EXCLUDE_COLS and df[c].dtype != object]

    raw_cols = select_raw_cols(area_cfg, df.columns, fs=fs,
                               pca_top_k=pca_top_k, auto_detect=_auto_detect)

    df = half_dl(df, raw_cols)
    lons = df["X"].to_numpy(float)
    lats = df["Y"].to_numpy(float)
    mean_lat_rad = float(np.radians(lats.mean()))

    xm, ym = lonlat_to_m(lons, lats, mean_lat_rad)

    # ── Auto-resolution ───────────────────────────────────────────────────────
    if resolution_m is None:
        pts  = np.column_stack([xm, ym])
        tree = KDTree(pts)
        d, _ = tree.query(pts, k=2)
        nn   = np.median(d[:, 1])
        # Round to nearest 10 m, minimum 10 m
        resolution_m = max(10.0, round(nn / 10.0) * 10.0)
        print(f"  Auto resolution: {resolution_m:.0f} m  (median NN={nn:.1f} m)")
    else:
        print(f"  Resolution: {resolution_m:.0f} m  (user-specified)")

    # ── Build grid ────────────────────────────────────────────────────────────
    pad   = resolution_m * 2
    x0    = xm.min() - pad;  x1 = xm.max() + pad
    y0    = ym.min() - pad;  y1 = ym.max() + pad

    # Round grid dimensions up to nearest multiple of patch_size
    nx_raw = int(np.ceil((x1 - x0) / resolution_m)) + 1
    ny_raw = int(np.ceil((y1 - y0) / resolution_m)) + 1
    nx = int(np.ceil(nx_raw / patch_size)) * patch_size
    ny = int(np.ceil(ny_raw / patch_size)) * patch_size

    grid_xs = x0 + np.arange(nx) * resolution_m   # [nx]
    grid_ys = y0 + np.arange(ny) * resolution_m   # [ny]

    n_rows = ny // patch_size
    n_cols = nx // patch_size
    print(f"  Grid: {nx}×{ny} px → {n_rows}×{n_cols} patches  "
          f"({nx*resolution_m/1000:.1f}×{ny*resolution_m/1000:.1f} km)")

    # ── Precompute IDW weights for all grid pixels ────────────────────────────
    GX, GY    = np.meshgrid(grid_xs, grid_ys)    # [ny, nx]
    query_pts = np.column_stack([GX.ravel(), GY.ravel()])   # [ny*nx, 2]
    src_pts   = np.column_stack([xm, ym])
    tree      = KDTree(src_pts)

    d, ind = tree.query(query_pts, k=min(IDW_K, len(xm)))
    d      = np.maximum(d, 1e-6).astype(np.float32)
    w      = (1.0 / d ** IDW_POWER)
    w     /= w.sum(axis=1, keepdims=True)         # [ny*nx, K]

    # ── CLR + StandardScaler on all samples ──────────────────────────────────
    X_raw  = df[raw_cols].to_numpy(dtype=np.float32)
    X_clr  = clr_transform(X_raw)
    scaler = StandardScaler().fit(X_clr)
    X_sc   = scaler.transform(X_clr).astype(np.float32)

    # ── Interpolate each element → PNG patches ────────────────────────────────
    out_dir = os.path.join(out_root, area_id, f"vae_patches{FS_SUFFIX[fs]}")
    os.makedirs(out_dir, exist_ok=True)

    for ei, col in enumerate(raw_cols):
        vals        = X_sc[:, ei]                          # [N]
        interp_flat = (w * vals[ind]).sum(axis=1)          # [ny*nx]
        grid        = interp_flat.reshape(ny, nx)          # [ny, nx]

        # Map to [0, 255]: assume values in ~[-4, 4] std
        img8 = np.clip((grid + 4.0) / 8.0 * 255.0, 0, 255).astype(np.uint8)

        short    = _sanitize(col)
        tile_dir = os.path.join(out_dir, f"patches_{short}", "patches")
        os.makedirs(tile_dir, exist_ok=True)

        for r in range(n_rows):
            for c in range(n_cols):
                tile = img8[r*patch_size:(r+1)*patch_size,
                            c*patch_size:(c+1)*patch_size]
                Image.fromarray(tile, mode="L").save(
                    os.path.join(tile_dir, f"patch_{r:02d}_{c:02d}.png"))

        if (ei + 1) % max(1, len(raw_cols) // 10) == 0 or ei == len(raw_cols) - 1:
            print(f"  [{ei+1}/{len(raw_cols)}] {col}")

    # ── Deposit pixel labels ──────────────────────────────────────────────────
    dep_df  = pd.read_csv(area_cfg["deposits"])
    dep_df  = _standardize_xy(dep_df)
    dep_xm, dep_ym = lonlat_to_m(
        dep_df["x"].to_numpy(), dep_df["y"].to_numpy(), mean_lat_rad)

    pixel_label = np.zeros((ny, nx), dtype=np.int8)   # default: normal

    if len(dep_xm) > 0:
        dep_pts = np.column_stack([dep_xm, dep_ym])
        dtree   = KDTree(dep_pts)
        d_dep, _ = dtree.query(query_pts, k=1)
        d_dep   = d_dep.ravel().reshape(ny, nx)

        pixel_label[d_dep > DEPOSIT_BUF_M]  = 0   # normal (>5 km)
        pixel_label[(d_dep > DEPOSIT_R_M) &
                    (d_dep <= DEPOSIT_BUF_M)] = -1  # ambiguous
        pixel_label[d_dep <= DEPOSIT_R_M]   = 1   # deposit

    # ── dataset_split.pkl ─────────────────────────────────────────────────────
    all_patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            tile_lbl = pixel_label[r*patch_size:(r+1)*patch_size,
                                   c*patch_size:(c+1)*patch_size]
            if (tile_lbl == 1).any():
                lbl = 1
            elif (tile_lbl == -1).any():
                lbl = -1
            else:
                lbl = 0
            all_patches.append({
                "patch_id": r * n_cols + c,
                "row": r, "col": c,
                "image": f"patch_{r:02d}_{c:02d}.png",
                "label": lbl,
            })

    split = {
        "patch_size_pixels": patch_size,
        "image_size":        [ny, nx],
        "num_patches":       [n_rows, n_cols],
        "total_patches":     len(all_patches),
        "train":             all_patches,
        "pixel_labels":      [pixel_label],   # single split
        "raw_cols":          raw_cols,
    }
    split_path = os.path.join(out_dir, "dataset_split.pkl")
    with open(split_path, "wb") as f:
        pickle.dump(split, f)

    # ── Metadata JSON (for inverse pixel→sample mapping) ─────────────────────
    meta = {
        "area_id":           area_id,
        "feature_selection": fs,
        "resolution_m":      resolution_m,
        "patch_size":        patch_size,
        "image_size":        [ny, nx],
        "x0_m":              float(x0),
        "y0_m":              float(y0),
        "mean_lat_rad":      mean_lat_rad,
        "raw_cols":          raw_cols,
        "n_channels":        len(raw_cols),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Done → {out_dir}  ({len(all_patches)} patches, {len(raw_cols)} channels)")
    return out_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="IDW interpolation → VAE patches")
    p.add_argument("--areas",        default="all")
    p.add_argument("--out_root",     default="outputs")
    p.add_argument("--resolution_m", type=float, default=None,
                   help="Grid resolution in metres (default: auto from data)")
    p.add_argument("--patch_size",   type=int,   default=16)
    p.add_argument("--feature_selection", default="none", choices=FS_CHOICES,
                   help="none = all element columns (baseline), "
                        "llm = shared/llm_feature_selection.json subset, "
                        "pca = target + top-k correlated columns")
    p.add_argument("--pca_top_k", type=int, default=15,
                   help="Top-k for --feature_selection pca (target added on top)")
    return p.parse_args()


def main():
    args  = get_args()
    areas = resolve_areas(args.areas)
    print(f"Interpolating {len(areas)} area(s)  [fs={args.feature_selection}]")
    for area_cfg in areas:
        interpolate_area(area_cfg, args.out_root,
                         resolution_m=args.resolution_m,
                         patch_size=args.patch_size,
                         fs=args.feature_selection,
                         pca_top_k=args.pca_top_k)
    print("\nAll interpolations complete.")


if __name__ == "__main__":
    main()
