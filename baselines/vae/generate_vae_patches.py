#!/usr/bin/env python3
"""
generate_vae_patches.py  –  IDW interpolation → VAE-compatible patch dataset.

Output structure (matches train_vae.py):
  <out_root>/<area_id>/vae_patches{suffix}/
      dataset_split.pkl
      metadata.json
      patches_<elem>/patches/patch_RR_CC.png   (one dir per element)

`{suffix}` is set by --feature_selection (see shared/feature_select.py):
    none → ""        (every element column;            baseline)
    llm  → "_fs"     (shared/llm_feature_selection.json subset)
    pca  → "_pca"    (target + top-k correlated columns)

Usage
-----
  python generate_vae_patches.py --areas all
  python generate_vae_patches.py --areas area1_sed1_au,area2_sed2_cu
  python generate_vae_patches.py --areas area1_sed1_au --resolution_m 500
  python generate_vae_patches.py --areas all --feature_selection llm --resume
  python generate_vae_patches.py --areas all --feature_selection pca --pca_top_k 15
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "preprocess", "_vendor"))

from shared.area_config import resolve_areas
from shared.feature_select import FS_CHOICES, FS_SUFFIX, select_raw_cols
import preprocessing.interpolation as _interp_mod
from preprocessing.interpolation import (
    interpolate_and_annotate_with_patches,
    split_image_into_patches_and_prepare_dataset,
)
from preprocessing.preprocessor import get_df_element
from skbio.stats.composition import clr, ilr

# Override geo_dataset's GDA2020-only helper with a globally valid WGS84 UTM version
def _wgs84_utm(lat: float, lon: float):
    from pyproj import CRS
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)

_interp_mod.get_utm_crs_from_latlon = _wgs84_utm


PATCH_SIZE = 16


def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")


def _to_utm(lons: np.ndarray, lats: np.ndarray):
    """Project geographic coords to WGS84 UTM (works globally)."""
    from pyproj import CRS, Transformer
    lon_mean = float(lons.mean())
    lat_mean = float(lats.mean())
    zone = int((lon_mean + 180) / 6) + 1
    epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    utm_crs = CRS.from_epsg(epsg)
    tr = Transformer.from_crs(CRS.from_epsg(4326), utm_crs, always_xy=True)
    xm, ym = tr.transform(lons, lats)
    return xm, ym


def _auto_resolution(df: pd.DataFrame,
                     max_patches: int = 4096,
                     min_m: float = 10.0) -> float:
    """
    Resolution = median nearest-neighbour distance in UTM metres.
    Scaled up if the resulting grid would produce more than max_patches patches.
    """
    from sklearn.neighbors import KDTree

    lons = df["X"].values.astype(float)
    lats = df["Y"].values.astype(float)
    xm, ym = _to_utm(lons, lats)

    pts = np.column_stack([xm, ym])
    _, uid = np.unique(pts, axis=0, return_index=True)
    pts_uniq = pts[uid]

    tree = KDTree(pts_uniq)
    d, _ = tree.query(pts_uniq, k=2)
    nn = float(np.median(d[:, 1]))

    # round to 1 significant figure
    mag = 10 ** np.floor(np.log10(max(nn, min_m)))
    res = max(round(nn / mag) * mag, min_m)

    # scale up until patch count within limit
    pad = res * 2
    while True:
        nx = int(np.ceil((xm.max() - xm.min() + 2 * pad) / res / 16)) * 16
        ny = int(np.ceil((ym.max() - ym.min() + 2 * pad) / res / 16)) * 16
        n_patches = (nx // 16) * (ny // 16)
        if n_patches <= max_patches:
            break
        res *= 2

    print(f"  Auto resolution: {res:.0f} m  (median NN = {nn:.0f} m  →  "
          f"grid {nx}×{ny}  {n_patches} patches)")
    return res


def generate_area(area_cfg: dict, out_root: str,
                  resolution_m: float = None,
                  method: str = "idw",
                  resume: bool = False,
                  fs: str = "none",
                  pca_top_k: int = 15,
                  transform: str = "clr"):

    area_id = area_cfg["area_id"]
    tsuf = "" if transform == "clr" else f"_{transform}"
    vae_dir = os.path.join(out_root, area_id, f"vae_patches{FS_SUFFIX[fs]}{tsuf}")
    split_path = os.path.join(vae_dir, "dataset_split.pkl")
    meta_path  = os.path.join(vae_dir, "metadata.json")

    if resume and os.path.exists(split_path) and os.path.exists(meta_path):
        print(f"  [skip] {area_id} already done")
        return

    print(f"\n{'='*60}\n  {area_id}\n{'='*60}")
    os.makedirs(vae_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(area_cfg["csv_path"], low_memory=False)
    df = df.replace(-9999, np.nan)

    gdf = pd.read_csv(area_cfg["deposits"])

    # ── Detect element columns (feature_selection-aware) ──────────────────────
    def _auto_detect():
        available = get_df_element(df, elements=[], include_coords=True)
        coord_kw = {"x", "y", "longitude", "latitude", "easting", "northing"}
        return [c for c in available
                if c.lower() not in coord_kw and df[c].dtype != object]

    raw_cols = select_raw_cols(area_cfg, df.columns, fs=fs,
                               pca_top_k=pca_top_k, auto_detect=_auto_detect)

    if not raw_cols:
        print(f"  [warn] No element columns found, skipping.")
        return

    # ── Compositional transform across ALL elements ──────────────────────────
    X = df[raw_cols].copy().astype(float)
    for col in raw_cols:
        s = X[col]
        min_pos = s[s > 0].min()
        half_dl = min_pos / 2 if pd.notna(min_pos) and min_pos > 0 else 1e-6
        X[col] = s.apply(lambda v: v if (pd.notna(v) and v > 0) else half_dl)

    Xnp = X.to_numpy(dtype=float)
    if transform == "clr":
        Xt = clr(Xnp)
        out_cols = [f"{c}_clr" for c in raw_cols]
    elif transform == "ilr":
        # ilr drops one dim → D-1 ilr coords; keep generic names ilr_0..ilr_{D-2}
        Xt = ilr(Xnp)
        out_cols = [f"ilr_{i}" for i in range(Xt.shape[1])]
    elif transform == "none":
        # Just log1p of raw values for VAE input range stability
        Xt = np.log1p(Xnp).astype(float)
        out_cols = [f"{c}_raw" for c in raw_cols]
    else:
        raise ValueError(f"unknown transform={transform!r}")

    clr_cols = out_cols   # keep name 'clr_cols' to preserve downstream code
    df_clr = pd.concat([
        df[["X", "Y"]].reset_index(drop=True),
        pd.DataFrame(Xt, columns=clr_cols),
    ], axis=1)

    # ── Auto resolution ────────────────────────────────────────────────────────
    res = resolution_m if resolution_m else _auto_resolution(df)
    print(f"  FS: {fs}  Elements: {len(raw_cols)}  "
          f"Resolution: {res:.0f} m  Method: {method}")

    # ── Compute UTM grid origin (mirrors interpolate_and_annotate_with_patches) ─
    lons_all = df["X"].values.astype(float)
    lats_all = df["Y"].values.astype(float)
    xm_all, ym_all = _to_utm(lons_all, lats_all)
    lon_mean = float(lons_all.mean()); lat_mean = float(lats_all.mean())
    zone = int((lon_mean + 180) / 6) + 1
    utm_epsg = 32600 + zone if lat_mean >= 0 else 32700 + zone
    x_min_raw, x_max_raw = xm_all.min(), xm_all.max()
    y_min_raw, y_max_raw = ym_all.min(), ym_all.max()
    nc = int(np.ceil((x_max_raw - x_min_raw) / res))
    nr = int(np.ceil((y_max_raw - y_min_raw) / res))
    if nc % PATCH_SIZE != 0: nc = ((nc // PATCH_SIZE) + 1) * PATCH_SIZE
    if nr % PATCH_SIZE != 0: nr = ((nr // PATCH_SIZE) + 1) * PATCH_SIZE
    cx = (x_min_raw + x_max_raw) / 2; cy = (y_min_raw + y_max_raw) / 2
    x_min_m = cx - nc * res / 2
    y_min_m = cy - nr * res / 2

    # ── Interpolate each element ───────────────────────────────────────────────
    tmp_dir = os.path.join(vae_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    shared_split = None  # reuse first element's split structure

    # For ilr, raw_cols (D) and clr_cols (D-1) differ — use clr_cols as the
    # iteration anchor so we cover every transformed channel exactly once.
    iter_pairs = (
        list(zip(raw_cols, clr_cols))
        if len(raw_cols) == len(clr_cols)
        else [(c, c) for c in clr_cols]
    )

    for i, (raw_col, clr_col) in enumerate(iter_pairs):
        valid = df_clr[clr_col].notna().sum()
        if valid < 6:
            print(f"  [{i+1}/{len(iter_pairs)}] skip {raw_col}: only {valid} valid values")
            continue

        elem_name = _sanitize(raw_col.replace("_ppm", "").replace("_pct", "").replace("_ppb", ""))
        patch_subdir = os.path.join(vae_dir, f"patches_{elem_name}")

        if resume and os.path.isdir(patch_subdir):
            if shared_split is None:
                sp = os.path.join(patch_subdir, "dataset_split.pkl")
                if os.path.exists(sp):
                    with open(sp, "rb") as f:
                        shared_split = pickle.load(f)
            print(f"  [{i+1}/{len(raw_cols)}] skip {raw_col}: patches exist")
            continue

        img_path  = os.path.join(tmp_dir, f"{elem_name}_interp.png")
        ann_path  = os.path.join(tmp_dir, f"{elem_name}_annotation.pkl")

        try:
            interpolate_and_annotate_with_patches(
                df=df_clr,
                gdf=gdf,
                column=clr_col,
                patch_size=PATCH_SIZE,
                resolution_m=res,
                method=method,
                output_image=img_path,
                output_json=ann_path,
            )

            # Pad image to nearest multiple of PATCH_SIZE if needed
            _img = Image.open(img_path)
            _w, _h = _img.size
            _new_w = ((_w + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
            _new_h = ((_h + PATCH_SIZE - 1) // PATCH_SIZE) * PATCH_SIZE
            if (_new_w, _new_h) != (_w, _h):
                _padded = Image.new(_img.mode, (_new_w, _new_h), 0)
                _padded.paste(_img, (0, 0))
                _padded.save(img_path)
            _img.close()

            os.makedirs(patch_subdir, exist_ok=True)
            split = split_image_into_patches_and_prepare_dataset(
                image_path=img_path,
                patch_label_json=ann_path,
                output_dir=patch_subdir,
                patch_size=PATCH_SIZE,
                save_images=True,
            )

            if shared_split is None:
                shared_split = split

            # clean up per-element tmp files
            for f in [img_path, ann_path]:
                if os.path.exists(f):
                    os.remove(f)

            print(f"  [{i+1}/{len(raw_cols)}] {raw_col} → {patch_subdir.split('/')[-1]}"
                  f"  ({split['total_patches']} patches, "
                  f"pos={sum(1 for p in split['train'] if p['label']==1)})")

        except Exception as e:
            print(f"  [{i+1}/{len(raw_cols)}] ERROR {raw_col}: {type(e).__name__}: {e}")
            continue

    # ── Save shared dataset_split.pkl + metadata.json ─────────────────────────
    if shared_split is None:
        print(f"  [warn] No elements completed, skipping metadata.")
        return

    with open(split_path, "wb") as f:
        pickle.dump(shared_split, f)

    meta = {
        "area_id":           area_id,
        "feature_selection": fs,
        "transform":         transform,
        "resolution_m":      res,
        "patch_size":        PATCH_SIZE,
        "image_size":        shared_split["image_size"],
        "num_patches":       shared_split["num_patches"],
        "method":            method,
        "n_channels":        len(iter_pairs),
        "raw_cols":          raw_cols,
        "channel_cols":      clr_cols,
        "utm_epsg":          utm_epsg,
        "x_min_m":           float(x_min_m),
        "y_min_m":           float(y_min_m),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # clean up tmp dir
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    print(f"  Done → {vae_dir}")
    print(f"         {shared_split['total_patches']} patches  ×  {len(raw_cols)} elements")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--areas",        default="all")
    p.add_argument("--out_root",     default="outputs")
    p.add_argument("--resolution_m", type=float, default=None,
                   help="Grid resolution metres (default: auto from data)")
    p.add_argument("--method",       default="idw",
                   choices=["idw", "kriging"],
                   help="Interpolation method (default: idw)")
    p.add_argument("--feature_selection", default="none", choices=FS_CHOICES,
                   help="none = all element columns (baseline), "
                        "llm = shared/llm_feature_selection.json subset, "
                        "pca = target + top-k correlated columns")
    p.add_argument("--pca_top_k", type=int, default=15,
                   help="Top-k for --feature_selection pca (target added on top)")
    p.add_argument("--transform",    default="clr",
                   choices=["clr", "ilr", "none"],
                   help="Compositional transform applied to element values "
                        "before interpolation. 'none' = log1p of raw values.")
    p.add_argument("--resume",       action="store_true",
                   help="Skip areas / elements that already have output")
    return p.parse_args()


def main():
    args  = get_args()
    areas = resolve_areas(args.areas)
    print(f"Generating VAE patches for {len(areas)} area(s)  "
          f"[fs={args.feature_selection}  method={args.method}  "
          f"transform={args.transform}  resume={args.resume}]")
    for area_cfg in areas:
        generate_area(
            area_cfg,
            out_root=args.out_root,
            resolution_m=args.resolution_m,
            method=args.method,
            resume=args.resume,
            fs=args.feature_selection,
            pca_top_k=args.pca_top_k,
            transform=args.transform,
        )
    print("\nAll done.")


if __name__ == "__main__":
    main()
