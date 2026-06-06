#!/usr/bin/env python3
"""
extract_latent_multi.py  –  Stage 2: extract GeoTransformerMulti latents.
==========================================================================
Loads a pretrain_multi.py checkpoint and runs a forward pass over every
point in the CSV to extract backbone latents.

Outputs (saved to --out_dir)
-----------------------------
  latents.npy          [N, H]     backbone latent per point
  neighbor_latents.npy [N, K, H]  backbone latents of K neighbours
  neighbor_indices.npy [N, K]     KDTree neighbour row indices (into N)
  coords.npy           [N, 2]     (X, Y) coordinates
  elem_features.npy    [N, C]     CLR-transformed feature matrix
  target_values.npy    [N, P]     raw (pre-CLR) values of predict_cols
  meta.pkl             dict with predict_cols, predict_feat_idx, feature_cols

Optional (when --site_csv is provided)
  site_latents.npy     [S, H]
  site_clr_vals.npy    [S, P]     CLR values at each site for predict_cols
  site_coords.npy      [S, 2]

Optional (when --n_random > 0)
  random_latents.npy   [R, H]     latents at random geographic locations
  random_clr_vals.npy  [R, P]
  random_coords.npy    [R, 2]
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KDTree
from tqdm import tqdm

from geo_transformer import half_detection_limit, EXCLUDE_COLS
from geo_transformer_multi import GeoTransformerMulti


def get_args():
    p = argparse.ArgumentParser(
        description="Extract GeoTransformerMulti latents for anomaly training")
    p.add_argument("--ckpt_path",   required=True,
                   help="Path to checkpoint saved by pretrain_multi.py")
    csv_group = p.add_mutually_exclusive_group(required=True)
    csv_group.add_argument("--csv_path",  default=None,
                           help="Single geochemical CSV (original behaviour).")
    csv_group.add_argument("--csv_paths", default=None,
                           help="Comma-separated list of geochemical CSVs. "
                                "Latents are extracted for each area into a "
                                "separate sub-directory under --out_dir.")
    p.add_argument("--out_dir",     default="outputs/latents_multi")
    # ── KNN ───────────────────────────────────────────────────────────────────
    p.add_argument("--k_neighbors", type=int, default=None,
                   help="Override K (default: read from checkpoint)")
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    # ── Site evaluation ───────────────────────────────────────────────────────
    p.add_argument("--site_csv",    default=None,
                   help="CSV of known mineral sites (requires X,Y columns). "
                        "Extracts exact backbone latents at site locations "
                        "via KNN interpolation from the area data.")
    p.add_argument("--site_radius_km", type=float, default=5.0,
                   help="Spatial buffer (km) used to warn if sites are far "
                        "from any area data point (informational only).")
    # ── Random negative pool ──────────────────────────────────────────────────
    p.add_argument("--n_random",    type=int, default=0,
                   help="Number of random geographic locations to extract "
                        "as a pre-built negative pool. 0 = disabled.")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def precompute_neighbors(coords_query: np.ndarray,
                         coords_ref: np.ndarray,
                         k: int,
                         exclude_self: bool) -> np.ndarray:
    tree    = KDTree(coords_ref.astype(np.float32))
    k_fetch = min(k + (1 if exclude_self else 0), len(coords_ref))
    _, ind  = tree.query(coords_query.astype(np.float32), k=k_fetch)
    if exclude_self:
        ind = ind[:, 1:]
    return ind[:, :k].astype(np.int32)


def build_neighbor_tokens_batch(coords_q: np.ndarray,
                                coords_ref: np.ndarray,
                                X_ref: np.ndarray,
                                nbr_idx: np.ndarray) -> tuple:
    """
    Returns
    -------
    tokens : [N, K, 2+C]  (dx, dy, features) for each query's K neighbours
    """
    K = nbr_idx.shape[1]
    N = len(coords_q)
    C = X_ref.shape[1]
    tokens = np.empty((N, K, 2 + C), dtype=np.float32)
    for i in range(N):
        nbr_xy   = coords_ref[nbr_idx[i]]      # [K, 2]
        dxy      = nbr_xy - coords_q[i]         # [K, 2]
        nbr_feat = X_ref[nbr_idx[i]]            # [K, C]
        tokens[i] = np.concatenate([dxy, nbr_feat], axis=1)
    return tokens


# ─── Per-area extraction helper ───────────────────────────────────────────────

def extract_one_area(csv_path: str, args,
                     model, scaler, feature_cols, feature_cols_raw,
                     predict_cols, predict_feat_idx, k, H, DEVICE,
                     out_dir: str, rng):
    """
    Extract backbone latents for a single area CSV and save to out_dir.
    Mirrors the original single-CSV extraction logic exactly.
    """
    os.makedirs(out_dir, exist_ok=True)
    bs = args.batch_size

    print(f"\n[extract] ══ Area: {os.path.basename(csv_path)} ══")
    df = pd.read_csv(csv_path).dropna(subset=["X", "Y"])

    missing_raw = [c for c in feature_cols_raw if c not in df.columns]
    if missing_raw:
        print(f"[extract] [warn] columns absent, will be NaN-imputed: {missing_raw}")
        for c in missing_raw:
            df[c] = np.nan

    df     = half_detection_limit(df, feature_cols_raw)
    coords = df[["X", "Y"]].to_numpy(dtype=np.float32)
    N      = len(df)

    X_clr = scaler.transform(df[feature_cols_raw].to_numpy(dtype=np.float32))
    C     = X_clr.shape[1]

    target_raw = df[
        [c for c in predict_cols if c in df.columns]
    ].to_numpy(dtype=np.float32)

    print(f"[extract] N={N}  C={C}  device={DEVICE}")

    print(f"[extract] Precomputing KNN (K={k}) …")
    nbr_idx = precompute_neighbors(coords, coords, k=k, exclude_self=True)

    print("[extract] Extracting backbone latents …")
    latents     = np.empty((N, H),    dtype=np.float32)
    nbr_latents = np.empty((N, k, H), dtype=np.float32)
    crd_t = torch.tensor(coords, dtype=torch.float32)

    with torch.no_grad():
        for start in tqdm(range(0, N, bs), desc="Latents"):
            end    = min(start + bs, N)
            tok_np = build_neighbor_tokens_batch(
                coords[start:end], coords, X_clr, nbr_idx[start:end])
            tok = torch.tensor(tok_np, dtype=torch.float32).to(DEVICE)
            crd = crd_t[start:end].to(DEVICE)
            _, h = model(crd, tok)
            latents[start:end] = h.cpu().numpy()

    print("[extract] Building neighbour latent matrix …")
    for i in range(N):
        nbr_latents[i] = latents[nbr_idx[i]]

    print(f"[extract] latents={latents.shape}  nbr_latents={nbr_latents.shape}")

    np.save(os.path.join(out_dir, "latents.npy"),          latents)
    np.save(os.path.join(out_dir, "neighbor_latents.npy"), nbr_latents)
    np.save(os.path.join(out_dir, "neighbor_indices.npy"), nbr_idx)
    np.save(os.path.join(out_dir, "coords.npy"),           coords)
    np.save(os.path.join(out_dir, "elem_features.npy"),    X_clr)
    np.save(os.path.join(out_dir, "target_values.npy"),    target_raw)

    meta = {
        "predict_cols":     predict_cols,
        "predict_feat_idx": predict_feat_idx,
        "feature_cols":     feature_cols,
        "feature_cols_raw": feature_cols_raw,
        "k_neighbors":      k,
        "hidden_dim":       H,
        "n_predict":        len(predict_cols),
        "ckpt_path":        args.ckpt_path,
        "csv_path":         csv_path,
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"[extract] Area latents saved → {out_dir}")

    # ── Optional: site latents ────────────────────────────────────────────────
    if args.site_csv is not None and os.path.isfile(args.site_csv):
        print(f"[extract] Extracting site latents from: {args.site_csv}")
        site_df     = pd.read_csv(args.site_csv).dropna(subset=["X", "Y"])
        site_coords = site_df[["X", "Y"]].to_numpy(dtype=np.float32)
        S           = len(site_coords)
        print(f"[extract] S={S} mineral sites")

        site_nbr_idx  = precompute_neighbors(site_coords, coords, k=k,
                                             exclude_self=False)
        site_latents  = np.empty((S, H),                  dtype=np.float32)
        site_clr_vals = np.empty((S, len(predict_cols)),  dtype=np.float32)

        with torch.no_grad():
            for s_start in tqdm(range(0, S, bs), desc="Site latents"):
                s_end  = min(s_start + bs, S)
                tok_np = build_neighbor_tokens_batch(
                    site_coords[s_start:s_end], coords, X_clr,
                    site_nbr_idx[s_start:s_end])
                tok = torch.tensor(tok_np, dtype=torch.float32).to(DEVICE)
                crd = torch.tensor(
                    site_coords[s_start:s_end], dtype=torch.float32).to(DEVICE)
                _, h = model(crd, tok)
                site_latents[s_start:s_end]  = h.cpu().numpy()
                nn_idx = site_nbr_idx[s_start:s_end, 0]
                site_clr_vals[s_start:s_end] = X_clr[nn_idx][:, predict_feat_idx]

        np.save(os.path.join(out_dir, "site_latents.npy"),  site_latents)
        np.save(os.path.join(out_dir, "site_clr_vals.npy"), site_clr_vals)
        np.save(os.path.join(out_dir, "site_coords.npy"),   site_coords)
        print(f"[extract] Site latents saved  S={S}")
    elif args.site_csv is not None:
        print(f"[extract] [warn] site_csv not found: {args.site_csv}")

    # ── Optional: random negative pool ───────────────────────────────────────
    if args.n_random > 0:
        R = args.n_random
        print(f"[extract] Generating R={R} random geographic locations …")
        xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
        ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
        rand_x      = rng.uniform(xmin, xmax, size=R).astype(np.float32)
        rand_y      = rng.uniform(ymin, ymax, size=R).astype(np.float32)
        rand_coords = np.stack([rand_x, rand_y], axis=1)

        rand_nbr_idx  = precompute_neighbors(rand_coords, coords, k=k,
                                             exclude_self=False)
        rand_latents  = np.empty((R, H),                 dtype=np.float32)
        rand_clr_vals = np.empty((R, len(predict_cols)), dtype=np.float32)

        with torch.no_grad():
            for r_start in tqdm(range(0, R, bs), desc="Random latents"):
                r_end  = min(r_start + bs, R)
                tok_np = build_neighbor_tokens_batch(
                    rand_coords[r_start:r_end], coords, X_clr,
                    rand_nbr_idx[r_start:r_end])
                tok = torch.tensor(tok_np, dtype=torch.float32).to(DEVICE)
                crd = torch.tensor(
                    rand_coords[r_start:r_end], dtype=torch.float32).to(DEVICE)
                _, h = model(crd, tok)
                rand_latents[r_start:r_end]  = h.cpu().numpy()
                nn_idx = rand_nbr_idx[r_start:r_end, 0]
                rand_clr_vals[r_start:r_end] = X_clr[nn_idx][:, predict_feat_idx]

        np.save(os.path.join(out_dir, "random_latents.npy"),  rand_latents)
        np.save(os.path.join(out_dir, "random_clr_vals.npy"), rand_clr_vals)
        np.save(os.path.join(out_dir, "random_coords.npy"),   rand_coords)
        print(f"[extract] Random latents saved  R={R}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    # ── Resolve CSV list ──────────────────────────────────────────────────────
    csv_list = (
        [p.strip() for p in args.csv_paths.split(",")]
        if args.csv_paths else [args.csv_path]
    )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"[extract] Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=DEVICE, weights_only=False)

    ckpt_args        = ckpt["args"]
    feature_cols     = ckpt["feature_cols"]
    feature_cols_raw = ckpt["feature_cols_raw"]
    predict_cols     = ckpt["predict_cols"]
    scaler           = ckpt["scaler"]
    in_dim           = ckpt["in_dim"]
    n_predict        = ckpt["n_predict"]
    k = args.k_neighbors or ckpt_args.get("k_neighbors", 64)

    predict_feat_idx = [feature_cols.index(c) for c in predict_cols
                        if c in feature_cols]

    model = GeoTransformerMulti(
        in_dim         = in_dim,
        n_predict      = n_predict,
        hidden_dim     = ckpt_args["hidden_dim"],
        n_heads        = ckpt_args["n_heads"],
        n_layers       = ckpt_args["n_layers"],
        dropout        = 0.0,
        use_cross_attn = bool(ckpt_args.get("use_cross_attn") or False),
        use_mlp_pe     = bool(ckpt_args.get("use_mlp_pe") or False),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    H = ckpt_args["hidden_dim"]
    print(f"[extract] Model loaded  H={H}  K={k}  n_predict={n_predict}")
    print(f"[extract] predict_cols: {predict_cols}")

    # ── Extract latents for each area ─────────────────────────────────────────
    if len(csv_list) == 1:
        # Single-area: save directly to out_dir (original behaviour)
        extract_one_area(
            csv_list[0], args, model, scaler,
            feature_cols, feature_cols_raw, predict_cols, predict_feat_idx,
            k, H, DEVICE, args.out_dir, rng)
    else:
        # Multi-area: each area gets its own sub-directory
        print(f"[extract] Multi-area mode: {len(csv_list)} area(s)")
        for csv_path in csv_list:
            area_tag = os.path.splitext(os.path.basename(csv_path))[0]
            area_dir = os.path.join(args.out_dir, area_tag)
            extract_one_area(
                csv_path, args, model, scaler,
                feature_cols, feature_cols_raw, predict_cols, predict_feat_idx,
                k, H, DEVICE, area_dir, rng)

    print(f"\n[extract] Done → {args.out_dir}")


if __name__ == "__main__":
    main()
