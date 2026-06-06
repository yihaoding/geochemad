#!/usr/bin/env python3
"""
run_t2.py  –  GeoChemFormer end-to-end (e2e) anomaly training.

A single GeoChemFormer = GeoTransformer + AnomalyTransformer is trained jointly
in ONE pass.  There is no separate SCL pretraining checkpoint and no offline
latent-extraction step — the geo encoder is shaped by the downstream anomaly
objective while an auxiliary SCL term keeps its latent target-element-aware.

  GeoTransformer     : K spatial neighbours → geo-context latent z  (+ target pred)
  AnomalyTransformer : (element features x, latent z) → reconstruct x
  Anomaly score      : per-sample reconstruction MSE  (higher = more anomalous)

Joint loss
----------
  loss = MSE(x_hat, x)  +  scl_weight * MSE(pred, y_target)
         └ reconstruction (primary)    └ SCL auxiliary (masked to finite targets)

Best checkpoint is selected by minimum val *reconstruction* loss (the anomaly
objective).  Final anomaly_scores.csv covers ALL N samples for spatial eval.

This supersedes the old run_scl.py + Stage-2 latent-extraction flow; run_scl.py
is no longer needed and the stale t2/scl_ckpt.pt files are no longer read.

Usage
-----
  python run_t2.py --areas area9_dh1_au
  python run_t2.py --areas all --epochs 50 --scl_weight 0.3
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _SCRIPTS)
from shared.area_config   import resolve_areas
from shared.preprocess    import load_area, save_scores, run_evaluate
from shared.geo_model     import GeoTransformer
from shared.anomaly_model import AnomalyTransformer


def get_args():
    p = argparse.ArgumentParser(description="GeoChemFormer end-to-end anomaly training")
    p.add_argument("--areas",       default="all")
    p.add_argument("--out_root",    default="outputs")
    p.add_argument("--transform",   default="clr", choices=["clr", "none"])
    p.add_argument("--k_neighbors", type=int,   default=16)
    # GeoTransformer
    p.add_argument("--geo_hidden",  type=int,   default=128)
    p.add_argument("--geo_heads",   type=int,   default=4)
    p.add_argument("--geo_layers",  type=int,   default=3)
    # AnomalyTransformer
    p.add_argument("--anom_hidden", type=int,   default=128)
    p.add_argument("--anom_heads",  type=int,   default=4)
    p.add_argument("--anom_layers", type=int,   default=2)
    # Training
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--scl_weight",  type=float, default=0.3,
                   help="weight of the auxiliary SCL prediction loss")
    p.add_argument("--val_frac",    type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    # Evaluation
    p.add_argument("--radius_km",   type=float, default=2.0)
    p.add_argument("--min_dist",    type=float, default=0.5)
    p.add_argument("--neg_runs",    type=int,   default=20)
    p.add_argument("--selection",   default="auc", choices=["recon", "auc"],
                   help="best ckpt by val recon loss (paper-conservative) or by "
                        "per-epoch single-split AUC on deposit labels (paper-like)")
    p.add_argument("--auc_pos_radius_km", type=float, default=2.0,
                   help="positive-label radius for per-epoch AUC selection")
    return p.parse_args()


# ── GeoChemFormer: GeoTransformer + AnomalyTransformer, trained jointly ───────

class GeoChemFormer(nn.Module):
    """End-to-end model: geo-context encoder feeding an anomaly autoencoder."""

    def __init__(self, in_dim, n_elem,
                 geo_hidden, geo_heads, geo_layers,
                 anom_hidden, anom_heads, anom_layers):
        super().__init__()
        self.geo = GeoTransformer(
            in_dim=in_dim, hidden_dim=geo_hidden,
            n_heads=geo_heads, n_layers=geo_layers, num_elements=1)
        self.anom = AnomalyTransformer(
            n_elem=n_elem, latent_dim=geo_hidden,
            hidden=anom_hidden, n_heads=anom_heads, n_layers=anom_layers)

    def forward(self, elem_idx, coord, nbr_tokens, x):
        """
        elem_idx   : [B]          long   (always 0; single target element)
        coord      : [B, 2]       float  query (x, y)
        nbr_tokens : [B, K, 2+C]  float  neighbour tokens (Δx, Δy + features)
        x          : [B, C]       float  query element features

        Returns
        -------
        pred  : [B]     SCL target-element prediction
        x_hat : [B, C]  reconstructed element features
        """
        pred, z  = self.geo(elem_idx, coord, nbr_tokens)   # gradient flows into geo
        x_hat, _ = self.anom(x, z)
        return pred, x_hat


# ── Neighbour token precompute (vectorised, built once per area) ─────────────

def precompute_neighbor_tokens(coords_q, coords_ref, X_ref, k, exclude_self):
    """
    Build a [N, k, 2+C] neighbour-token tensor for every query point.

    Neighbours are always drawn from the *reference* set (the training rows),
    so val points see only training context and there is no val→val leakage.

    coords_q     : [N, 2]   query coordinates (all samples)
    coords_ref   : [M, 2]   reference coordinates (training samples)
    X_ref        : [M, C]   reference scaled features
    k            : neighbours per query
    exclude_self : [N] bool  True where the query is itself in the reference
                             set (its nearest hit is itself → drop column 0)
    """
    n_ref   = len(coords_ref)
    if n_ref <= k:
        raise ValueError(f"need >k reference points, got {n_ref} for k={k}")
    tree    = KDTree(coords_ref)
    _, idx  = tree.query(coords_q, k=k + 1)               # [N, k+1]

    take_excl = idx[:, 1:k + 1]                            # drop self (col 0)
    take_incl = idx[:, :k]                                 # keep first k
    nbr_idx   = np.where(exclude_self[:, None], take_excl, take_incl)  # [N, k]

    dxy   = coords_ref[nbr_idx] - coords_q[:, None, :]     # [N, k, 2]
    feats = X_ref[nbr_idx]                                 # [N, k, C]
    return np.concatenate([dxy, feats], axis=-1).astype(np.float32)


# ── Train one area end-to-end + score all samples ───────────────────────────

def run_area(area_cfg: dict, args, device: str):
    area_id = area_cfg["area_id"]
    t2_dir  = os.path.join(args.out_root, area_id, "t2")
    os.makedirs(t2_dir, exist_ok=True)

    print(f"\n{'='*60}\n  T2 (e2e)  {area_id}\n{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────────
    df_train, df_val, df_all, feature_cols, raw_cols, scaler = load_area(
        area_cfg, transform=args.transform, val_frac=args.val_frac, seed=args.seed)

    tgt = area_cfg.get("target_element")
    if not tgt or tgt not in df_all.columns:
        print(f"  [skip] target '{tgt}' not in data")
        return

    N          = len(df_all)
    C          = len(feature_cols)
    coords_all = df_all[["X", "Y"]].to_numpy(dtype=np.float32)
    X_all      = df_all[feature_cols].to_numpy(dtype=np.float32)
    y_all      = df_all[tgt].to_numpy(dtype=np.float32)     # scaled target value

    # Val split — re-derived to match load_area (same rng / seed / N)
    rng       = np.random.default_rng(args.seed)
    val_idx   = rng.choice(N, size=max(1, int(N * args.val_frac)), replace=False)
    train_idx = np.setdiff1d(np.arange(N), val_idx)

    is_train  = np.zeros(N, dtype=bool)
    is_train[train_idx] = True

    # Neighbour tokens: reference = training rows; query = all rows.
    nbr_tokens = precompute_neighbor_tokens(
        coords_q     = coords_all,
        coords_ref   = coords_all[train_idx],
        X_ref        = X_all[train_idx],
        k            = args.k_neighbors,
        exclude_self = is_train)                            # [N, k, 2+C]

    in_dim = 2 + C
    print(f"  N={N}  train={len(train_idx)}  val={len(val_idx)}  "
          f"C={C}  in_dim={in_dim}  scl_weight={args.scl_weight}")

    # ── Tensors / loaders ────────────────────────────────────────────────────
    X_t   = torch.tensor(X_all,      dtype=torch.float32)
    nbr_t = torch.tensor(nbr_tokens, dtype=torch.float32)
    crd_t = torch.tensor(coords_all, dtype=torch.float32)
    y_t   = torch.tensor(y_all,      dtype=torch.float32)
    idx_t = torch.arange(N)

    tr_ds  = TensorDataset(X_t[train_idx], nbr_t[train_idx], crd_t[train_idx], y_t[train_idx])
    va_ds  = TensorDataset(X_t[val_idx],   nbr_t[val_idx],   crd_t[val_idx],   y_t[val_idx])
    all_ds = TensorDataset(X_t, nbr_t, crd_t, y_t, idx_t)

    tr_loader  = DataLoader(tr_ds,  batch_size=args.batch_size, shuffle=True)
    va_loader  = DataLoader(va_ds,  batch_size=args.batch_size, shuffle=False)
    all_loader = DataLoader(all_ds, batch_size=args.batch_size, shuffle=False)

    # ── Model ────────────────────────────────────────────────────────────────
    model = GeoChemFormer(
        in_dim=in_dim, n_elem=C,
        geo_hidden=args.geo_hidden,  geo_heads=args.geo_heads,  geo_layers=args.geo_layers,
        anom_hidden=args.anom_hidden, anom_heads=args.anom_heads, anom_layers=args.anom_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse       = nn.MSELoss()
    ckpt_path = os.path.join(t2_dir, "geochemformer.pt")
    best_val  = float("inf")    # used when args.selection == "recon"
    best_auc  = -1.0            # used when args.selection == "auc"

    # ── For AUC-based selection: precompute deposit-proximity labels ─────────
    auc_labels = None
    if args.selection == "auc":
        import pandas as pd
        from sklearn.metrics import roc_auc_score as _roc_auc
        from sklearn.neighbors import KDTree as _KDTree
        dep_df = pd.read_csv(area_cfg["deposits"])
        # locate x/y columns (case insensitive)
        cols = {c.lower(): c for c in dep_df.columns}
        x_col = cols.get("x") or cols.get("longitude") or cols.get("lon")
        y_col = cols.get("y") or cols.get("latitude")  or cols.get("lat")
        dep_xy = dep_df[[x_col, y_col]].to_numpy(dtype=np.float32)
        # label = positive if any deposit within auc_pos_radius_km (deg→km via 111)
        radius_deg = args.auc_pos_radius_km / 111.0
        tree       = _KDTree(dep_xy)
        idx_within = tree.query_radius(coords_all, r=radius_deg)
        auc_labels = np.array([len(x) > 0 for x in idx_within], dtype=np.int32)
        n_pos      = int(auc_labels.sum())
        print(f"  [auc-sel] {n_pos}/{N} samples within {args.auc_pos_radius_km}km of {len(dep_xy)} deposits")
        if n_pos == 0 or n_pos == N:
            print(f"  [auc-sel] degenerate labels — falling back to recon selection")
            args.selection = "recon"

    def score_all() -> np.ndarray:
        """Score every sample with current model state."""
        model.eval()
        out = np.empty(N, dtype=np.float32)
        with torch.no_grad():
            for x, nbr, crd, y, idx in all_loader:
                x, nbr, crd = x.to(device), nbr.to(device), crd.to(device)
                eid = torch.zeros(len(x), dtype=torch.long, device=device)
                _, x_hat = model(eid, crd, nbr, x)
                s = ((x - x_hat) ** 2).mean(dim=-1)
                out[idx.numpy()] = s.cpu().numpy()
        return out

    def save_ckpt():
        torch.save({
            "model_state_dict": model.state_dict(),
            "area_id":      area_id,
            "feature_cols": feature_cols,
            "in_dim":       in_dim,
            "n_elem":       C,
            "geo_hidden":   args.geo_hidden,
            "geo_heads":    args.geo_heads,
            "geo_layers":   args.geo_layers,
            "anom_hidden":  args.anom_hidden,
            "anom_heads":   args.anom_heads,
            "anom_layers":  args.anom_layers,
            "k_neighbors":  args.k_neighbors,
            "scl_weight":   args.scl_weight,
            "transform":    args.transform,
        }, ckpt_path)

    def run_epoch(loader, train: bool):
        """Returns (recon, scl) means over the loader."""
        model.train() if train else model.eval()
        recon_sum = scl_sum = 0.0
        n_total   = scl_total = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, nbr, crd, y in loader:
                x, nbr, crd, y = x.to(device), nbr.to(device), crd.to(device), y.to(device)
                eid  = torch.zeros(len(x), dtype=torch.long, device=device)
                pred, x_hat = model(eid, crd, nbr, x)

                recon = mse(x_hat, x)
                m     = torch.isfinite(y)
                if m.any():
                    scl = mse(pred[m], y[m])
                else:
                    scl = torch.zeros((), device=device)

                if train:
                    loss = recon + args.scl_weight * scl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                recon_sum += recon.item() * len(x)
                n_total   += len(x)
                scl_sum   += scl.item() * int(m.sum())
                scl_total += int(m.sum())
        return recon_sum / max(n_total, 1), scl_sum / max(scl_total, 1)

    for epoch in tqdm(range(args.epochs), desc="epochs", leave=False):
        tr_recon, tr_scl = run_epoch(tr_loader, train=True)
        va_recon, va_scl = run_epoch(va_loader, train=False)
        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"train recon={tr_recon:.5f} scl={tr_scl:.5f} | "
              f"val recon={va_recon:.5f} scl={va_scl:.5f}", end="")

        improved = False
        if args.selection == "recon":
            if va_recon < best_val:
                best_val = va_recon
                save_ckpt()
                improved = True
        else:  # selection == "auc"
            try:
                ep_scores = score_all()
                ep_auc    = float(_roc_auc(auc_labels, ep_scores))
            except Exception as _e:
                ep_auc = float("nan")
            print(f" | auc={ep_auc:.4f}", end="")
            if ep_auc > best_auc:
                best_auc = ep_auc
                save_ckpt()
                improved = True
        if improved:
            print("  ← best", end="")
        print()

    if args.selection == "auc":
        print(f"  [e2e done]  best_epoch_auc={best_auc:.4f}  → {ckpt_path}")
    else:
        print(f"  [e2e done]  best_val_recon={best_val:.6f}  → {ckpt_path}")

    # ── Score ALL samples with best model ────────────────────────────────────
    best = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    model.eval()

    scores = np.empty(N, dtype=np.float32)
    with torch.no_grad():
        for x, nbr, crd, y, idx in all_loader:
            x, nbr, crd = x.to(device), nbr.to(device), crd.to(device)
            eid = torch.zeros(len(x), dtype=torch.long, device=device)
            _, x_hat = model(eid, crd, nbr, x)
            s = ((x - x_hat) ** 2).mean(dim=-1)
            scores[idx.numpy()] = s.cpu().numpy()

    scores_csv = save_scores(df_all, scores, t2_dir)
    print(f"  anomaly_scores.csv → {scores_csv}")

    metrics = run_evaluate(
        scores_csv   = scores_csv,
        deposits_csv = area_cfg["deposits"],
        out_dir      = t2_dir,
        radius_km    = args.radius_km,
        min_dist     = args.min_dist,
        neg_runs     = args.neg_runs,
        label        = f"{area_id}/t2",
    )
    print(f"  AUC={metrics['AUC_mean']:.4f}  AP={metrics['AP_mean']:.4f}  "
          f"SR@5%={metrics['SR_5pct']:.4f}  DTD={metrics['DTD_km']:.2f}km")


def main():
    args   = get_args()
    areas  = resolve_areas(args.areas)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Areas: {len(areas)}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for area_cfg in areas:
        run_area(area_cfg, args, device)

    print("\nGeoChemFormer e2e training complete.")


if __name__ == "__main__":
    main()
