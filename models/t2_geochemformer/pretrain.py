#!/usr/bin/env python3
"""
pretrain.py  –  Stage 1: Pretrain GeoTransformer with masked-element prediction.
=================================================================================
Task:
  For each training sample, randomly select ONE element to predict.
  Context = K nearest neighbours' FULL element measurements + coordinates.
  Neighbour tokens are NOT masked for the predicted element.

  Optional neighbour augmentation (training only):
    --neighbor_mask_prob : randomly zero each element in each neighbour token
    --neighbor_noise_std : add Gaussian noise to neighbour element features

Val set:
  Spatially-stratified held-out points (never used during training).
  Fast val each epoch: one fixed random element assignment per val point.
  Full per-element breakdown every --eval_every epochs + final epoch.

Saves
-----
  <out_dir>/geo_transformer.pt
      model_state_dict, feature_cols, feature_cols_raw, predict_cols,
      elem_to_id, raw_cols, scaler, in_dim, args
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from geo_transformer import (
    GeoTransformer, half_detection_limit, CompositionalTransformer,
    EXCLUDE_COLS,
)

SHARED_ELEMENTS = [
    "As", "Sb", "Ag", "W",  "Mo", "Bi",
    "Sn", "Nb", "Ta",
    "Au", "Pb", "Zn",
    "Co", "Cu", "Fe", "Mg", "Ca", "Ni",
]


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(description="GeoTransformer masked-element pretraining")
    p.add_argument("--csv_path",            required=True)
    p.add_argument("--target_element",      default="Au_ppm",
                   help="Single col, comma-separated list, or 'all'")
    p.add_argument("--shared_elements",     default=None,
                   help="Comma-separated base element names. Default: built-in 18.")
    p.add_argument("--k_neighbors",         type=int,   default=64)
    p.add_argument("--hidden_dim",          type=int,   default=256)
    p.add_argument("--n_heads",             type=int,   default=8)
    p.add_argument("--n_layers",            type=int,   default=4)
    p.add_argument("--epochs",              type=int,   default=50)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--batch_size",          type=int,   default=64)
    p.add_argument("--weight_decay",        type=float, default=1e-4)
    p.add_argument("--dropout",             type=float, default=0.0)
    p.add_argument("--warmup_epochs",       type=int,   default=2)
    p.add_argument("--val_size",            type=int,   default=2000)
    p.add_argument("--val_grid_n",          type=int,   default=20)
    p.add_argument("--eval_every",          type=int,   default=10)
    p.add_argument("--transform",           default="clr",
                   choices=["none", "clr"])
    p.add_argument("--neighbor_mask_prob",  type=float, default=0.0)
    p.add_argument("--neighbor_noise_std",  type=float, default=0.0)
    p.add_argument("--out_dir",             default="outputs/pretrain")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--num_workers",         type=int,   default=4)
    return p.parse_args()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class GeoDataset(Dataset):
    """
    Stores precomputed neighbour indices and builds tokens on-the-fly.

    coords_query : [N_query, 2]  – query point coordinates
    coords_ref   : [N_ref,   2]  – reference (training) coordinates
    X_ref        : [N_ref,   C]  – scaled features for reference points
    neighbor_idx : [N_query, K]  – indices into coords_ref / X_ref
    y_all        : [N_query, P]  – ground truth for each predict element
    """
    def __init__(self, coords_query: np.ndarray,
                 coords_ref: np.ndarray, X_ref: np.ndarray,
                 neighbor_idx: np.ndarray, y_all: np.ndarray,
                 mask_prob: float = 0.0, noise_std: float = 0.0):
        self.coords_query = torch.tensor(coords_query, dtype=torch.float32)  # [N_q, 2]
        self.coords_ref   = torch.tensor(coords_ref,   dtype=torch.float32)  # [N_r, 2]
        self.X_ref        = torch.tensor(X_ref,        dtype=torch.float32)  # [N_r, C]
        self.neighbor_idx = torch.tensor(neighbor_idx, dtype=torch.long)     # [N_q, K]
        self.y_all        = torch.tensor(y_all,        dtype=torch.float32)  # [N_q, P]
        self.mask_prob    = mask_prob
        self.noise_std    = noise_std

    def __len__(self):
        return len(self.coords_query)

    def __getitem__(self, idx):
        coord    = self.coords_query[idx]                  # [2]
        nbr_idx  = self.neighbor_idx[idx]                  # [K]
        nbr_xy   = self.coords_ref[nbr_idx]                # [K, 2]  ← ref coords
        nbr_feat = self.X_ref[nbr_idx].clone()             # [K, C]  ← ref features

        if self.mask_prob > 0:
            mask = torch.rand_like(nbr_feat) < self.mask_prob
            nbr_feat[mask] = 0.0
        if self.noise_std > 0:
            nbr_feat = nbr_feat + torch.randn_like(nbr_feat) * self.noise_std

        dxy    = nbr_xy - coord.unsqueeze(0)               # [K, 2]
        tokens = torch.cat([dxy, nbr_feat], dim=-1)        # [K, 2+C]
        return tokens, coord, self.y_all[idx]


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


def spatial_stratified_split(df: pd.DataFrame, val_size: int,
                              grid_n: int = 20, seed: int = 42):
    """Spatially stratified val split using a grid_n × grid_n spatial grid."""
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


def precompute_neighbors(coords_query: np.ndarray, coords_ref: np.ndarray,
                         k: int, exclude_self: bool) -> np.ndarray:
    """Batch KDTree query. Returns [N_query, K] int32 neighbor indices into coords_ref."""
    tree    = KDTree(coords_ref.astype(np.float32))
    k_fetch = k + 1 if exclude_self else k
    k_fetch = min(k_fetch, len(coords_ref))
    _, ind  = tree.query(coords_query.astype(np.float32), k=k_fetch)
    if exclude_self:
        ind = ind[:, 1:]        # drop self (first column)
    ind = ind[:, :k]            # ensure exactly K neighbours
    return ind.astype(np.int32)


def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _interp_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rmse   = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae    = float(np.mean(np.abs(yt - yp)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2     = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    r      = float(np.corrcoef(yt, yp)[0, 1])
    return rmse, mae, float(r2), r


def _fmt(v):
    return f"{v:8.4f}" if np.isfinite(v) else "     nan"


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"[pretrain] Loading {args.csv_path}")
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
    print(f"[pretrain] {len(shared_cols)} shared cols: {shared_cols}")

    # ── Resolve predict columns ───────────────────────────────────────────────
    target_str = args.target_element.strip()
    if target_str.lower() == "all":
        predict_cols = list(shared_cols)
    elif "," in target_str:
        predict_cols = [c.strip() for c in target_str.split(",")]
    else:
        predict_cols = [target_str]

    missing = [c for c in predict_cols if c not in numeric_cols]
    if missing:
        raise ValueError(f"Target columns not found in CSV: {missing}")

    seen, feature_cols_raw = set(), []
    for c in shared_cols + predict_cols:
        if c in numeric_cols and c not in seen:
            feature_cols_raw.append(c); seen.add(c)

    df = half_detection_limit(df, feature_cols_raw)
    print(f"[pretrain] {len(df)} samples | {len(feature_cols_raw)} feature cols | "
          f"{len(predict_cols)} predict cols: {predict_cols}")

    # ── Spatial stratified val split ──────────────────────────────────────────
    train_idx, val_idx = spatial_stratified_split(
        df, val_size=args.val_size, grid_n=args.val_grid_n, seed=args.seed)
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    print(f"[pretrain] split → train={len(train_df)}  val={len(val_df)}")

    # ── Fit transformer on training data only ─────────────────────────────────
    transformer = CompositionalTransformer(method=args.transform)
    X_train = transformer.fit_transform(
        train_df[feature_cols_raw].to_numpy(dtype=np.float32), feature_cols_raw)
    X_val   = transformer.transform(
        val_df[feature_cols_raw].to_numpy(dtype=np.float32))
    feature_cols = transformer.feature_cols_

    elem_to_id       = {col: i for i, col in enumerate(predict_cols)}
    id_to_elem       = {i: col for col, i in elem_to_id.items()}
    n_predict        = len(predict_cols)
    elem_to_feat_idx = {col: feature_cols.index(col) for col in predict_cols}

    # [N, n_predict] ground truth in scaled space
    y_train_all = X_train[:, [elem_to_feat_idx[c] for c in predict_cols]]
    y_val_all   = X_val[:,   [elem_to_feat_idx[c] for c in predict_cols]]

    coords_train = train_df[["X", "Y"]].to_numpy(dtype=np.float32)
    coords_val   = val_df[["X", "Y"]].to_numpy(dtype=np.float32)

    # ── Precompute neighbour indices ──────────────────────────────────────────
    print(f"[pretrain] Precomputing neighbour indices (K={args.k_neighbors}) ...")
    train_nbr_idx = precompute_neighbors(
        coords_train, coords_train, k=args.k_neighbors, exclude_self=True)
    val_nbr_idx   = precompute_neighbors(
        coords_val, coords_train,   k=args.k_neighbors, exclude_self=False)
    print("[pretrain] Done precomputing.")

    in_dim = X_train.shape[1] + 2   # (dx, dy) + C features
    print(f"[pretrain] device={DEVICE}  in_dim={in_dim}  num_elements={n_predict}  "
          f"batch_size={args.batch_size}")
    print(f"[pretrain] neighbor_mask_prob={args.neighbor_mask_prob}  "
          f"neighbor_noise_std={args.neighbor_noise_std}  dropout={args.dropout}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_dataset = GeoDataset(
        coords_train, coords_train, X_train, train_nbr_idx, y_train_all,
        mask_prob=args.neighbor_mask_prob, noise_std=args.neighbor_noise_std)
    val_dataset   = GeoDataset(
        coords_val, coords_train, X_train, val_nbr_idx, y_val_all)  # no augmentation

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GeoTransformer(
        in_dim       = in_dim,
        hidden_dim   = args.hidden_dim,
        n_heads      = args.n_heads,
        n_layers     = args.n_layers,
        num_elements = n_predict,
        dropout      = args.dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)
    criterion = nn.MSELoss()

    # Fixed val element assignments for reproducible fast-val each epoch
    rng_val     = np.random.default_rng(args.seed + 99)
    val_eis_fix = rng_val.integers(0, n_predict, size=len(val_dataset)).tolist()

    # Checkpoint name encodes key hyperparameters to avoid confusion across runs
    target_tag = (
        "all" if args.target_element.strip().lower() == "all"
        else args.target_element.replace(",", "+").replace(" ", "")
    )
    ckpt_name = (
        f"geo_transformer"
        f"_K{args.k_neighbors}"
        f"_H{args.hidden_dim}"
        f"_L{args.n_layers}"
        f"_h{args.n_heads}"
        f"_B{args.batch_size}"
        f"_{args.transform}"
        f"_{target_tag}"
        f".pt"
    )
    ckpt_path = os.path.join(args.out_dir, ckpt_name)
    print(f"[pretrain] Checkpoint name: {ckpt_name}")

    best_val_loss = float("inf")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_steps = 0.0, 0

        for tokens_b, coords_b, y_all_b in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            B = tokens_b.size(0)
            # One random element per sample in this batch
            eis      = torch.randint(n_predict, (B,), device=DEVICE)
            y_target = y_all_b[torch.arange(B), eis.cpu()].to(DEVICE)  # [B]

            valid = torch.isfinite(y_target)
            if not valid.any():
                continue
            if not valid.all():
                tokens_b  = tokens_b[valid]
                coords_b  = coords_b[valid]
                eis       = eis[valid]
                y_target  = y_target[valid]

            tok_t = tokens_b.to(DEVICE)
            crd_t = coords_b.to(DEVICE)

            pred, _ = model(eis, crd_t, tok_t)
            loss = criterion(pred, y_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * valid.sum().item()
            n_steps    += valid.sum().item()

        scheduler.step()
        train_loss = total_loss / max(n_steps, 1)
        current_lr = scheduler.get_last_lr()[0]

        # ── Fast val (fixed element assignments, no augmentation) ─────────────
        model.eval()
        val_losses = []
        sample_offset = 0
        with torch.no_grad():
            for tokens_b, coords_b, y_all_b in val_loader:
                B   = tokens_b.size(0)
                eis = torch.tensor(
                    val_eis_fix[sample_offset:sample_offset + B],
                    dtype=torch.long, device=DEVICE)
                y_true = y_all_b[torch.arange(B), eis.cpu()].to(DEVICE)
                valid  = torch.isfinite(y_true)

                pred, _ = model(eis, coords_b.to(DEVICE), tokens_b.to(DEVICE))
                mse = ((pred[valid] - y_true[valid]) ** 2).cpu().numpy()
                val_losses.extend(mse.tolist())
                sample_offset += B

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        print(f"Epoch {epoch+1:3d}/{args.epochs} | lr={current_lr:.2e} | "
              f"train_MSE={train_loss:.5f} | val_MSE={avg_val_loss:.5f}")

        # ── Full per-element val breakdown ────────────────────────────────────
        is_last      = (epoch + 1 == args.epochs)
        do_full_eval = ((epoch + 1) % args.eval_every == 0) or is_last
        if do_full_eval:
            elem_metrics = {}
            with torch.no_grad():
                for ei, elem_col in enumerate(predict_cols):
                    preds = []
                    for tokens_b, coords_b, y_all_b in val_loader:
                        B     = tokens_b.size(0)
                        eis_b = torch.full((B,), ei, dtype=torch.long, device=DEVICE)
                        pred, _ = model(eis_b, coords_b.to(DEVICE), tokens_b.to(DEVICE))
                        preds.extend(pred.cpu().numpy().tolist())
                    preds  = np.array(preds, dtype=np.float64)
                    y_true = y_val_all[:, ei].astype(np.float64)
                    elem_metrics[elem_col] = _interp_metrics(y_true, preds)

            print(f"  ── Per-element val (epoch {epoch+1}) {'─'*38}")
            print(f"  {'Element':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'r':>8}")
            rmses, maes, r2s, rs = [], [], [], []
            for col, (rmse, mae, r2, r) in elem_metrics.items():
                print(f"  {col:<22}{_fmt(rmse)}{_fmt(mae)}{_fmt(r2)}{_fmt(r)}")
                for lst, v in [(rmses, rmse), (maes, mae), (r2s, r2), (rs, r)]:
                    if np.isfinite(v): lst.append(v)
            avg = (np.mean(rmses) if rmses else float("nan"),
                   np.mean(maes)  if maes  else float("nan"),
                   np.mean(r2s)   if r2s   else float("nan"),
                   np.mean(rs)    if rs    else float("nan"))
            print(f"  {'[average]':<22}{_fmt(avg[0])}{_fmt(avg[1])}{_fmt(avg[2])}{_fmt(avg[3])}")
            print(f"  {'─'*54}")
            # Use avg RMSE as checkpoint criterion on full-eval epochs
            if np.isfinite(avg[0]):
                avg_val_loss = float(avg[0])

        # ── Checkpoint ───────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict":  model.state_dict(),
                "feature_cols":      feature_cols,
                "feature_cols_raw":  feature_cols_raw,
                "predict_cols":      predict_cols,
                "elem_to_id":        elem_to_id,
                "raw_cols":          feature_cols_raw,
                "scaler":            transformer,
                "in_dim":            in_dim,
                "args":              vars(args),
            }, ckpt_path)
            print(f"  [ckpt] saved  → {ckpt_path}")

    print(f"\n[pretrain] Done.  Best val metric = {best_val_loss:.6f}")
    print(f"[pretrain] Checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
