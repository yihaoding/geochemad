#!/usr/bin/env python3
"""
run_scl.py  –  Stage 1: Spatial Context Learning (SCL) pretraining.

DEPRECATED — superseded by run_t2.py, which now trains the GeoTransformer and
AnomalyTransformer jointly end-to-end (SCL kept as an auxiliary loss).  This
script is retained only for reference; it is no longer part of the pipeline.

Trains a GeoTransformer to predict the target element concentration at each
location from the element measurements of its K nearest spatial neighbours.
The hidden state of the target-location token encodes spatial geo-context.

Saves
-----
  <out_root>/<area_id>/t2/scl_ckpt.pt
      model_state_dict, feature_cols, raw_cols, scaler, in_dim, args

Usage
-----
  python run_scl.py --areas area9_dh1_au
  python run_scl.py --areas all --epochs 50
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KDTree
from tqdm import tqdm

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _SCRIPTS)
from shared.area_config import resolve_areas
from shared.preprocess  import load_area
from shared.geo_model   import GeoTransformer, build_neighbor_tokens


def get_args():
    p = argparse.ArgumentParser(description="GeoChemFormer Stage-1 SCL pretraining")
    p.add_argument("--areas",       default="all")
    p.add_argument("--out_root",    default="outputs")
    p.add_argument("--transform",   default="clr", choices=["clr", "none"])
    p.add_argument("--k_neighbors", type=int,   default=16)
    p.add_argument("--hidden_dim",  type=int,   default=128)
    p.add_argument("--n_heads",     type=int,   default=4)
    p.add_argument("--n_layers",    type=int,   default=3)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--val_frac",    type=float, default=0.05)
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


def run_area(area_cfg: dict, args, device: str):
    area_id = area_cfg["area_id"]
    out_dir = os.path.join(args.out_root, area_id, "t2")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "scl_ckpt.pt")

    print(f"\n{'='*60}\n  SCL  {area_id}\n{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────────
    df_train, df_val, df_all, feature_cols, raw_cols, scaler = load_area(
        area_cfg, transform=args.transform, val_frac=args.val_frac, seed=args.seed)

    tgt = area_cfg["target_element"]
    if tgt not in df_train.columns:
        print(f"  [skip] target '{tgt}' not in data")
        return

    # Filter rows with valid target
    tr_mask  = df_train[tgt].notna() & np.isfinite(df_train[tgt].to_numpy(float))
    df_train = df_train[tr_mask].reset_index(drop=True)
    va_mask  = df_val[tgt].notna() & np.isfinite(df_val[tgt].to_numpy(float))
    df_val   = df_val[va_mask].reset_index(drop=True)

    coords_tr = df_train[["X", "Y"]].to_numpy(dtype=np.float32)
    X_tr      = df_train[feature_cols].to_numpy(dtype=np.float32)
    y_tr      = df_train[tgt].to_numpy(dtype=np.float32)

    coords_va = df_val[["X", "Y"]].to_numpy(dtype=np.float32)
    y_va      = df_val[tgt].to_numpy(dtype=np.float32)

    tree_tr = KDTree(coords_tr)

    in_dim = len(feature_cols) + 2   # (Δx, Δy) + C features
    print(f"  train={len(df_train)}  val={len(df_val)}  in_dim={in_dim}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GeoTransformer(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_elements=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion  = nn.MSELoss()

    best_val_mse = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = np.random.permutation(len(df_train))

        for idx in tqdm(order, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):
            tokens = build_neighbor_tokens(
                coords_tr[idx], tree_tr, coords_tr, X_tr,
                k=args.k_neighbors, exclude_self=True)

            tok_t = torch.tensor(tokens,       dtype=torch.float32).unsqueeze(0).to(device)
            crd_t = torch.tensor(coords_tr[idx], dtype=torch.float32).unsqueeze(0).to(device)
            eid_t = torch.zeros(1, dtype=torch.long, device=device)
            tgt_t = torch.tensor([y_tr[idx]], dtype=torch.float32).to(device)

            pred, _ = model(eid_t, crd_t, tok_t)
            loss = criterion(pred, tgt_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_mse = total_loss / len(df_train)

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        val_preds = []
        with torch.no_grad():
            for i in range(len(df_val)):
                tokens = build_neighbor_tokens(
                    coords_va[i], tree_tr, coords_tr, X_tr,
                    k=args.k_neighbors, exclude_self=False)
                tok_t  = torch.tensor(tokens,        dtype=torch.float32).unsqueeze(0).to(device)
                crd_t  = torch.tensor(coords_va[i],  dtype=torch.float32).unsqueeze(0).to(device)
                eid_t  = torch.zeros(1, dtype=torch.long, device=device)
                pred, _ = model(eid_t, crd_t, tok_t)
                val_preds.append(pred.item())

        val_mse = float(np.mean((np.array(val_preds) - y_va) ** 2))

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_MSE={train_mse:.5f}  val_MSE={val_mse:.5f}", end="")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_cols":     feature_cols,
                "raw_cols":         raw_cols,
                "scaler":           scaler,
                "in_dim":           in_dim,
                "hidden_dim":       args.hidden_dim,
                "n_heads":          args.n_heads,
                "n_layers":         args.n_layers,
                "k_neighbors":      args.k_neighbors,
                "area_id":          area_id,
                "target_element":   tgt,
                "transform":        args.transform,
            }, ckpt_path)
            print("  ← best", end="")
        print()

    print(f"  [SCL done]  best_val_MSE={best_val_mse:.6f}  → {ckpt_path}")


def main():
    args   = get_args()
    areas  = resolve_areas(args.areas)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Areas: {len(areas)}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for area_cfg in areas:
        run_area(area_cfg, args, device)

    print("\nSCL pretraining complete.")


if __name__ == "__main__":
    main()
