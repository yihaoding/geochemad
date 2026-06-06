#!/usr/bin/env python3
"""
run_t1.py  –  T1 baseline: AnomalyTransformer without SCL pretraining.

Trains on all samples. Evaluates every epoch on the full dataset.
Saves two checkpoints per area:
  best  : highest test AUC across all epochs
  last  : final epoch weights

Outputs per area (in <out_root>/<area_id>/t1/):
  ckpt_best.pt / ckpt_last.pt
  anomaly_scores_best.csv / anomaly_scores_last.csv
  metrics_best.json / metrics_last.json
"""

import argparse
import datetime
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _SCRIPTS)
from shared.area_config   import resolve_areas
from shared.preprocess    import load_area, save_scores, run_evaluate
from shared.anomaly_model import AnomalyTransformer


def get_args():
    p = argparse.ArgumentParser(description="T1 Vanilla Transformer anomaly baseline")
    p.add_argument("--areas",      default="all")
    p.add_argument("--out_root",   default="outputs")
    p.add_argument("--transform",  default="clr", choices=["clr", "none"])
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--n_heads",    type=int,   default=4)
    p.add_argument("--n_layers",   type=int,   default=2)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--radius_km",  type=float, default=2.0)
    p.add_argument("--min_dist",   type=float, default=0.5)
    p.add_argument("--neg_runs",   type=int,   default=20)
    return p.parse_args()


def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _log(msg):
    print(f"[{_now()}] {msg}", flush=True)


def _make_model(C, args, device):
    return AnomalyTransformer(
        n_elem=C, latent_dim=args.hidden,
        hidden=args.hidden, n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device)


def _save_ckpt(model, C, args, feature_cols, area_id, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_elem":           C,
        "latent_dim":       args.hidden,
        "feature_cols":     feature_cols,
        "area_id":          area_id,
    }, path)


def _score_all(model, X_al, Z_al, args, device):
    loader = DataLoader(
        TensorDataset(torch.tensor(X_al), torch.tensor(Z_al)),
        batch_size=args.batch_size, shuffle=False)
    scores = []
    model.eval()
    with torch.no_grad():
        for x, z in loader:
            scores.append(model.score(x.to(device), z.to(device)).cpu().numpy())
    return np.concatenate(scores)


def _evaluate(model, X_al, Z_al, df_all, area_cfg, args, out_dir, device, fname_suffix):
    scores     = _score_all(model, X_al, Z_al, args, device)
    scores_csv = save_scores(df_all, scores, out_dir,
                             fname=f"anomaly_scores_{fname_suffix}.csv")
    metrics    = run_evaluate(
        scores_csv    = scores_csv,
        deposits_csv  = area_cfg["deposits"],
        out_dir       = out_dir,
        radius_km     = args.radius_km,
        min_dist      = args.min_dist,
        neg_runs      = args.neg_runs,
        label         = f"{area_cfg['area_id']}/t1/{fname_suffix}",
        metrics_fname = f"metrics_{fname_suffix}.json",
    )
    return metrics


def run_area(area_cfg: dict, args, device: str):
    area_id = area_cfg["area_id"]
    out_dir = os.path.join(args.out_root, area_id, "t1")
    os.makedirs(out_dir, exist_ok=True)

    print(flush=True)
    print(f"{'='*60}", flush=True)
    _log(f"T1  {area_id}  START")
    print(f"{'='*60}", flush=True)

    df_train, _, df_all, feature_cols, _, _ = load_area(
        area_cfg, transform=args.transform, val_frac=0.0, seed=args.seed)

    C    = len(feature_cols)
    N    = len(df_all)
    X_al = df_all[feature_cols].to_numpy(dtype=np.float32)
    Z_al = np.zeros((N, args.hidden), dtype=np.float32)

    loader = DataLoader(TensorDataset(torch.tensor(X_al), torch.tensor(Z_al)),
                        batch_size=args.batch_size, shuffle=True)

    _log(f"  samples={N}  features={C}  epochs={args.epochs}  device={device}")

    model     = _make_model(C, args, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    ckpt_best = os.path.join(out_dir, "ckpt_best.pt")
    ckpt_last = os.path.join(out_dir, "ckpt_last.pt")

    best_auc   = -1.0
    best_epoch = -1
    best_metrics = None
    last_metrics = None

    print(f"\n  {'Ep':>4}  {'train_loss':>12}  {'AUC':>7}  {'AP':>7}  {'SR@5%':>7}  {'DTD':>8}  note",
          flush=True)
    print(f"  {'-'*65}", flush=True)

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0
        for x, z in loader:
            x, z = x.to(device), z.to(device)
            x_hat, _ = model(x, z)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(x)
        tr_loss /= N

        metrics = _evaluate(model, X_al, Z_al, df_all, area_cfg, args, out_dir, device,
                             fname_suffix="current")

        auc = metrics["AUC_mean"]
        note = ""
        if auc > best_auc:
            best_auc     = auc
            best_epoch   = epoch + 1
            best_metrics = metrics
            _save_ckpt(model, C, args, feature_cols, area_id, ckpt_best)
            note = " ← best"

        print(f"  [{_now()}] {epoch+1:4d}  {tr_loss:12.6f}"
              f"  {auc:7.4f}  {metrics['AP_mean']:7.4f}"
              f"  {metrics['SR_5pct']:7.4f}  {metrics['DTD_km']:7.2f}km{note}",
              flush=True)

    _save_ckpt(model, C, args, feature_cols, area_id, ckpt_last)

    last_metrics = _evaluate(model, X_al, Z_al, df_all, area_cfg, args, out_dir, device,
                              fname_suffix="last")

    # copy best scores/metrics to their final filenames
    import shutil
    src = os.path.join(out_dir, "anomaly_scores_current.csv")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(out_dir, "anomaly_scores_best.csv"))

    print(f"\n  {'='*50}", flush=True)
    _log(f"  [best]  epoch={best_epoch}  AUC={best_metrics['AUC_mean']:.4f}"
         f"  AP={best_metrics['AP_mean']:.4f}"
         f"  SR@5%={best_metrics['SR_5pct']:.4f}"
         f"  DTD={best_metrics['DTD_km']:.2f}km")
    _log(f"  [last]  AUC={last_metrics['AUC_mean']:.4f}"
         f"  AP={last_metrics['AP_mean']:.4f}"
         f"  SR@5%={last_metrics['SR_5pct']:.4f}"
         f"  DTD={last_metrics['DTD_km']:.2f}km")
    _log(f"T1  {area_id}  DONE")


def main():
    args  = get_args()
    areas = resolve_areas(args.areas)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available, running on CPU — will be slow!", flush=True)

    _log(f"T1 start | device={device} | areas={len(areas)} | epochs={args.epochs}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for area_cfg in areas:
        run_area(area_cfg, args, device)

    _log("T1 all areas complete.")


if __name__ == "__main__":
    main()
