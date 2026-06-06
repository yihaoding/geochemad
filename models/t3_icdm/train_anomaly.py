"""
Stage 3: train AnomalyT1 on the area, using the frozen backbone features as
extra per-point input.

The transformer architecture matches paper T1 (encoder + decoder, d_model=128,
nhead=2, 4+4 layers). The ONLY change vs paper T1 is the input feature dim:

    per-point token feature = [concentrations(C) ; BDL_flags(C) ; h_backbone(d_b)]
                              ↑↑↑ standard T1 hands this to in_proj

Reconstruction MSE on the query's element vector is the training loss (same as
paper T1).  Best ckpt by validation MSE on a 90/10 split.

Per-area sed1 defaults:
  • K_neighbors  = 64  (T1 in paper used 16; we boost to 64 since K=512 in the
                       backbone is already captured in h_i)
  • epochs       = 100  (paper T1 standard)
  • batch_size   = 32   (memory OK with K=64)
  • lr=1e-4, weight_decay=0.01, dropout=0.1   (paper T1 defaults)
"""

import argparse, os, json, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data  import load_area, build_neighbour_index, collate, iter_batches
from model import AnomalyT1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path",       required=True)
    p.add_argument("--target_element", required=True)
    p.add_argument("--features_npy",   required=True,
                   help="Stage-2 backbone hidden state per sample [N, d_b]")
    p.add_argument("--meta_json",      required=True,
                   help="Stage-2 meta.json — must match preprocessing of this run")
    p.add_argument("--transform",      default=None,
                   help="override transform (default: use meta.json)")
    p.add_argument("--input_elements", default=None,
                   help="override element subset (default: use meta.json)")
    p.add_argument("--bdl_strategy",   default="flag")
    p.add_argument("--bdl_threshold",  type=float, default=0.95)
    p.add_argument("--k_neighbors",    type=int,   default=64)
    p.add_argument("--d_model",        type=int,   default=128)
    p.add_argument("--n_heads",        type=int,   default=2)
    p.add_argument("--n_enc_layers",   type=int,   default=4)
    p.add_argument("--n_dec_layers",   type=int,   default=4)
    p.add_argument("--dim_feedforward",type=int,   default=128)
    p.add_argument("--dropout",        type=float, default=0.1)
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--warmup",         type=int,   default=5)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--val_frac",       type=float, default=0.1)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--out_dir",        required=True)
    return p.parse_args()


def build_point_features(ds, h):
    """
    Concat per-point feature  [elem ; bdl ; backbone_h]  → [N, 2C + d_b].
    h is the frozen backbone hidden state (already standardised via LayerNorm
    inside the backbone), no further scaling needed.
    """
    return np.concatenate([ds["elem"], ds["bdl"], h], axis=1).astype(np.float32)


def collate_aug(ds, h_feat, nbr_idx, query_ids, device):
    coords = ds["coords"]
    nbr = nbr_idx[query_ids]

    q_xy   = torch.tensor(coords[query_ids],          device=device)
    q_feat = torch.tensor(h_feat[query_ids],          device=device)
    n_xy   = torch.tensor(coords[nbr],                device=device)
    n_feat = torch.tensor(h_feat[nbr],                device=device)
    rel_xy = n_xy - q_xy.unsqueeze(1)
    return q_xy, q_feat, rel_xy, n_feat


def run_epoch(model, ds, h_feat, nbr_idx, id_list, args, device, train, optimizer=None):
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    sum_recon = 0.0; n_seen = 0
    target_dim = ds["elem"].shape[1]

    with ctx:
        for q_ids in iter_batches(id_list, args.batch_size, shuffle=train, seed=args.seed):
            q_xy, q_feat, rel_xy, n_feat = collate_aug(ds, h_feat, nbr_idx, q_ids, device)
            recon = model(q_xy, q_feat, rel_xy, n_feat)  # [B, C]
            target = torch.tensor(ds["elem"][q_ids], device=device)
            loss   = F.mse_loss(recon, target)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            B = len(q_ids)
            sum_recon += loss.item() * B; n_seen += B
    return sum_recon / max(n_seen, 1)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(args.meta_json))
    transform      = args.transform      if args.transform      else meta["transform"]
    input_elements = args.input_elements if args.input_elements else None
    # If meta lists element_cols and user gave nothing, use meta cols (drop_col
    # might have removed some during pretrain).
    if input_elements is None:
        input_elements = ",".join(meta["element_cols"])

    ds = load_area(
        args.csv_path, args.target_element,
        transform      = transform,
        input_elements = input_elements,
        bdl_strategy   = args.bdl_strategy,
        bdl_threshold  = args.bdl_threshold,
    )
    N = len(ds["coords"]); C = ds["elem"].shape[1]
    h_raw = np.load(args.features_npy)
    if h_raw.shape != (N, meta["d_model"]):
        raise ValueError(f"feature shape {h_raw.shape} != ({N}, {meta['d_model']})")
    d_backbone = meta["d_model"]
    h_feat = build_point_features(ds, h_raw)              # [N, 2C + d_b]
    print(f"[train_anomaly] N={N}  C={C}  d_backbone={d_backbone}  "
          f"d_in_per_token={2*C + d_backbone}")

    rng = np.random.default_rng(args.seed)
    n_val   = max(1, int(N * args.val_frac))
    val_idx = rng.choice(N, size=n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(N), val_idx)

    nbr_idx = build_neighbour_index(ds["coords"], args.k_neighbors)

    model = AnomalyT1(
        n_elem=C, d_backbone=d_backbone,
        d_model=args.d_model, nhead=args.n_heads,
        num_enc_layers=args.n_enc_layers, num_dec_layers=args.n_dec_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
    ).to(device)
    print(f"[train_anomaly] params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched     = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"val_mse": float("inf"), "epoch": -1}
    history, t0 = [], time.time()
    for ep in range(1, args.epochs + 1):
        if ep <= args.warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * ep / args.warmup
        tr = run_epoch(model, ds, h_feat, nbr_idx, tr_idx,  args, device, True,  optimizer)
        va = run_epoch(model, ds, h_feat, nbr_idx, val_idx, args, device, False)
        if ep > args.warmup: sched.step()
        print(f"  Ep {ep:3d}/{args.epochs}  tr_mse={tr:.5f}  va_mse={va:.5f}", end="")
        if va < best["val_mse"]:
            best = {"val_mse": va, "epoch": ep}
            torch.save({
                "model_state_dict": model.state_dict(),
                "args":             vars(args),
                "epoch":            ep,
                "n_elem":           C,
                "d_backbone":       d_backbone,
                "element_cols":     ds["element_cols"],
                "transform":        ds["transform"],
                "target_idx":       ds["target_idx"],
                "scaler_mean":      ds["scaler"].mean_,
                "scaler_scale":     ds["scaler"].scale_,
            }, os.path.join(args.out_dir, "anomaly_model.pt"))
            print("  ← best")
        else:
            print()
        history.append({"epoch": ep, "tr_mse": tr, "va_mse": va})

    json.dump(history, open(os.path.join(args.out_dir, "anomaly_history.json"), "w"), indent=2)
    print(f"[train_anomaly] done.  best val_mse={best['val_mse']:.5f} @ ep {best['epoch']}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
