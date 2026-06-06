"""
Stage 1 pretrain for SC-AnomalyFormer-v3.

Three SSL tasks (different weights than v2 to reflect new priorities):

  PRIMARY  L_target  — spatial-context-only target prediction
                         target Au is ALWAYS masked from query input
                         model has to learn  E[Au | neighbours]
  SECOND   L_mcm     — masked-element regression on NON-target elements
                         random 20% non-target elements masked
  AUX      L_con     — hard contrastive between two views of the SAME query
                         negatives = other queries in batch (true contrastive)

Total = 1.0*L_target + 0.5*L_mcm + 0.3*L_con

Stopping: 80 epochs OR early-stop on val_target with patience=15.

Per-area defaults (sed1 N=1392):
  K=512, hierarchical (k_local=16, k_mid=128, k_global=368)
  d_model=128, encoders 4+3+3 layers, heads=4, dropout=0.2
  lr=1e-4, wd=0.05, batch=32, mask 20%, warmup 5
"""

import argparse, os, json, time
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data     import load_area, build_neighbour_index, collate, iter_batches
from model_v3 import HierarchicalBackbone, augment_query


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path",       required=True)
    p.add_argument("--target_element", required=True)
    p.add_argument("--transform",      default="none",
                   choices=["none", "log", "clr", "ilr"])
    p.add_argument("--input_elements", default=None)
    p.add_argument("--bdl_strategy",   default="flag",
                   choices=["flag", "mask", "drop_col", "none"])
    p.add_argument("--bdl_threshold",  type=float, default=0.95)
    p.add_argument("--paper_compatible", action="store_true", default=True,
                   help="use paper element regex + median half-DL (default: on)")
    p.add_argument("--no_paper_compat",  dest="paper_compatible",
                   action="store_false")
    p.add_argument("--k_neighbors",    type=int,   default=512)
    p.add_argument("--k_local",        type=int,   default=16)
    p.add_argument("--k_mid",          type=int,   default=128)
    p.add_argument("--d_model",        type=int,   default=128)
    p.add_argument("--n_heads",        type=int,   default=4)
    p.add_argument("--n_layers_local", type=int,   default=4)
    p.add_argument("--n_layers_mid",   type=int,   default=3)
    p.add_argument("--n_layers_glob",  type=int,   default=3)
    p.add_argument("--dropout",        type=float, default=0.20)
    p.add_argument("--epochs",         type=int,   default=80)
    p.add_argument("--early_patience", type=int,   default=15)
    p.add_argument("--warmup",         type=int,   default=5)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=0.05)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--mcm_mask_ratio", type=float, default=0.20)
    p.add_argument("--elem_drop",      type=float, default=0.15)
    p.add_argument("--bdl_flip",       type=float, default=0.05)
    p.add_argument("--info_nce_T",     type=float, default=0.10)
    p.add_argument("--w_target",       type=float, default=1.0)
    p.add_argument("--w_mcm",          type=float, default=0.5)
    p.add_argument("--w_con",          type=float, default=0.3)
    # VICReg anti-collapse regulariser (per-batch on h_q)
    p.add_argument("--w_var",          type=float, default=1.0,
                   help="variance hinge: each dim std must be ≥ 1")
    p.add_argument("--w_cov",          type=float, default=0.04,
                   help="off-diagonal covariance penalty (decorrelation)")
    p.add_argument("--val_frac",       type=float, default=0.10)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--out_dir",        required=True)
    # ── Path B architecture ablation toggles ────────────────────────────
    p.add_argument("--use_scales",     default="local,mid,glob",
                   help="Comma list selected from {local,mid,glob}; "
                        "controls which encoder scales are included. "
                        "Single value (e.g. 'local') => single-scale ablation.")
    p.add_argument("--no_scale_tag",   action="store_true",
                   help="Disable learnable scale_tag (zero-frozen) for "
                        "scale_tag ablation.")
    return p.parse_args()


def split_neighbours(nbr_idx, k_local, k_mid, K_total):
    """Indices already sorted by distance (nearest first).  Return slice ids."""
    sl_local = slice(0, k_local)
    sl_mid   = slice(k_local, k_local + k_mid)
    sl_glob  = slice(k_local + k_mid, K_total)
    return sl_local, sl_mid, sl_glob


def info_nce_batch(z1, z2, T):
    """In-batch negatives.  z1, z2 are L2-normalised [B, d]."""
    B = z1.size(0)
    if B <= 1:
        return torch.zeros((), device=z1.device)
    logits = z1 @ z2.t() / T                              # [B, B]
    labels = torch.arange(B, device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                  F.cross_entropy(logits.t(), labels))


def make_view(batch, target_idx, sl_l, sl_m, sl_g,
              elem_drop, bdl_flip,
              mcm_mask_ratio: float, with_mcm_mask: bool, device):
    """
    Produce one augmented view.  Returns the 13 tensors needed for
    backbone forward + the mcm mask (if requested).
    """
    q_elem = batch["q_elem"].clone()
    q_bdl  = batch["q_bdl" ].clone()

    # always mask target element
    q_elem[:, target_idx] = 0.0
    q_bdl [:, target_idx] = 0.0

    # augmentation (random element dropout + BDL flip)
    q_elem, q_bdl = augment_query(q_elem, q_bdl,
                                  elem_drop=elem_drop, bdl_flip=bdl_flip)

    # optional MCM mask on NON-target elements
    if with_mcm_mask:
        mcm_mask = (torch.rand_like(q_elem) < mcm_mask_ratio)
        mcm_mask[:, target_idx] = False
        q_elem = torch.where(mcm_mask, torch.zeros_like(q_elem), q_elem)
        q_bdl  = torch.where(mcm_mask, torch.zeros_like(q_bdl),  q_bdl)
    else:
        mcm_mask = None

    return {
        "q_elem":   q_elem,
        "q_bdl":    q_bdl,
        "n_elem_l": batch["n_elem"][:, sl_l],
        "n_bdl_l":  batch["n_bdl" ][:, sl_l],
        "rel_xy_l": batch["rel_xy"][:, sl_l],
        "dens_l":   batch["n_density"][:, sl_l],
        "n_elem_m": batch["n_elem"][:, sl_m],
        "n_bdl_m":  batch["n_bdl" ][:, sl_m],
        "rel_xy_m": batch["rel_xy"][:, sl_m],
        "dens_m":   batch["n_density"][:, sl_m],
        "n_elem_g": batch["n_elem"][:, sl_g],
        "n_bdl_g":  batch["n_bdl" ][:, sl_g],
        "rel_xy_g": batch["rel_xy"][:, sl_g],
        "dens_g":   batch["n_density"][:, sl_g],
        "mcm_mask": mcm_mask,
    }


def run_epoch(model, ds, nbr_idx, id_list, args, device, train: bool,
              optimizer=None):
    model.train() if train else model.eval()
    ctx = torch.enable_grad() if train else torch.no_grad()
    sums = {"target": 0.0, "mcm": 0.0, "bdl": 0.0, "con": 0.0,
            "var": 0.0, "cov": 0.0, "total": 0.0}
    n_seen = 0
    sl_l, sl_m, sl_g = split_neighbours(nbr_idx, args.k_local, args.k_mid,
                                        args.k_neighbors)

    with ctx:
        for q_ids in iter_batches(id_list, args.batch_size,
                                  shuffle=train, seed=args.seed):
            batch = collate(ds, nbr_idx, q_ids, device)
            B = batch["q_elem"].size(0)

            # ── two augmented views of the SAME queries ───────────────────
            view1 = make_view(batch, ds["target_idx"], sl_l, sl_m, sl_g,
                              args.elem_drop, args.bdl_flip,
                              args.mcm_mask_ratio, with_mcm_mask=True,
                              device=device)
            view2 = make_view(batch, ds["target_idx"], sl_l, sl_m, sl_g,
                              args.elem_drop * 1.3, args.bdl_flip * 1.5,
                              args.mcm_mask_ratio, with_mcm_mask=False,
                              device=device)

            keys = ["q_elem","q_bdl",
                    "n_elem_l","n_bdl_l","rel_xy_l","dens_l",
                    "n_elem_m","n_bdl_m","rel_xy_m","dens_m",
                    "n_elem_g","n_bdl_g","rel_xy_g","dens_g"]
            out1 = model.forward_pretrain(**{k: view1[k] for k in keys})
            out2 = model.forward_pretrain(**{k: view2[k] for k in keys})

            # Loss 1: target prediction (primary)
            L_tgt = F.mse_loss(out1["target_pred"], batch["q_target"])

            # Loss 2: MCM (secondary) on view1's masked positions only
            mcm_mask = view1["mcm_mask"]
            if mcm_mask.any():
                L_mcm = ((out1["mcm_value"] - batch["q_elem"]) ** 2
                         * mcm_mask).sum() / mcm_mask.sum()
                L_bdl = F.binary_cross_entropy_with_logits(
                    out1["mcm_bdl"][mcm_mask], batch["q_bdl"][mcm_mask])
            else:
                L_mcm = torch.zeros((), device=device)
                L_bdl = torch.zeros((), device=device)

            # Loss 3: hard contrastive — in-batch negatives (different queries!)
            L_con = info_nce_batch(out1["contrast_z"], out2["contrast_z"],
                                   T=args.info_nce_T)

            # Loss 4+5: VICReg anti-collapse on h_q
            # Variance hinge: each latent dim should have std ≥ 1.
            # Covariance: off-diagonal of cov(h_q) → 0 (decorrelate dims).
            h_q_1 = out1["h_q"]
            d     = h_q_1.size(1)
            std_q = torch.sqrt(h_q_1.var(dim=0) + 1e-4)
            L_var = torch.relu(1.0 - std_q).mean()
            h_c   = h_q_1 - h_q_1.mean(dim=0)
            cov   = (h_c.t() @ h_c) / max(h_q_1.size(0) - 1, 1)
            L_cov = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / d

            loss = (args.w_target * L_tgt
                    + args.w_mcm  * (L_mcm + 0.5 * L_bdl)
                    + args.w_con  * L_con
                    + args.w_var  * L_var
                    + args.w_cov  * L_cov)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            n_seen           += B
            sums["target"]   += L_tgt.item() * B
            sums["mcm"]      += L_mcm.item() * B
            sums["bdl"]      += L_bdl.item() * B
            sums["con"]      += L_con.item() * B
            sums["var"]      += L_var.item() * B
            sums["cov"]      += L_cov.item() * B
            sums["total"]    += loss .item() * B

    return {k: v / max(n_seen, 1) for k, v in sums.items()}


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[v3] loading {args.csv_path}")
    ds = load_area(
        args.csv_path, args.target_element,
        transform=args.transform, input_elements=args.input_elements,
        bdl_strategy=args.bdl_strategy, bdl_threshold=args.bdl_threshold,
        paper_compatible=args.paper_compatible,
    )
    N = len(ds["coords"]); C = ds["elem"].shape[1]
    print(f"[v3] N={N}  C={C}  target_idx={ds['target_idx']}  "
          f"K={args.k_neighbors} (local={args.k_local}, mid={args.k_mid}, "
          f"glob={args.k_neighbors - args.k_local - args.k_mid})")

    rng = np.random.default_rng(args.seed)
    n_val   = max(1, int(N * args.val_frac))
    val_idx = rng.choice(N, size=n_val, replace=False)
    tr_idx  = np.setdiff1d(np.arange(N), val_idx)

    nbr_idx = build_neighbour_index(ds["coords"], args.k_neighbors)

    use_scales = tuple(s.strip() for s in args.use_scales.split(",") if s.strip())
    model = HierarchicalBackbone(
        n_elem=C, d_model=args.d_model, n_heads=args.n_heads,
        n_layers_local=args.n_layers_local,
        n_layers_mid  =args.n_layers_mid,
        n_layers_glob =args.n_layers_glob,
        dropout=args.dropout,
        k_local=args.k_local, k_mid=args.k_mid,
        use_scales=use_scales,
        use_scale_tag=(not args.no_scale_tag),
    ).to(device)
    print(f"[v3] params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched     = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best = {"val_target": float("inf"), "epoch": -1}
    patience, history, t0 = 0, [], time.time()

    for ep in range(1, args.epochs + 1):
        if ep <= args.warmup:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * ep / args.warmup
        tr = run_epoch(model, ds, nbr_idx, tr_idx,  args, device, True,  optimizer)
        va = run_epoch(model, ds, nbr_idx, val_idx, args, device, False)
        if ep > args.warmup: sched.step()

        print(f"  Ep {ep:3d}/{args.epochs}  "
              f"tr_tot={tr['total']:.4f} tgt={tr['target']:.4f} "
              f"mcm={tr['mcm']:.4f} con={tr['con']:.4f}  | "
              f"va_tot={va['total']:.4f} tgt={va['target']:.4f}", end="")

        # selection on val_target (primary task)
        if va["target"] < best["val_target"]:
            best = {"val_target": va["target"], "epoch": ep}; patience = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args":             vars(args),
                "epoch":            ep,
                "n_elem":           C,
                "element_cols":     ds["element_cols"],
                "target_idx":       ds["target_idx"],
                "transform":        ds["transform"],
                "scaler_mean":      ds["scaler"].mean_,
                "scaler_scale":     ds["scaler"].scale_,
            }, os.path.join(args.out_dir, "backbone.pt"))
            print("  ← best")
        else:
            patience += 1
            print(f"  (no improve, patience={patience}/{args.early_patience})")

        history.append({"epoch": ep, **{f"tr_{k}": v for k, v in tr.items()},
                                      **{f"va_{k}": v for k, v in va.items()}})
        if patience >= args.early_patience:
            print(f"[v3] early stop at epoch {ep} (best={best['epoch']})")
            break

    json.dump(history, open(os.path.join(args.out_dir, "pretrain_history.json"),
                            "w"), indent=2)
    print(f"[v3] done.  best val_target={best['val_target']:.4f} @ ep {best['epoch']}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
