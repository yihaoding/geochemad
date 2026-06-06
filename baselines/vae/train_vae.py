#!/usr/bin/env python3
"""
train_vae.py  –  Train VAE-family models and score all samples.

Models (--model flag):
  ae              Convolutional Autoencoder
  dae             Denoising Autoencoder
  vae             Standard VAE
  vaegan          VAE + GAN discriminator
  vae_cascade_gan VAE + multi-scale decoder + GAN
  vaediff         VAE encoder + DDPM decoder
  all             Run all six models sequentially

Pipeline
--------
  1. Load patches from <out_root>/<area_id>/vae_patches/
  2. Train selected model(s), best checkpoint by val reconstruction loss
  3. Build full (H, W) anomaly pixel map from best model
  4. Map pixel scores → sample coordinates → anomaly_scores.csv
  5. Run spatial evaluation → metrics.json
"""

import argparse
import copy
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.stats import kurtosis, spearmanr
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # baselines/ for vae.models
from shared.area_config   import resolve_areas
from shared.feature_select import FS_CHOICES, FS_SUFFIX
from shared.preprocess     import save_scores, run_evaluate
from shared.evaluate       import evaluate as _evaluate_fn, _standardize_xy
from vae.models import (AEModel, DAEModel, VAEModel,
                        VAEGANModel, VAECascadeGANModel, VAEDiffModel)

ALL_MODELS = ["ae", "dae", "vae", "vaegan", "vae_cascade_gan", "vaediff"]

# Per-epoch selection criteria — must match gadformer's emitted columns
SELECT_CRITERIA = ["moran", "kurtosis", "p99_p50", "jaccard_late",
                   "train_loss", "auc"]


# ── Gadformer-style unsupervised metric helpers ───────────────────────────────

def build_knn_idx(coords, k=8):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    return idx[:, 1:]


def moran_i_rowstd(values, knn_idx):
    z = values - values.mean()
    lag = z[knn_idx].mean(axis=1)
    den = (z * z).sum()
    if den <= 0:
        return 0.0
    return float((z * lag).sum() / den)


def topk_jaccard(s_prev, s_cur, k):
    t1 = set(np.argpartition(-s_prev, k)[:k])
    t2 = set(np.argpartition(-s_cur,  k)[:k])
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union else 0.0


def compute_unsup_metrics(scores, knn_idx, prev_scores, k_top):
    out = {
        "moran_score":    moran_i_rowstd(scores, knn_idx),
        "kurtosis_score": float(kurtosis(scores)),
        "p99_over_p50":   float(np.percentile(scores, 99) /
                                max(np.percentile(scores, 50), 1e-12)),
    }
    if prev_scores is not None:
        out["jaccard_top5pct"] = topk_jaccard(prev_scores, scores, k_top)
        rho, _ = spearmanr(prev_scores, scores)
        out["spearman_rho"]    = float(rho)
    else:
        out["jaccard_top5pct"] = float("nan")
        out["spearman_rho"]    = float("nan")
    return out


# ── Dataset ───────────────────────────────────────────────────────────────────

class PatchDataset(Dataset):
    def __init__(self, patches_dir, patch_records, elem_folders, patch_size=16):
        self.patches_dir = patches_dir
        self.records     = patch_records
        self.folders     = sorted(elem_folders)
        self.size        = (patch_size, patch_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        fname = rec["image"]
        chans = []
        for folder in self.folders:
            p = os.path.join(self.patches_dir, folder, "patches", fname)
            if os.path.exists(p):
                with Image.open(p) as img:
                    arr = np.array(img.convert("L").resize(self.size),
                                   dtype=np.float32) / 255.0
                    t = torch.from_numpy(arr).unsqueeze(0)
                    t = (t - 0.5) / 0.5
                    chans.append(t)
            else:
                chans.append(torch.zeros(1, *self.size))
        return torch.cat(chans, dim=0), rec["label"]


# ── Loss helpers ──────────────────────────────────────────────────────────────

bce = nn.BCEWithLogitsLoss()
l1  = nn.L1Loss()

def _kl(mu, logvar):
    return 0.5 * torch.mean(torch.sum(logvar.exp() + mu**2 - 1.0 - logvar, dim=1))


class _Cfg:
    lambda_recon = 1.0
    lambda_kl    = 0.5
    lambda_adv   = 0.1
    lambda_fm    = 1.0


# ── Build model ───────────────────────────────────────────────────────────────

def build_model(model_name, C, args, device):
    S = args.img_size
    if model_name == "ae":
        m = AEModel(C, args.latent_dim, S)
        return {"vae": m}, None
    elif model_name == "dae":
        m = DAEModel(C, args.latent_dim, S, noise_std=args.dae_noise)
        return {"vae": m}, None
    elif model_name == "vae":
        m = VAEModel(C, args.latent_dim, S)
        return {"vae": m}, None
    elif model_name == "vaegan":
        m = VAEGANModel(C, args.latent_dim, S)
        return {"vae": m, "disc": m.disc}, m.disc
    elif model_name == "vae_cascade_gan":
        m = VAECascadeGANModel(C, args.latent_dim, S)
        return {"vae": m, "disc": m.disc}, m.disc
    elif model_name == "vaediff":
        m = VAEDiffModel(C, args.latent_dim, S, t_eval=args.diff_t_eval)
        return {"vae": m}, None
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_one_epoch(model_name, models, opts, loader, cfg, device,
                    scaler_g, scaler_d):
    vae  = models["vae"].train()
    disc = models.get("disc")
    if disc:
        disc.train()
    total_g = total_d = 0.0

    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        B = x.size(0)

        if model_name in ("ae", "dae"):
            opts["vae"].zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                x_hat = vae(x)
                loss  = l1(x_hat, x)
            scaler_g.scale(loss).backward()
            scaler_g.step(opts["vae"]); scaler_g.update()
            total_g += loss.item()

        elif model_name == "vae":
            opts["vae"].zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                x_hat, mu, lv = vae(x)
                loss = cfg.lambda_recon * l1(x_hat, x) + cfg.lambda_kl * _kl(mu, lv)
            scaler_g.scale(loss).backward()
            scaler_g.step(opts["vae"]); scaler_g.update()
            total_g += loss.item()

        elif model_name == "vaediff":
            opts["vae"].zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                diff_loss, kl = vae(x)
                loss = diff_loss + cfg.lambda_kl * kl
            scaler_g.scale(loss).backward()
            scaler_g.step(opts["vae"]); scaler_g.update()
            total_g += loss.item()

        elif model_name in ("vaegan", "vae_cascade_gan"):
            # D step
            opts["disc"].zero_grad(set_to_none=True)
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == "cuda")):
                if model_name == "vaegan":
                    x_hat, mu, lv = vae(x)
                    x_fake = vae.decoder(torch.randn(B, args_latent_dim(vae), device=device))
                else:
                    (x_hat, *_), mu, lv = vae(x)
                    x_fake = vae.decoder(torch.randn(B, args_latent_dim(vae), device=device))[0]
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                rl, _  = disc(x)
                fl, _  = disc(x_fake)
                rcl, _ = disc(x_hat.detach())
                d_loss = (bce(rl, torch.ones_like(rl)) +
                          0.5 * bce(rcl, torch.zeros_like(rcl)) +
                          0.5 * bce(fl,  torch.zeros_like(fl)))
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opts["disc"]); scaler_d.update()
            total_d += d_loss.item()

            # G step
            opts["vae"].zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device == "cuda")):
                if model_name == "vaegan":
                    x_hat, mu, lv = vae(x)
                    rec = l1(x_hat, x)
                    adv_l, fh = disc(x_hat)
                else:
                    (x_hat, xh, xq), mu, lv = vae(x)
                    rec = (l1(x_hat, x) + 0.5 * l1(xh, F.avg_pool2d(x, 2)) +
                           0.25 * l1(xq, F.avg_pool2d(x, 4)))
                    adv_l, fh = disc(x_hat)
                kl    = _kl(mu, lv)
                g_adv = bce(adv_l, torch.ones_like(adv_l))
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == "cuda")):
                _, fr = disc(x)
            fm = sum(F.mse_loss(a, b) for a, b in zip(fr, fh))
            g_loss = (cfg.lambda_recon * rec + cfg.lambda_kl * kl +
                      cfg.lambda_adv * g_adv + cfg.lambda_fm * fm)
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opts["vae"]); scaler_g.update()
            total_g += g_loss.item()

    n = max(len(loader), 1)
    return total_g / n, total_d / n


def args_latent_dim(vae):
    """Extract latent dim from model for sampling."""
    if hasattr(vae, 'latent_dim'):
        return vae.latent_dim
    return vae.encoder.fc_mu.out_features


@torch.no_grad()
def val_loss(model_name, vae, loader, device):
    vae.eval()
    tot = 0.0
    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            if model_name in ("ae", "dae"):
                x_hat = vae(x)
                tot += l1(x_hat, x).item()
            elif model_name == "vae":
                x_hat, _, _ = vae(x)
                tot += l1(x_hat, x).item()
            elif model_name == "vaegan":
                x_hat, _, _ = vae(x)
                tot += l1(x_hat, x).item()
            elif model_name == "vae_cascade_gan":
                (x_hat, *_), _, _ = vae(x)
                tot += l1(x_hat, x).item()
            elif model_name == "vaediff":
                diff_loss, kl = vae(x)
                tot += (diff_loss + 0.5 * kl).item()
    return tot / max(len(loader), 1)


# ── Anomaly map ───────────────────────────────────────────────────────────────

@torch.no_grad()
def build_anomaly_map(vae, dataset, device, n_rows, n_cols, patch_size):
    vae.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
    H, W   = n_rows * patch_size, n_cols * patch_size
    full   = np.zeros((H, W), dtype=np.float32)
    cnt    = np.zeros((H, W), dtype=np.float32)
    patch_idx = 0

    for batch in loader:
        x = (batch[0] if isinstance(batch, (list, tuple)) else batch).to(device)
        with torch.amp.autocast('cuda', enabled=(device == "cuda")):
            score = vae.score_pixel(x)
        scores_np = score.cpu().float().numpy()
        B = len(scores_np)
        for bi in range(B):
            rec = dataset.records[patch_idx + bi]
            r0, r1 = rec["row"] * patch_size, (rec["row"] + 1) * patch_size
            c0, c1 = rec["col"] * patch_size, (rec["col"] + 1) * patch_size
            full[r0:r1, c0:c1] += scores_np[bi]
            cnt [r0:r1, c0:c1] += 1.0
        patch_idx += B

    return full / np.maximum(cnt, 1.0)


# ── Pixel → sample scores ─────────────────────────────────────────────────────

def compute_pixel_indices(meta, area_csv):
    """Precompute (df_all, rows_px, cols_px) once, reused per epoch."""
    df   = pd.read_csv(area_csv, low_memory=False).dropna(subset=["X", "Y"])
    lons = df["X"].to_numpy(float)
    lats = df["Y"].to_numpy(float)

    if "utm_epsg" in meta:
        from pyproj import CRS, Transformer
        tr   = Transformer.from_crs(CRS.from_epsg(4326),
                                    CRS.from_epsg(meta["utm_epsg"]),
                                    always_xy=True)
        xm, ym = tr.transform(lons, lats)
        x0, y0 = meta["x_min_m"], meta["y_min_m"]
    elif "mean_lat_rad" in meta:
        mlr   = meta["mean_lat_rad"]
        xm    = lons * np.cos(mlr) * 111000.0
        ym    = lats * 111000.0
        x0, y0 = meta["x0_m"], meta["y0_m"]
    else:
        raise KeyError("metadata.json missing both 'utm_epsg' and 'mean_lat_rad'")

    res    = meta["resolution_m"]
    H, W   = meta["image_size"]
    cols_px = np.clip(np.round((xm - x0) / res).astype(int), 0, W - 1)
    rows_px = np.clip((H - 1) - np.round((ym - y0) / res).astype(int), 0, H - 1)
    return df, rows_px, cols_px


def pixel_to_sample_scores(anomaly_map, meta, area_csv):
    df, rows_px, cols_px = compute_pixel_indices(meta, area_csv)
    return df, anomaly_map[rows_px, cols_px]


# ── Per-model run ─────────────────────────────────────────────────────────────

def _effective_args(args, area_cfg):
    """
    Return a shallow copy of `args` with this area's hparams overlaid.
    CLI values stay as fallback when an area is missing a key.
    """
    eff = copy.copy(args)
    for k, v in (area_cfg.get("hparams") or {}).items():
        if hasattr(eff, k):
            setattr(eff, k, v)
    return eff


def run_area(area_cfg, args, device, model_name):
    area_id     = area_cfg["area_id"]
    args        = _effective_args(args, area_cfg)   # ← per-area override
    if getattr(args, "max_epochs", 0) > 0:
        args.epochs = min(args.epochs, args.max_epochs)
    suffix      = FS_SUFFIX[args.feature_selection]
    transform   = getattr(args, "transform", "clr")
    tsuf        = "" if transform == "clr" else f"_{transform}"
    patches_dir = os.path.join(args.out_root, area_id, f"vae_patches{suffix}{tsuf}")
    split_path  = os.path.join(patches_dir, "dataset_split.pkl")
    meta_path   = os.path.join(patches_dir, "metadata.json")
    out_dir     = os.path.join(args.out_root, area_id, f"{model_name}{suffix}{tsuf}")

    if not os.path.exists(split_path):
        print(f"  [skip] Patches not found for {area_id} (fs={args.feature_selection})")
        return

    # Skip if already done (metrics.json exists)
    if args.resume and os.path.exists(os.path.join(out_dir, "metrics.json")):
        print(f"  [done] {area_id}/{model_name} — skipping")
        return

    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*60}\n  {model_name.upper()}  {area_id}\n{'='*60}")
    if area_cfg.get("hparams"):
        print(f"  hparams: latent={args.latent_dim} epochs={args.epochs} "
              f"batch={args.batch_size} lr={args.lr:.1e} lr_d={args.lr_d:.1e} "
              f"min_dist={args.min_dist}km radius={args.radius_km}km")

    with open(split_path, "rb") as f:
        split = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)

    patch_size     = split["patch_size_pixels"]
    n_rows, n_cols = split["num_patches"]
    records        = split["train"]

    elem_folders = sorted(
        d for d in os.listdir(patches_dir)
        if os.path.isdir(os.path.join(patches_dir, d)) and d.startswith("patches_")
    )
    C = len(elem_folders)
    if C == 0:
        print("  [skip] No element folders found")
        return
    print(f"  {C} channels, {len(records)} patches, {n_rows}×{n_cols} grid")

    ds_full   = PatchDataset(patches_dir, records, elem_folders, patch_size)
    tr_loader = DataLoader(ds_full, batch_size=args.batch_size,
                           shuffle=True, num_workers=2, pin_memory=(device == "cuda"))

    models, disc = build_model(model_name, C, args, device)
    for m in models.values():
        m.to(device)
    vae = models["vae"]

    opts = {"vae": torch.optim.Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.999))}
    if disc:
        opts["disc"] = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    scaler_g = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    scaler_d = torch.amp.GradScaler('cuda', enabled=(device == "cuda"))
    cfg      = _Cfg()

    ckpt_best = os.path.join(out_dir, "checkpoint_best.pt")
    ckpt_last = os.path.join(out_dir, "checkpoint_last.pt")
    scores_dir = os.path.join(out_dir, "scores")
    ckpts_dir  = os.path.join(out_dir, "ckpts")
    os.makedirs(scores_dir, exist_ok=True)
    if args.save_ckpts:
        os.makedirs(ckpts_dir, exist_ok=True)

    def _save(path):
        torch.save({
            "model_state_dict": vae.state_dict(),
            "n_channels": C,
            "latent_dim": args.latent_dim,
            "img_size":   args.img_size,
            "model_name": model_name,
            "area_id":    area_id,
        }, path)

    # ── Precompute pixel→sample mapping + Moran KNN (once) ────────────────────
    df_all_full, rows_px, cols_px = compute_pixel_indices(meta, area_cfg["csv_path"])
    coords = df_all_full[["X", "Y"]].to_numpy(float)
    knn_idx = build_knn_idx(coords, k=args.knn_k)
    N_samp  = len(coords)
    k_top   = max(1, int(args.top_pct * N_samp))

    # Cache deposits XY once for direct (no-CSV-roundtrip) evaluation
    if not args.no_label_metrics:
        dep_df = _standardize_xy(pd.read_csv(area_cfg["deposits"]))
        dep_xy = dep_df[["x", "y"]].to_numpy(float)
        samp_xy_arr = df_all_full[["X", "Y"]].to_numpy(float)
    else:
        dep_xy = None
        samp_xy_arr = None

    # ── Per-epoch state ───────────────────────────────────────────────────────
    metrics_csv = os.path.join(out_dir, "metrics_per_epoch.csv")
    rows        = []
    prev_scores = None
    crit        = args.select_criterion
    # max-better criteria
    max_better  = crit in ("moran", "kurtosis", "p99_p50", "jaccard_late", "auc")
    best_val    = -float("inf") if max_better else float("inf")
    best_epoch  = -1
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        gl, dl = train_one_epoch(model_name, models, opts, tr_loader, cfg,
                                  device, scaler_g, scaler_d)

        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)
        tag = ""
        row = {"epoch": epoch, "train_g": gl, "train_d": dl}

        if do_eval:
            amap = build_anomaly_map(vae, ds_full, device,
                                     n_rows, n_cols, patch_size)
            scores_vec = amap[rows_px, cols_px]
            np.save(os.path.join(scores_dir, f"epoch_{epoch:03d}.npy"),
                    scores_vec)

            unsup = compute_unsup_metrics(
                scores_vec, knn_idx, prev_scores, k_top)
            row.update(unsup)

            if not args.no_label_metrics:
                lab = _evaluate_fn(samp_xy_arr, scores_vec, dep_xy,
                                   radius_km = args.radius_km,
                                   min_dist  = args.min_dist,
                                   neg_runs  = args.neg_runs,
                                   label     = "")
                for k in ("AUC_mean", "AP_mean", "PR_AUC_mean",
                          "SR_5pct", "Recall_50", "DTD_km", "Density"):
                    if k in lab:
                        row[k] = lab[k]

            # Selection by chosen criterion
            half = max(1, args.epochs // 2)
            crit_val = {
                "moran":        row.get("moran_score", -np.inf),
                "kurtosis":     row.get("kurtosis_score", -np.inf),
                "p99_p50":      row.get("p99_over_p50", -np.inf),
                "jaccard_late": (row.get("jaccard_top5pct", -np.inf)
                                 if epoch >= half else -np.inf),
                "train_loss":   gl,
                "auc":          row.get("AUC_mean", -np.inf),
            }[crit]

            improved = (crit_val > best_val) if max_better else (crit_val < best_val)
            if improved and np.isfinite(crit_val):
                best_val   = crit_val
                best_epoch = epoch
                _save(ckpt_best)
                tag = f"  ← best ({crit}={crit_val:.4f})"

            if args.save_ckpts:
                _save(os.path.join(ckpts_dir, f"epoch_{epoch:03d}.pt"))

            prev_scores = scores_vec.copy()
            rows.append(row)
            pd.DataFrame(rows).to_csv(metrics_csv, index=False)

        # train-loss fallback: when criterion is train_loss we don't need eval
        if crit == "train_loss" and not do_eval:
            if gl < best_val:
                best_val   = gl
                best_epoch = epoch
                _save(ckpt_best)
                tag = f"  ← best (train_loss={gl:.4f})"

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"g={gl:.4f}  d={dl:.4f} | "
              f"{(time.time()-t0)/60:.1f}min{tag}", flush=True)

    _save(ckpt_last)

    # ── Post-hoc: regenerate final anomaly_scores.csv + metrics.json ──────────
    if os.path.exists(ckpt_best):
        best = torch.load(ckpt_best, map_location=device, weights_only=False)
        vae.load_state_dict(best["model_state_dict"])
    amap = build_anomaly_map(vae, ds_full, device, n_rows, n_cols, patch_size)
    np.save(os.path.join(out_dir, "anomaly_map.npy"), amap)
    final_scores = amap[rows_px, cols_px]
    scores_csv = save_scores(df_all_full, final_scores, out_dir)
    print(f"  anomaly_scores.csv → {scores_csv}")

    metrics = run_evaluate(
        scores_csv   = scores_csv,
        deposits_csv = area_cfg["deposits"],
        out_dir      = out_dir,
        radius_km    = args.radius_km,
        min_dist     = args.min_dist,
        neg_runs     = args.neg_runs,
        label        = f"{area_id}/{model_name}",
    )
    print(f"  [selected ep={best_epoch} by {crit}]  "
          f"AUC={metrics['AUC_mean']:.4f}  AP={metrics['AP_mean']:.4f}  "
          f"SR@5%={metrics['SR_5pct']:.4f}  DTD={metrics['DTD_km']:.2f}km")

    # ── Selection log: best-epoch-by-each-criterion (gadformer parity) ────────
    if rows:
        df_e = pd.DataFrame(rows)
        sel = {}
        if "moran_score" in df_e:
            sel["moran"]        = int(df_e["epoch"].iloc[df_e["moran_score"].idxmax()])
        if "kurtosis_score" in df_e:
            sel["kurtosis"]     = int(df_e["epoch"].iloc[df_e["kurtosis_score"].idxmax()])
        if "p99_over_p50" in df_e:
            sel["p99_p50"]      = int(df_e["epoch"].iloc[df_e["p99_over_p50"].idxmax()])
        if ("jaccard_top5pct" in df_e
                and df_e["jaccard_top5pct"].notna().sum() > 1):
            late = df_e.iloc[max(0, len(df_e)//2):].dropna(subset=["jaccard_top5pct"])
            if len(late) > 0:
                sel["jaccard_late"] = int(late["epoch"].iloc[
                    late["jaccard_top5pct"].argmax()])
        sel["train_loss"]       = int(df_e["epoch"].iloc[df_e["train_g"].idxmin()])
        if "AUC_mean" in df_e:
            sel["auc"]          = int(df_e["epoch"].iloc[df_e["AUC_mean"].idxmax()])

        with open(os.path.join(out_dir, "selection_log.json"), "w") as f:
            json.dump({
                "selected_criterion":   crit,
                "selected_epoch":       best_epoch,
                "best_epoch_per_crit":  sel,
                "epochs_evaluated":     len(df_e),
            }, f, indent=2)
        print(f"  best-epoch-per-criterion: {sel}")


# ── Main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--areas",        default="all")
    p.add_argument("--model",        default="all",
                   choices=ALL_MODELS + ["all"])
    p.add_argument("--out_root",     default="outputs")
    p.add_argument("--latent_dim",   type=int,   default=64)
    p.add_argument("--img_size",     type=int,   default=16)
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--lr_d",         type=float, default=1e-4)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--radius_km",    type=float, default=2.0)
    p.add_argument("--min_dist",     type=float, default=0.5)
    p.add_argument("--neg_runs",     type=int,   default=20)
    p.add_argument("--dae_noise",    type=float, default=0.2)
    p.add_argument("--diff_t_eval",  type=int,   default=20)
    p.add_argument("--transform",    default="clr",
                   choices=["clr", "ilr", "none"],
                   help="Reads patches dir suffixed by transform (e.g. "
                        "vae_patches_fs_ilr/). Must match the patches you "
                        "generated with generate_vae_patches.py --transform X.")
    p.add_argument("--feature_selection", default="none", choices=FS_CHOICES,
                   help="Reads vae_patches{,_fs,_pca}/ and writes "
                        "<model>{,_fs,_pca}/. Must match what was used at "
                        "patch-generation time.")
    p.add_argument("--resume",       action="store_true",
                   help="Skip area/model combos that already have metrics.json")
    p.add_argument("--max_epochs",   type=int, default=0,
                   help="If > 0, cap effective epochs (applied after per-area "
                        "hparams override). Useful for smoke tests.")

    # Gadformer-style checkpoint selection
    p.add_argument("--select_criterion", default="moran",
                   choices=SELECT_CRITERIA,
                   help="Per-epoch criterion that decides checkpoint_best.pt. "
                        "moran/kurtosis/p99_p50/jaccard_late are unsupervised "
                        "(gadformer-style). train_loss is the legacy default. "
                        "auc uses labeled deposits (snooping; for ablation).")
    p.add_argument("--eval_every",    type=int, default=1,
                   help="Eval+log every N epochs. Heavy for vaediff; "
                        "raise to 5 there if too slow.")
    p.add_argument("--knn_k",         type=int, default=8,
                   help="k for Moran's I KNN on sample coords")
    p.add_argument("--top_pct",       type=float, default=0.05,
                   help="top fraction for Jaccard stability")
    p.add_argument("--save_ckpts",    action="store_true",
                   help="Save per-epoch ckpts/epoch_NNN.pt (~10–20MB each)")
    p.add_argument("--no_label_metrics", action="store_true",
                   help="Skip AUC/AP/SR/DTD computation each epoch "
                        "(faster; still computed at the end on selected ckpt)")
    return p.parse_args()


def main():
    args   = get_args()
    areas  = resolve_areas(args.areas)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    models_to_run = ALL_MODELS if args.model == "all" else [args.model]
    print(f"Device: {device}  |  Models: {models_to_run}  |  "
          f"Areas: {len(areas)}  |  FS: {args.feature_selection}")

    for area_cfg in areas:
        for model_name in models_to_run:
            run_area(area_cfg, args, device, model_name)

    print("\nAll done.")


if __name__ == "__main__":
    main()
