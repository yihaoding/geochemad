#!/usr/bin/env python3
"""
run_t1_cl.py  –  T1-CL: exact port of new_geo_transformer.py into geochemadv2.

Architecture and training are identical to new_geo_transformer.py:
  - GeoTransformer: PosEncMLP + TransformerEncoder + TransformerDecoder
  - nhead=4, num_enc_layers=2, num_dec_layers=2, dim_feedforward=128  (from main.py)
  - Loss: L1(recon_all, t) + L1(recon_target, tfeats)*[if tfeats] + 0.01*MSE(r1, r2)
  - Contrastive: _mask_view on spatial context → two independent augmented views
  - Score: L2 norm  sqrt(dot(diff, diff))  per sample (not MSE)
  - Best checkpoint: UNSUPERVISED score-distribution separation
        default = top_gap_mad = (mean(top-k%) - median) / MAD   (higher = better)
        alternatives: kurtosis | skewness | bimodality | loss
        (original loss-based behavior is still selectable via --ckpt_metric loss)
  - tfeats = None here (no SCL pretraining), so l_rec_target = 0

Neighbour strategy (matches run_geo_transformer.sh --no_distance_constraint):
  - Small areas (N <= 2000): distance_constraint=False → ALL other samples as context
    capped at max_ctx = N-1 (same as GeoDataset default max_ctx=1392 ≈ area1 N)
  - Large areas  (N >  2000): distance_constraint=True, radius_km=50, max_ctx=1024

Outputs per area (<out_root>/<area_id>/t1_cl/):
  ckpt_best.pt / ckpt_last.pt
  anomaly_scores_best.csv / anomaly_scores_last.csv
  metrics_best.json / metrics_last.json
"""

import argparse
import datetime
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_SCRIPTS = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _SCRIPTS)
from shared.area_config import resolve_areas
from shared.preprocess  import load_area, save_scores, run_evaluate

CONTRASTIVE_WEIGHT = 0.01   # identical to new_geo_transformer.py


# ── Model (exact copy from new_geo_transformer.py) ────────────────────────────

class PosEncMLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
    def forward(self, rel_xy):
        return self.net(rel_xy)


class GeoTransformer(nn.Module):
    def __init__(self, feat_dim, d_model=128, nhead=2,
                 num_enc_layers=4, num_dec_layers=4, dim_feedforward=128):
        super().__init__()
        self.in_proj         = nn.Linear(feat_dim, d_model)
        self.pos_mlp         = PosEncMLP(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder         = nn.TransformerEncoder(enc, num_layers=num_enc_layers)
        self.decoder         = nn.TransformerDecoder(dec, num_layers=num_dec_layers)
        self.out_proj_all    = nn.Linear(d_model, feat_dim)
        self.out_proj_target = nn.Linear(d_model, d_model)

    def forward(self, target_feat, target_xy, ctx_feats, ctx_xy):
        rel = ctx_xy - target_xy
        pos = self.pos_mlp(rel)
        ctx = self.in_proj(ctx_feats) + pos
        tgt = self.in_proj(target_feat)
        mem = self.encoder(ctx)
        dec = self.decoder(tgt, mem)
        return self.out_proj_all(dec), self.out_proj_target(dec), dec


# ── Dataset (exact copy from new_geo_transformer.py GeoDataset) ───────────────

class GeoDataset(Dataset):
    def __init__(self, feats, coords_xy, tfeats=None,
                 max_mask_ratio=0.5, radius_km=50.0, max_ctx=512,
                 distance_constraint=True, use_haversine=None):
        self.X   = feats.astype(np.float32)
        self.XY  = coords_xy.astype(np.float32)
        self.N   = self.X.shape[0]
        self.F   = self.X.shape[1]
        self.max_mask_ratio     = max_mask_ratio
        self.radius_km          = float(radius_km)
        self.max_ctx            = int(max_ctx)
        self.distance_constraint = bool(distance_constraint)

        # Auto-detect lat/lon vs projected (same logic as original)
        if use_haversine is None:
            use_haversine = (np.abs(self.XY[:, 0]).max() <= 180 and
                             np.abs(self.XY[:, 1]).max() <= 90)
        self.use_haversine = use_haversine

        if tfeats is not None:
            self.TF         = tfeats.astype(np.float32)
            self.hidden_dim = self.TF.shape[1]
        else:
            self.TF         = None
            self.hidden_dim = 0

        if self.use_haversine:
            self._rad = np.deg2rad(self.XY.astype(np.float64))
        else:
            self._rad = None

    def __len__(self):
        return self.N

    def _neighbor_indices(self, idx):
        r_m = self.radius_km * 1000.0
        if self.distance_constraint:
            if self.use_haversine:
                tgt  = self._rad[idx]
                dlat = self._rad[:, 0] - tgt[0]
                dlon = self._rad[:, 1] - tgt[1]
                s    = (np.sin(dlat / 2.0) ** 2
                        + np.cos(tgt[0]) * np.cos(self._rad[:, 0])
                        * np.sin(dlon / 2.0) ** 2)
                dists = 2.0 * 6371000.0 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))
            else:
                dists = np.linalg.norm(self.XY - self.XY[idx], axis=1)
            mask      = dists <= r_m
            mask[idx] = False
            idxs      = np.where(mask)[0]
            if idxs.size == 0:
                return idxs
            order = np.argsort(dists[idxs])
            return idxs[order][: self.max_ctx]
        else:
            idxs = np.delete(np.arange(self.N), idx)
            return idxs[: self.max_ctx]

    def _mask_view(self, X_ctx, XY_ctx):
        """Randomly mask a fraction of context rows (independent draw per view)."""
        if self.max_mask_ratio <= 0 or len(X_ctx) == 0:
            return X_ctx, XY_ctx
        ratio = np.random.rand() * self.max_mask_ratio
        keep  = np.random.rand(len(X_ctx)) > ratio
        if not np.any(keep):
            keep[np.random.randint(len(keep))] = True
        return X_ctx[keep], XY_ctx[keep]

    def __getitem__(self, idx):
        x_i    = self.X[idx]
        xy_i   = self.XY[idx]
        tf_i   = (self.TF[idx] if self.TF is not None
                  else np.zeros(self.hidden_dim, dtype=np.float32))

        neigh  = self._neighbor_indices(idx)
        ctxX   = self.X[neigh]  if len(neigh) > 0 else np.zeros((0, self.F), dtype=np.float32)
        ctxXY  = self.XY[neigh] if len(neigh) > 0 else np.zeros((0, 2),     dtype=np.float32)

        Xr, Yr = self._mask_view(ctxX, ctxXY)
        X1, Y1 = self._mask_view(ctxX, ctxXY)
        X2, Y2 = self._mask_view(ctxX, ctxXY)

        return {"target": x_i, "txy": xy_i, "tfeats": tf_i,
                "Xr": Xr, "Yr": Yr,
                "X1": X1, "Y1": Y1,
                "X2": X2, "Y2": Y2,
                "F": self.F, "M": self.max_ctx}


def _collate(batch):
    B = len(batch)
    F = batch[0]["F"]
    M = batch[0]["M"]

    def stack_scalar(key):
        arr = np.stack([b[key] for b in batch], axis=0)
        return torch.tensor(arr[:, None, :], dtype=torch.float32)

    def stack_hidden():
        arr = np.stack([b["tfeats"] for b in batch], axis=0)
        return torch.tensor(arr[:, None, :], dtype=torch.float32) if arr.shape[1] > 0 \
               else torch.zeros(B, 1, 0, dtype=torch.float32)

    def pad_stack(xkey, ykey):
        X_out = np.zeros((B, M, F), dtype=np.float32)
        Y_out = np.zeros((B, M, 2), dtype=np.float32)
        for i, b in enumerate(batch):
            m = min(b[xkey].shape[0], M)
            if m > 0:
                X_out[i, :m] = b[xkey][:m]
                Y_out[i, :m] = b[ykey][:m]
        return (torch.tensor(X_out, dtype=torch.float32),
                torch.tensor(Y_out, dtype=torch.float32))

    tgt    = stack_scalar("target")
    txy    = stack_scalar("txy")
    tfeats = stack_hidden()
    Xr, Yr = pad_stack("Xr", "Yr")
    X1, Y1 = pad_stack("X1", "Y1")
    X2, Y2 = pad_stack("X2", "Y2")
    return {"t": tgt, "txy": txy, "tfeats": tfeats,
            "Xr": Xr, "Yr": Yr, "X1": X1, "Y1": Y1, "X2": X2, "Y2": Y2}


# ── Training (exact copy of train_one_epoch from new_geo_transformer.py) ──────

def _train_one_epoch(model, loader, optimizer, device, epoch, total_epochs,
                     contrastive_weight):
    model.train()
    R_all, R_target, C_sum = 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}]",
                unit="batch", leave=False)

    for b in pbar:
        t,  txy, tfeats = b["t"].to(device),  b["txy"].to(device), b["tfeats"].to(device)
        Xr, Yr          = b["Xr"].to(device), b["Yr"].to(device)
        X1, Y1          = b["X1"].to(device), b["Y1"].to(device)
        X2, Y2          = b["X2"].to(device), b["Y2"].to(device)

        optimizer.zero_grad()

        recon_all, recon_target, _ = model(t, txy, Xr, Yr)
        l_rec_all = F.l1_loss(recon_all, t)

        has_tfeats   = tfeats.shape[-1] > 0
        l_rec_target = (F.l1_loss(recon_target, tfeats) if has_tfeats
                        else recon_target.new_zeros(1).squeeze())

        r1, *_ = model(t, txy, X1, Y1)
        r2, *_ = model(t, txy, X2, Y2)
        l_con  = F.mse_loss(r1, r2)

        loss = l_rec_all + l_rec_target + contrastive_weight * l_con
        loss.backward()
        optimizer.step()

        R_all    += l_rec_all.item()
        R_target += l_rec_target.item()
        C_sum    += l_con.item()

        pbar.set_postfix({
            "L1":    f"{l_rec_all.item():.4f}",
            "Con":   f"{l_con.item():.4f}",
            "Total": f"{loss.item():.4f}",
        })

    n = len(loader)
    return R_all / n, C_sum / n, R_target / n


# ── Inference (exact copy of infer_scores from new_geo_transformer.py) ─────────
# For large N, use radius-based context instead of leave-one-out to keep it tractable.

def _infer_scores(model, X_np, XY_np, radius_km, max_ctx, distance_constraint, device):
    """
    Anomaly score per sample = L2 norm sqrt(dot(diff, diff)).

    distance_constraint=False: context = all other samples capped at max_ctx
                                (mirrors --no_distance_constraint in original)
    distance_constraint=True : context = radius_km KDTree neighbours
    """
    model.eval()
    N = X_np.shape[0]

    use_haversine = (np.abs(XY_np[:, 0]).max() <= 180 and
                     np.abs(XY_np[:, 1]).max() <= 90)

    if use_haversine:
        cx = np.cos(np.radians(XY_np[:, 1].mean()))
        km = XY_np * np.array([111.0 * cx, 111.0])
    else:
        km = XY_np / 1000.0
    tree = KDTree(km)

    X_t  = torch.as_tensor(X_np,  dtype=torch.float32, device=device)
    XY_t = torch.as_tensor(XY_np, dtype=torch.float32, device=device)
    all_idx = np.arange(N)

    scores = np.zeros(N, dtype=np.float32)

    with torch.no_grad():
        for i in range(N):
            if not distance_constraint:
                # --no_distance_constraint: all other samples, capped at max_ctx
                idx_r = np.delete(all_idx, i)[:max_ctx]
            else:
                idx_r = tree.query_radius(km[i:i+1], r=radius_km)[0]
                idx_r = idx_r[idx_r != i]
                if len(idx_r) == 0:
                    idx_r = tree.query(km[i:i+1], k=min(16, N))[1][0]
                    idx_r = idx_r[idx_r != i]
                if len(idx_r) > max_ctx:
                    idx_r = idx_r[:max_ctx]

            t_t  = X_t[i].unsqueeze(0).unsqueeze(0)
            txy  = XY_t[i].unsqueeze(0).unsqueeze(0)
            ctx  = X_t[idx_r].unsqueeze(0)
            cxy  = XY_t[idx_r].unsqueeze(0)

            rec, _, _ = model(t_t, txy, ctx, cxy)
            diff      = X_np[i] - rec.squeeze().cpu().numpy()
            scores[i] = float(np.sqrt(np.dot(diff, diff)))

    return scores


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")

def _log(msg):
    print(f"[{_now()}] {msg}", flush=True)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--areas",              default="all")
    p.add_argument("--out_root",           default="outputs")
    p.add_argument("--method_name",        default="t1_cl",
                   help="Output subdir under <out_root>/<area_id>/. Use a "
                        "distinct name (e.g. t1_cl_topgap) to keep prior "
                        "loss-based runs intact.")
    p.add_argument("--transform",          default="clr", choices=["clr", "none"])
    p.add_argument("--d_model",            type=int,   default=128)
    p.add_argument("--nhead",              type=int,   default=4)
    p.add_argument("--num_enc_layers",     type=int,   default=2)
    p.add_argument("--num_dec_layers",     type=int,   default=2)
    p.add_argument("--dim_feedforward",    type=int,   default=128)
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--batch_size",         type=int,   default=32)
    p.add_argument("--radius_km",          type=float, default=50.0,
                   help="Context radius when distance_constraint=True (km)")
    p.add_argument("--max_ctx",            type=int,   default=1392,
                   help="Max context size per sample (default 1392 = GeoDataset default)")
    p.add_argument("--max_mask_ratio",     type=float, default=0.5,
                   help="Max fraction of context rows to mask per view")
    p.add_argument("--contrastive_weight", type=float, default=CONTRASTIVE_WEIGHT)
    p.add_argument("--ckpt_metric",        default="top_gap_mad",
                   choices=["top_gap_mad", "kurtosis", "skewness",
                            "bimodality", "loss"],
                   help="Best-checkpoint selection criterion. Default "
                        "'top_gap_mad' is unsupervised score-separation; "
                        "'loss' restores the original lowest-train-loss rule.")
    p.add_argument("--top_frac",           type=float, default=0.05,
                   help="Top fraction used by top_gap_mad metric (default 0.05).")
    p.add_argument("--ckpt_warmup_epochs", type=int,   default=3,
                   help="Ignore the first N epochs when picking best ckpt "
                        "(early epochs have noisy score distributions).")
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--eval_radius_km",     type=float, default=2.0)
    p.add_argument("--min_dist",           type=float, default=0.5)
    p.add_argument("--neg_runs",           type=int,   default=20)
    return p.parse_args()


def _save_ckpt(model, args, feature_cols, area_id, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols":     feature_cols,
        "area_id":          area_id,
        "d_model":          args.d_model,
        "nhead":            args.nhead,
        "num_enc_layers":   args.num_enc_layers,
        "num_dec_layers":   args.num_dec_layers,
    }, path)


def _score_separation(scores, metric="top_gap_mad", top_frac=0.05):
    """
    Unsupervised checkpoint-selection metric (higher = anomalies more clearly
    separated from the bulk):
      top_gap_mad : (mean(top-k%) - median) / MAD          (robust, default)
      kurtosis    : excess kurtosis                         (heavy-tailed)
      skewness    : right-skewness                          (positive tail mass)
      bimodality  : Pearson bimodality coefficient          ((g^2 + 1) / kt')
    Returns 0.0 if too few finite scores.
    """
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n < 8:
        return 0.0
    if metric == "top_gap_mad":
        k   = max(1, int(np.ceil(n * top_frac)))
        top = np.partition(s, -k)[-k:]
        med = np.median(s)
        mad = np.median(np.abs(s - med))
        return float((top.mean() - med) / (mad + 1e-12))
    if metric == "kurtosis":
        from scipy.stats import kurtosis
        return float(kurtosis(s, fisher=True, bias=False))
    if metric == "skewness":
        from scipy.stats import skew
        return float(skew(s, bias=False))
    if metric == "bimodality":
        from scipy.stats import kurtosis, skew
        g  = skew(s, bias=False)
        kt = kurtosis(s, fisher=False, bias=False)   # non-Fisher (normal=3)
        denom = kt + 3.0 * (n - 1) ** 2 / max(1.0, (n - 2) * (n - 3))
        return float((g * g + 1.0) / denom) if denom > 0 else 0.0
    raise ValueError(f"unknown ckpt_metric: {metric}")


def _evaluate(scores, df_all, area_cfg, args, out_dir, fname_suffix):
    scores_csv = save_scores(df_all, scores, out_dir,
                             fname=f"anomaly_scores_{fname_suffix}.csv")
    return run_evaluate(
        scores_csv    = scores_csv,
        deposits_csv  = area_cfg["deposits"],
        out_dir       = out_dir,
        radius_km     = args.eval_radius_km,
        min_dist      = args.min_dist,
        neg_runs      = args.neg_runs,
        label         = f"{area_cfg['area_id']}/{args.method_name}/{fname_suffix}",
        metrics_fname = f"metrics_{fname_suffix}.json",
    )


# ── Per-area training ─────────────────────────────────────────────────────────

def run_area(area_cfg, args, device):
    area_id = area_cfg["area_id"]
    out_dir = os.path.join(args.out_root, area_id, args.method_name)
    os.makedirs(out_dir, exist_ok=True)

    print(flush=True)
    print("=" * 60, flush=True)
    _log(f"T1-CL  {area_id}  ({args.method_name})  START")
    print("=" * 60, flush=True)

    _, _, df_all, feature_cols, _, _ = load_area(
        area_cfg, transform=args.transform, val_frac=0.0, seed=args.seed)

    X_np  = df_all[feature_cols].to_numpy(dtype=np.float32)
    XY_np = df_all[["X", "Y"]].to_numpy(dtype=np.float32)
    N, C  = X_np.shape

    # Neighbour strategy matching run_geo_transformer.sh:
    #   small (N<=2000): --no_distance_constraint, context = all other samples
    #   large (N> 2000): distance_constraint=True, radius_km=50
    use_dc  = N > 2000
    max_ctx = min(N - 1, args.max_ctx)   # never exceed N-1

    _log(f"  N={N}  features={C}  distance_constraint={use_dc}"
         f"  radius_km={args.radius_km if use_dc else 'N/A'}"
         f"  max_ctx={max_ctx}  epochs={args.epochs}  device={device}")

    dataset = GeoDataset(X_np, XY_np, tfeats=None,
                         max_mask_ratio=args.max_mask_ratio,
                         radius_km=args.radius_km,
                         max_ctx=max_ctx,
                         distance_constraint=use_dc)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, collate_fn=_collate,
                         num_workers=0)

    model = GeoTransformer(
        feat_dim       = C,
        d_model        = args.d_model,
        nhead          = args.nhead,
        num_enc_layers = args.num_enc_layers,
        num_dec_layers = args.num_dec_layers,
        dim_feedforward= args.dim_feedforward,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_best = os.path.join(out_dir, "ckpt_best.pt")
    ckpt_last = os.path.join(out_dir, "ckpt_last.pt")

    use_loss_criterion = (args.ckpt_metric == "loss")
    crit_name = "loss(↓)" if use_loss_criterion else f"{args.ckpt_metric}(↑)"
    best_criterion = float("inf") if use_loss_criterion else float("-inf")
    best_epoch, best_metrics, best_sep = -1, None, None

    _log(f"  ckpt selection: {crit_name}  warmup={args.ckpt_warmup_epochs}"
         + (f"  top_frac={args.top_frac}" if args.ckpt_metric == "top_gap_mad" else ""))

    print(f"\n  {'Ep':>4}  {'L1':>8}  {'Con':>8}  {'Total':>8}"
          f"  {'Sep':>8}  {'AUC':>7}  {'AP':>7}  {'SR@5%':>7}  {'DTD':>8}  note",
          flush=True)
    print(f"  {'-'*86}", flush=True)

    for epoch in range(1, args.epochs + 1):
        l1_avg, con_avg, _ = _train_one_epoch(
            model, loader, optimizer, device,
            epoch, args.epochs, args.contrastive_weight)

        total = l1_avg + args.contrastive_weight * con_avg

        scores  = _infer_scores(model, X_np, XY_np,
                                 args.radius_km, max_ctx, use_dc, device)
        metrics = _evaluate(scores, df_all, area_cfg, args, out_dir, "current")
        auc     = metrics["AUC_mean"]

        sep = _score_separation(scores, args.ckpt_metric, args.top_frac) \
              if not use_loss_criterion else float("nan")

        eligible = (epoch > args.ckpt_warmup_epochs)
        if use_loss_criterion:
            improved = eligible and (total < best_criterion - 1e-8)
            new_crit = total
        else:
            improved = eligible and (sep   > best_criterion + 1e-8)
            new_crit = sep

        note = ""
        if improved:
            best_criterion = new_crit
            best_epoch, best_metrics, best_sep = epoch, metrics, sep
            _save_ckpt(model, args, feature_cols, area_id, ckpt_best)
            for src_name, dst_name in [
                ("anomaly_scores_current.csv", "anomaly_scores_best.csv"),
                ("metrics_current.json",       "metrics_best.json"),
            ]:
                src = os.path.join(out_dir, src_name)
                if os.path.exists(src):
                    shutil.copy(src, os.path.join(out_dir, dst_name))
            note = " ← best"

        sep_disp = "    n/a " if use_loss_criterion else f"{sep:+8.3f}"
        print(f"  [{_now()}] {epoch:4d}  {l1_avg:8.5f}  {con_avg:8.5f}  {total:8.5f}"
              f"  {sep_disp}  {auc:7.4f}  {metrics['AP_mean']:7.4f}"
              f"  {metrics['SR_5pct']:7.4f}  {metrics['DTD_km']:7.2f}km{note}",
              flush=True)

    # Fallback: warmup never crossed (e.g. epochs <= warmup) → take last as best.
    if best_metrics is None:
        _log("  no eligible best epoch (warmup >= epochs?) — using last as best")
        _save_ckpt(model, args, feature_cols, area_id, ckpt_best)
        best_epoch, best_metrics, best_sep = args.epochs, metrics, sep
        for src_name, dst_name in [
            ("anomaly_scores_current.csv", "anomaly_scores_best.csv"),
            ("metrics_current.json",       "metrics_best.json"),
        ]:
            src = os.path.join(out_dir, src_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out_dir, dst_name))

    _save_ckpt(model, args, feature_cols, area_id, ckpt_last)
    last_scores  = _infer_scores(model, X_np, XY_np,
                                  args.radius_km, max_ctx, use_dc, device)
    last_metrics = _evaluate(last_scores, df_all, area_cfg, args, out_dir, "last")

    print(f"\n  {'='*50}", flush=True)
    sep_str = "n/a" if (best_sep is None or np.isnan(best_sep)) else f"{best_sep:+.4f}"
    _log(f"  [best]  epoch={best_epoch}  sep={sep_str}  ({crit_name})"
         f"  AUC={best_metrics['AUC_mean']:.4f}"
         f"  AP={best_metrics['AP_mean']:.4f}"
         f"  SR@5%={best_metrics['SR_5pct']:.4f}"
         f"  DTD={best_metrics['DTD_km']:.2f}km")
    _log(f"  [last]  AUC={last_metrics['AUC_mean']:.4f}"
         f"  AP={last_metrics['AP_mean']:.4f}"
         f"  SR@5%={last_metrics['SR_5pct']:.4f}"
         f"  DTD={last_metrics['DTD_km']:.2f}km")
    _log(f"T1-CL  {area_id}  ({args.method_name})  DONE")


def main():
    args   = get_args()
    areas  = resolve_areas(args.areas)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _log(f"T1-CL start | device={device} | areas={len(areas)} | "
         f"epochs={args.epochs} | cw={args.contrastive_weight} | "
         f"radius_km={args.radius_km}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for area_cfg in areas:
        run_area(area_cfg, args, device)

    _log("T1-CL all areas complete.")


if __name__ == "__main__":
    main()
