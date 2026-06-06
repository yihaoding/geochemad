# --- Model ---
import os, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.neighbors import KDTree

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTRASTIVE_WEIGHT = 0.01
RANDOM_SEED = 38
rng = np.random.default_rng(RANDOM_SEED)

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
    def __init__(self, feat_dim, d_model=128, nhead=2, num_enc_layers=4, num_dec_layers=4, dim_feedforward=128):
        super().__init__()
        self.in_proj = nn.Linear(feat_dim, d_model)
        self.pos_mlp = PosEncMLP(2, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_enc_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_dec_layers)
        self.out_proj_all = nn.Linear(d_model, feat_dim)
        self.out_proj_target = nn.Linear(d_model, d_model)

    def forward(self, target_feat, target_xy, ctx_feats, ctx_xy):
        rel = ctx_xy - target_xy
        pos = self.pos_mlp(rel)
        ctx = self.in_proj(ctx_feats) + pos
        tgt = self.in_proj(target_feat)
        mem = self.encoder(ctx)
        dec = self.decoder(tgt, mem)
        return self.out_proj_all(dec), self.out_proj_target(dec), dec

class GeoDataset(Dataset):
    def __init__(self, feats_ilr, coords_xy, tfeats=None, max_mask_ratio=0.5,
                 radius_km=50.0, max_ctx=1392, distance_constraint=True, use_haversine=True):
        self.X = feats_ilr.astype(np.float32)
        self.XY = coords_xy.astype(np.float32)
        self.N = self.X.shape[0]
        self.F = self.X.shape[1]
        self.max_mask_ratio = max_mask_ratio
        self.radius_km = float(radius_km)
        self.max_ctx = int(max_ctx)
        self.distance_constraint = bool(distance_constraint)
        self.use_haversine = bool(use_haversine)

        # --- Add extracted hidden features ---
        if tfeats is not None:
            assert len(tfeats) == self.N, "tfeats must have the same length as feats_ilr"
            self.TF = tfeats.astype(np.float32)
            self.hidden_dim = self.TF.shape[1]
        else:
            self.TF = None
            self.hidden_dim = 0

        if self.use_haversine:
            self._rad = np.deg2rad(self.XY.astype(np.float64))
        else:
            self._rad = None

    def __len__(self):
        return self.N

    def _haversine_m(self, latlon_a, latlon_b):
        """Haversine distance between two points given in radians. Returns metres."""
        R = 6371000.0
        dlat = latlon_b[0] - latlon_a[0]
        dlon = latlon_b[1] - latlon_a[1]
        s = np.sin(dlat/2.0)**2 + np.cos(latlon_a[0]) * np.cos(latlon_b[0]) * np.sin(dlon/2.0)**2
        return 2.0 * R * np.arcsin(np.sqrt(s))

    def _neighbor_indices(self, idx):
        """Return indices of points within radius_km of idx (excluding idx), capped at max_ctx."""
        r_m = self.radius_km * 1000.0
        if self.distance_constraint:
            if self.use_haversine:
                tgt  = self._rad[idx]
                dlat = self._rad[:, 0] - tgt[0]
                dlon = self._rad[:, 1] - tgt[1]
                s    = (np.sin(dlat/2.0)**2
                        + np.cos(tgt[0]) * np.cos(self._rad[:, 0]) * np.sin(dlon/2.0)**2)
                dists = 2.0 * 6371000.0 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))
            else:
                dists = np.linalg.norm(self.XY - self.XY[idx], axis=1)
            mask      = dists <= r_m
            mask[idx] = False
            idxs      = np.where(mask)[0]
            if idxs.size == 0:
                return idxs
            order = np.argsort(dists[idxs])
            return idxs[order][:self.max_ctx]
        else:
            # no distance constraint: all points except idx, capped at max_ctx
            idxs = np.delete(np.arange(self.N), idx)
            return idxs[:self.max_ctx]

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
        x_i = self.X[idx]
        xy_i = self.XY[idx]
        tfeat_i = self.TF[idx] if self.TF is not None else np.zeros((self.hidden_dim,), dtype=np.float32)

        neigh_idx = self._neighbor_indices(idx)
        ctxX = self.X[neigh_idx] if len(neigh_idx) > 0 else np.zeros((0, self.F), dtype=np.float32)
        ctxXY = self.XY[neigh_idx] if len(neigh_idx) > 0 else np.zeros((0, 2), dtype=np.float32)

        Xr, Yr = self._mask_view(ctxX, ctxXY)
        X1, Y1 = self._mask_view(ctxX, ctxXY)
        X2, Y2 = self._mask_view(ctxX, ctxXY)

        return {
            "target": x_i,
            "txy": xy_i,
            "tfeats": tfeat_i,   # <--- added here
            "Xr": Xr, "Yr": Yr,
            "X1": X1, "Y1": Y1,
            "X2": X2, "Y2": Y2,
            "F": self.F,
            "M": self.max_ctx
        }


def collate(batch):
    B = len(batch)
    F = batch[0]["F"]
    M = batch[0]["M"]

    def stack_scalar(key):
        arr = np.stack([b[key] for b in batch], axis=0)
        return torch.tensor(arr[:, None, :], dtype=torch.float32)

    def stack_hidden():
        arr = np.stack([b["tfeats"] for b in batch], axis=0)  # (B, hidden_dim)
        return torch.tensor(arr[:, None, :], dtype=torch.float32)  # (B,1,H)

    def pad_stack(pair_key_X, pair_key_Y, feat_dim):
        X_out = np.zeros((B, M, feat_dim), dtype=np.float32)
        Y_out = np.zeros((B, M, 2), dtype=np.float32)
        attn = np.zeros((B, M), dtype=np.float32)
        for i, b in enumerate(batch):
            m = min(b[pair_key_X].shape[0], M)
            if m > 0:
                X_out[i, :m, :] = b[pair_key_X][:m]
                Y_out[i, :m, :] = b[pair_key_Y][:m]
                attn[i, :m] = 1.0
        return (
            torch.tensor(X_out, dtype=torch.float32),
            torch.tensor(Y_out, dtype=torch.float32),
            torch.tensor(attn, dtype=torch.float32)
        )

    tgt = stack_scalar("target")
    txy = stack_scalar("txy")
    tfeats = stack_hidden()  # <--- added

    Xr, Yr, attn_r = pad_stack("Xr", "Yr", F)
    X1, Y1, attn_1 = pad_stack("X1", "Y1", F)
    X2, Y2, attn_2 = pad_stack("X2", "Y2", F)

    return {
        "t": tgt, "txy": txy, "tfeats": tfeats,
        "Xr": Xr, "Yr": Yr, "attn_r": attn_r,
        "X1": X1, "Y1": Y1, "attn_1": attn_1,
        "X2": X2, "Y2": Y2, "attn_2": attn_2,
    }


# --- Train & Infer ---
from tqdm import tqdm

def train_one_epoch(model, loader, opt, epoch=None, total_epochs=None,
                    distill_weight=1.0, contrastive_weight=None):
    """
    Train the model for one epoch with tqdm progress bar.

    distill_weight: multiplier on L1(recon_target, tfeats). Default 1.0 keeps
        historical behaviour; lower (e.g. 0.3) relaxes the distillation pull.
    contrastive_weight: override for module-level CONTRASTIVE_WEIGHT (0.01).
    """
    model.train()
    R_all, R_target, C = 0.0, 0.0, 0.0
    cw = CONTRASTIVE_WEIGHT if contrastive_weight is None else float(contrastive_weight)
    dw = float(distill_weight)

    desc = f"Epoch [{epoch}/{total_epochs}]" if epoch is not None else "Training"
    pbar = tqdm(loader, desc=desc, unit="batch")

    for b in pbar:
        t,  txy, tfeats = b["t"].to(DEVICE),   b["txy"].to(DEVICE), b["tfeats"].to(DEVICE)
        Xr, Yr  = b["Xr"].to(DEVICE),  b["Yr"].to(DEVICE)
        X1, Y1  = b["X1"].to(DEVICE),  b["Y1"].to(DEVICE)
        X2, Y2  = b["X2"].to(DEVICE),  b["Y2"].to(DEVICE)

        opt.zero_grad()


        recon_all, recon_target, _ = model(t, txy, Xr, Yr)
        l_rec_all = F.l1_loss(recon_all, t)
        # target reconstruction loss only when hidden features are available
        has_tfeats = tfeats.shape[-1] > 0
        l_rec_target = F.l1_loss(recon_target, tfeats) if has_tfeats else recon_target.new_zeros(1).squeeze()

        r1, *_ = model(t, txy, X1, Y1)
        r2, *_ = model(t, txy, X2, Y2)
        l_con = F.mse_loss(r1, r2)

        loss = l_rec_all + dw * l_rec_target + cw * l_con

        loss.backward()
        opt.step()

        R_all    += l_rec_all.item()
        R_target += l_rec_target.item()
        C        += l_con.item()

        pbar.set_postfix({
            "L1": f"{l_rec_all.item():.4f}",
            "Con": f"{l_con.item():.4f}",
            "Total": f"{loss.item():.4f}"
        })

    n = len(loader)
    return R_all / n, C / n, R_target / n


def infer_scores(model, X_ilr, XY, device=DEVICE, ctx_knn=None):
    """
    Returns:
      scores: (N,)  Euclidean reconstruction error per sample
      recon_all: (N, F) reconstructed features for every sample

    ctx_knn: if set, attend to only the k nearest neighbours per sample
        (matches T1's training-time KNN context). This makes per-sample
        forward O(k²) instead of O(N²), turning total inference O(N*k²)
        instead of O(N³). For large N this is the difference between
        seconds and hours.
    """
    model.eval()

    # ---- standardize to numpy ----
    X_np  = X_ilr.to_numpy() if hasattr(X_ilr, "to_numpy") else np.asarray(X_ilr)
    XY_np = XY.to_numpy()     if hasattr(XY, "to_numpy")    else np.asarray(XY)

    N, F = X_np.shape
    scores    = np.zeros(N, dtype=np.float32)
    recon_all = np.zeros((N, 128), dtype=np.float32)

    # ---- precompute KNN index once if requested ----
    use_knn = ctx_knn is not None and 0 < ctx_knn < N - 1
    if use_knn:
        from sklearn.neighbors import NearestNeighbors
        # use raw spatial coords (XY) for KNN, matching training-time context
        nbrs = NearestNeighbors(n_neighbors=int(ctx_knn) + 1).fit(XY_np)
        _, knn_idx = nbrs.kneighbors(XY_np)
        knn_idx = knn_idx[:, 1:]   # drop self column

    # ---- move full tensors once to device ----
    X_full  = torch.as_tensor(X_np,  dtype=torch.float32, device=device)
    XY_full = torch.as_tensor(XY_np, dtype=torch.float32, device=device)
    if use_knn:
        knn_idx_t = torch.as_tensor(knn_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        for i in range(N):
            # target (single)
            t   = X_full[i:i+1].unsqueeze(0)      # [1, 1, F]
            txy = XY_full[i:i+1].unsqueeze(0)     # [1, 1, C]

            if use_knn:
                # k nearest neighbours (self already excluded)
                ctx_ids = knn_idx_t[i]
                Xc = X_full[ctx_ids].unsqueeze(0)     # [1, k, F]
                Yc = XY_full[ctx_ids].unsqueeze(0)    # [1, k, C]
            else:
                # legacy full leave-one-out context
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[i] = False
                Xc = X_full[mask].unsqueeze(0)        # [1, N-1, F]
                Yc = XY_full[mask].unsqueeze(0)       # [1, N-1, C]

            # forward
            rec,_,dec = model(t, txy, Xc, Yc)
            rec = rec.squeeze().reshape(-1)       # -> [F]

            # store
            r = rec.detach().cpu().numpy().astype(np.float32)
            d = dec.detach().cpu().numpy().astype(np.float32)
            recon_all[i] = d

            # score (L2)
            diff = X_np[i] - r
            scores[i] = float(np.sqrt(np.dot(diff, diff)))

    return scores, recon_all

def _standardize_xy_cols(df):
    for a, b in [("x","y"), ("X","Y"), ("LONGITUDE","LATITUDE"),
                 ("longitude","latitude"), ("lon","lat"),
                 ("EASTING","NORTHING"), ("easting","northing")]:
        if a in df.columns and b in df.columns:
            return df.rename(columns={a:"x", b:"y"})[["x","y"]].copy()
    raise AssertionError("deposits.csv must contain x/y or equivalent columns.")

def _pairwise_dist_km(A, B, latlon=True):
    A = np.asarray(A, float); B = np.asarray(B, float)
    if latlon:
        # Haversine (vectorized)
        Alon = np.radians(A[:,0])[:,None]; Alat = np.radians(A[:,1])[:,None]
        Blon = np.radians(B[:,0])[None,:]; Blat = np.radians(B[:,1])[None,:]
        dlon = Blon - Alon; dlat = Blat - Alat
        a = np.sin(dlat/2.0)**2 + np.cos(Alat)*np.cos(Blat)*np.sin(dlon/2.0)**2
        c = 2.0*np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
        return 6371.0 * c  # km
    else:
        diff = A[:,None,:] - B[None,:,:]
        return np.sqrt((diff**2).sum(axis=2)) / 1000.0

def map_deposit_scores(dep_xy, samp_xy, samp_scores, method="nn", radius_km=2.0):
    if method == "nn":
        tree = KDTree(samp_xy)
        _, idx = tree.query(dep_xy, k=1)
        return samp_scores[idx.ravel()]
    D = _pairwise_dist_km(samp_xy, dep_xy)
    within = D <= radius_km
    scores = []
    for j in range(dep_xy.shape[0]):
        idxs = np.where(within[:, j])[0]
        if len(idxs) == 0:
            tree = KDTree(samp_xy)
            _, idx = tree.query(dep_xy[j][None, :], k=1)
            scores.append(samp_scores[idx.ravel()[0]])
        else:
            vals = samp_scores[idxs]
            scores.append(np.max(vals) if method == "radius_max" else np.median(vals))
    return np.array(scores)

# def sample_negatives(samp_xy, dep_xy, samp_scores, k, min_km=5.0):
#     D = _pairwise_dist_km(samp_xy, dep_xy, latlon = True)
#     min_d = D.min(axis=1)
#     eligible = np.where(min_d >= min_km)[0]
#     if len(eligible) == 0:
#         print(f"⚠️ No eligible negatives >= {min_km}km; using random fallback.")
#         eligible = np.arange(len(samp_scores))
#     idx = rng.choice(eligible, size=k, replace=True)
#     return samp_scores[idx]

import numpy as np

def sample_negatives(
    samp_xy, 
    dep_xy, 
    samp_scores, 
    k, 
    min_km=5.0, 
    allow_replacement=True, 
    random_state=None, 
    weighted=False
):
    """
    Sample negative sites at least `min_km` away from all deposits.

    Args:
        samp_xy (ndarray): (N,2) sample coordinates.
        dep_xy (ndarray): (M,2) deposit coordinates.
        samp_scores (ndarray): (N,) anomaly scores for samples.
        k (int): number of negatives to sample.
        min_km (float): minimum distance to nearest deposit.
        allow_replacement (bool): allow repeated sampling.
        random_state (int | None): random seed for reproducibility.
        weighted (bool): if True, weight sampling by distance (farther = higher chance).

    Returns:
        np.ndarray: selected negative sample scores (length k).
    """
    rng = np.random.default_rng(random_state)

    # Compute pairwise distances (N samples × M deposits)
    D = _pairwise_dist_km(samp_xy, dep_xy)
    min_d = D.min(axis=1)

    # Select eligible negatives beyond distance threshold
    eligible = np.where(min_d >= min_km)[0]
    if len(eligible) == 0:
        print(f"⚠️ No eligible negatives >= {min_km:.1f} km; using random fallback.")
        eligible = np.arange(len(samp_scores))

    # Optionally weight sampling probability by distance
    if weighted:
        weights = min_d[eligible]
        weights = np.clip(weights, 1e-3, None)
        weights = weights / weights.sum()
        idx = rng.choice(eligible, size=k, replace=allow_replacement, p=weights)
    else:
        idx = rng.choice(eligible, size=k, replace=allow_replacement)

    return samp_scores[idx]


def evaluate_and_visualize(model, X_ilr, coords, work_df, epoch, CHECKPOINT_PATH, DEPOSITS_CSV,DEPOSIT_MAPPING_METHOD,DEPOSIT_RADIUS_KM,NEG_MIN_DIST_KM):
    """Run evaluation + visualization for the current best model."""
    model.eval()
    scores, recon = infer_scores(model, X_ilr, coords)
    out = work_df.copy()
    out["anomaly_score"] = scores

    # === Save scores ===
    # OUTPUT_SCORES = f"anomaly_scores_best_epoch{epoch}.csv"
    # out.to_csv(OUTPUT_SCORES, index=False)
    # print(f"✅ Epoch {epoch}: anomaly scores saved -> {OUTPUT_SCORES}")

    # === Load deposits ===
    dep_raw = pd.read_csv(DEPOSITS_CSV)
    dep_xy_df = _standardize_xy_cols(dep_raw).dropna()
    dep_xy = dep_xy_df.to_numpy()
    samp_xy = out[["x","y"]].to_numpy()
    samp_scores = out["anomaly_score"].to_numpy()

    pos_scores = map_deposit_scores(dep_xy, samp_xy, samp_scores, DEPOSIT_MAPPING_METHOD, DEPOSIT_RADIUS_KM)
    neg_scores = sample_negatives(samp_xy, dep_xy, samp_scores, len(pos_scores), NEG_MIN_DIST_KM)

    y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
    y_score = np.concatenate([pos_scores, neg_scores])

    auc_oneshot = roc_auc_score(y_true, y_score)
    ap_oneshot  = average_precision_score(y_true, y_score)
    print(f"  📈 Epoch {epoch}: AUC={auc_oneshot:.3f}, AP={ap_oneshot:.3f}")

    # === Visuals ===
    plt.figure(figsize=(6,5))
    sc = plt.scatter(out["x"], out["y"], s=8, c=out["anomaly_score"], cmap="coolwarm")
    plt.colorbar(sc, label="Anomaly Score")
    plt.title(f"Anomaly Score Map (Epoch {epoch})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.show()

    plt.figure()
    plt.hist(pos_scores, bins=20, alpha=0.6, label="Deposits")
    plt.hist(neg_scores, bins=20, alpha=0.6, label="Negatives")
    plt.legend(); plt.title(f"Score Distributions (Epoch {epoch})")
    plt.show()

    return auc_oneshot, ap_oneshot


# ── Spatial metric helpers (matching geo_trainer.py API) ─────────────────────

def _sr_at_pct(samp_scores, dep_nn_idx, top_pct=0.05):
    """Success rate: fraction of deposits whose NN sample is in top top_pct% by score."""
    n_top  = max(1, int(len(samp_scores) * top_pct))
    top_set = set(np.argsort(-samp_scores)[:n_top])
    n_dep   = len(dep_nn_idx)
    if n_dep == 0:
        return float('nan')
    return sum(1 for i in dep_nn_idx if i in top_set) / n_dep


def _recall_k(samp_scores, dep_nn_idx, k=50):
    """Fraction of deposits whose NN sample is in the top-k by score."""
    top_set = set(np.argsort(-samp_scores)[:k])
    n_dep   = len(dep_nn_idx)
    if n_dep == 0:
        return float('nan')
    return sum(1 for i in dep_nn_idx if i in top_set) / n_dep


def _dtd_km(samp_xy, samp_scores, dep_xy, top_pct=0.01):
    """Mean distance (km) from top top_pct% anomalous samples to nearest deposit."""
    n_top   = max(1, int(len(samp_scores) * top_pct))
    top_idx = np.argsort(-samp_scores)[:n_top]
    D = _pairwise_dist_km(samp_xy[top_idx], dep_xy)  # (n_top, n_dep)
    return float(D.min(axis=1).mean())


def _density_score(samp_xy, samp_scores, dep_xy, radius_km=2.0):
    """Mean anomaly score of samples within radius_km of each deposit."""
    vals = []
    for d in dep_xy:
        D      = _pairwise_dist_km(samp_xy, d[None, :]).ravel()
        within = np.where(D <= radius_km)[0]
        if len(within) > 0:
            vals.append(samp_scores[within].mean())
    return float(np.mean(vals)) if vals else float('nan')


def _print_metrics(results: dict, epoch):
    """Print metrics in the same tabular style as geo_trainer.print_metrics."""
    print(f"\n{'='*50}")
    print(f"  GeoTransformer  epoch={epoch}")
    print(f"{'='*50}")
    groups = {
        'Discrimination':      [('AUC_mean',    'AUC_std'),
                                ('AP_mean',     'AP_std'),
                                ('PR_AUC_mean', 'PR_AUC_std')],
        'Spatial Targeting':   [('SR_5pct',   None),
                                ('Recall_50', None)],
        'Spatial Consistency': [('DTD_km',  None),
                                ('Density', None)],
    }
    for grp, keys in groups.items():
        print(f"\n--- {grp} ---")
        for mean_k, std_k in keys:
            if mean_k not in results:
                continue
            v = results[mean_k]
            if std_k and std_k in results:
                print(f"  {mean_k:14s}: {v:.4f} ± {results[std_k]:.4f}")
            else:
                print(f"  {mean_k:14s}: {v:.4f}")
    print(f"{'='*50}\n")


def evaluate_and_visualize_multineg(
    model, X_ilr, coords, work_df,
    epoch, CHECKPOINT_PATH, DEPOSITS_CSV,
    DEPOSIT_MAPPING_METHOD="nearest",
    DEPOSIT_RADIUS_KM=2.0,
    NEG_MIN_DIST_KM=5.0,
    NEG_SAMPLE_RUNS=20,
    NEG_SAMPLES_PER_RUN=None,
    output_path=None,
    precomputed_scores=None,
    ctx_knn=None,
):
    """
    Evaluation with multiple negative sampling + full AE-equivalent spatial metrics.

    Metrics returned (matching geo_trainer.py naming where applicable):
      AUC_mean / AUC_std       – mean ROC-AUC over NEG_SAMPLE_RUNS negative draws
      AP_mean  / AP_std        – mean Average Precision
      PR_AUC_mean / PR_AUC_std – mean PR-AUC
      SR_5pct                  – success rate at top 5% of samples by score
      Recall_50                – fraction of deposits in top-50 samples
      DTD_km                   – mean distance (km) from top-1% samples to nearest deposit
      Density                  – mean anomaly score within DEPOSIT_RADIUS_KM of deposits
    """
    model.eval()
    if precomputed_scores is not None:
        scores = np.asarray(precomputed_scores, dtype=np.float32)
    else:
        scores, _ = infer_scores(model, X_ilr, coords, ctx_knn=ctx_knn)
    out = work_df.copy()
    out["anomaly_score"] = scores

    # === Load deposits ===
    dep_raw   = pd.read_csv(DEPOSITS_CSV)
    dep_xy_df = _standardize_xy_cols(dep_raw).dropna()
    dep_xy    = dep_xy_df.to_numpy()
    samp_xy   = out[["x", "y"]].to_numpy()
    samp_scores = out["anomaly_score"].to_numpy()

    # NN sample index for each deposit (used by spatial metrics)
    tree = KDTree(samp_xy)
    _, dep_nn_idx = tree.query(dep_xy, k=1)
    dep_nn_idx = dep_nn_idx.ravel()

    pos_scores = map_deposit_scores(dep_xy, samp_xy, samp_scores,
                                    DEPOSIT_MAPPING_METHOD, DEPOSIT_RADIUS_KM)
    k = len(pos_scores)

    # === Multiple negative sampling: AUC / AP / PR-AUC ===
    aucs, aps, pr_aucs = [], [], []
    neg_scores_all = []

    print(f"\n🔁 Multiple Negative Sampling Evaluation ({NEG_SAMPLE_RUNS} runs)")
    for _ in range(NEG_SAMPLE_RUNS):
        neg_scores = sample_negatives(
            samp_xy, dep_xy, samp_scores,
            k if NEG_SAMPLES_PER_RUN is None else NEG_SAMPLES_PER_RUN,
            NEG_MIN_DIST_KM, allow_replacement=True,
        )
        y_true  = np.array([1]*k + [0]*len(neg_scores))
        y_score = np.concatenate([pos_scores, neg_scores])

        aucs.append(roc_auc_score(y_true, y_score))
        aps.append(average_precision_score(y_true, y_score))
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        pr_aucs.append(float(np.trapz(prec[::-1], rec[::-1])))
        neg_scores_all.extend(neg_scores)

    aucs    = np.array(aucs)
    aps     = np.array(aps)
    pr_aucs = np.array(pr_aucs)

    # === Spatial metrics (single-pass, no negative sampling) ===
    sr_5pct   = _sr_at_pct(samp_scores, dep_nn_idx, top_pct=0.05)
    recall_50 = _recall_k(samp_scores, dep_nn_idx, k=50)
    dtd_km    = _dtd_km(samp_xy, samp_scores, dep_xy, top_pct=0.01)
    density   = _density_score(samp_xy, samp_scores, dep_xy, radius_km=DEPOSIT_RADIUS_KM)

    results = {
        "AUC_mean"   : float(aucs.mean()),    "AUC_std"    : float(aucs.std()),
        "AP_mean"    : float(aps.mean()),     "AP_std"     : float(aps.std()),
        "PR_AUC_mean": float(pr_aucs.mean()), "PR_AUC_std" : float(pr_aucs.std()),
        "SR_5pct"    : float(sr_5pct),
        "Recall_50"  : float(recall_50),
        "DTD_km"     : float(dtd_km),
        "Density"    : float(density),
        # raw arrays kept for callers
        "pos_scores"    : pos_scores,
        "neg_scores_all": np.array(neg_scores_all),
    }

    _print_metrics(results, epoch)

    # === Save anomaly scores ===
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out.to_csv(output_path, index=False)
        print(f"  💾 Anomaly scores saved → {output_path}")

    # === Visualization ===
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(out["x"], out["y"], s=8, c=samp_scores, cmap="coolwarm")
    plt.colorbar(sc, label="Anomaly Score")
    plt.title(f"Anomaly Score Map (Epoch {epoch})")
    plt.xlabel("x"); plt.ylabel("y")
    plt.tight_layout(); plt.show()

    plt.figure()
    plt.hist(pos_scores, bins=20, alpha=0.6, label="Deposits (pos)")
    plt.hist(neg_scores_all, bins=20, alpha=0.6, label=f"Negatives ({NEG_SAMPLE_RUNS}×)")
    plt.legend(); plt.title(f"Score Distributions (Epoch {epoch})")
    plt.tight_layout(); plt.show()

    # === Success-Rate curve ===
    rank_idx  = np.argsort(-samp_scores)
    area_frac = np.arange(1, len(rank_idx) + 1) / len(rank_idx)
    covered   = np.zeros(len(rank_idx))
    hits = 0
    for ksel, si in enumerate(rank_idx, start=1):
        hits += int(np.sum(dep_nn_idx == si))
        covered[ksel - 1] = hits

    plt.figure()
    plt.plot(area_frac, covered / max(1, len(dep_nn_idx)))
    plt.xlabel("Cumulative area fraction")
    plt.ylabel("Cumulative fraction of deposits covered")
    plt.title(f"Success-Rate Curve (Epoch {epoch})")
    plt.tight_layout(); plt.show()

    return results
