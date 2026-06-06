#!/usr/bin/env python3
"""
pretrain_multi.py  –  Pretrain GeoTransformerMulti to predict ALL element values.
==================================================================================
Improvements over original single-element pretrain.py
------------------------------------------------------
  * Multi-output: single forward pass → [B, P] predictions.
  * elem_mask_prob   : zero entire element column across all K neighbours
                       → forces cross-element chemical inference (not KNN copy).
  * spatial_dropout  : randomly drop entire neighbour rows (← geo_dataset idea)
                       → forces robustness to sparse/uneven sampling.
  * Contrastive loss : two independent augmented views of the same context;
                       their latents must agree (← geo_dataset idea).
                       Prevents latent collapse and improves anomaly separability.
  * L1 / MSE loss    : L1 more robust to heavy-tailed geochemical distributions.
  * Cross-attention  : Encoder-Decoder mode for cleaner target↔context flow.
  * MLP PE           : learnable positional encoding that adapts to coord range.
"""

import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from geo_transformer import (
    half_detection_limit, CompositionalTransformer, EXCLUDE_COLS,
)
from geo_transformer_multi import GeoTransformerMulti

SHARED_ELEMENTS = [
    "As", "Sb", "Ag", "W",  "Mo", "Bi",
    "Sn", "Nb", "Ta",
    "Au", "Pb", "Zn",
    "Co", "Cu", "Fe", "Mg", "Ca", "Ni",
]


# ─── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(
        description="GeoTransformerMulti: predict all element values per location")
    # Accept a single CSV (original) or multiple CSVs (new multi-deposit mode).
    # Provide exactly one of --csv_path or --csv_paths.
    csv_group = p.add_mutually_exclusive_group(required=True)
    csv_group.add_argument("--csv_path",  default=None,
                           help="Single geochemical CSV (original behaviour).")
    csv_group.add_argument("--csv_paths", default=None,
                           help="Comma-separated list of geochemical CSVs. "
                                "KNN neighbours are constrained within each area.")
    p.add_argument("--target_element",      default="all",
                   help="Single col, comma-separated list, or 'all'")
    p.add_argument("--shared_elements",     default=None,
                   help="Comma-separated base element names. Default: built-in 18.")
    p.add_argument("--k_neighbors",         type=int,   default=64)
    p.add_argument("--hidden_dim",          type=int,   default=256)
    p.add_argument("--n_heads",             type=int,   default=8)
    p.add_argument("--n_layers",            type=int,   default=6)
    p.add_argument("--epochs",              type=int,   default=500)
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
    # ── Augmentation ──────────────────────────────────────────────────────────
    p.add_argument("--neighbor_mask_prob",  type=float, default=0.0,
                   help="Per-cell random zero probability in neighbour features.")
    p.add_argument("--neighbor_noise_std",  type=float, default=0.0,
                   help="Gaussian noise std added to neighbour features.")
    p.add_argument("--elem_mask_prob",      type=float, default=0.3,
                   help="Prob that each predict element is zeroed across ALL "
                        "neighbour tokens (forces cross-element inference, "
                        "prevents KNN degeneracy).")
    p.add_argument("--spatial_dropout_prob", type=float, default=0.1,
                   help="Prob that each neighbour row is zeroed entirely "
                        "(simulates missing/sparse measurements). "
                        "Borrowed from geo_dataset contrastive approach.")
    # ── Contrastive loss ──────────────────────────────────────────────────────
    p.add_argument("--contrastive_weight",  type=float, default=0.05,
                   help="Weight for contrastive loss between two augmented views "
                        "of the same context. 0 = disabled. "
                        "Borrowed from geo_dataset/new_geo_transformer.py.")
    # ── Loss function ─────────────────────────────────────────────────────────
    p.add_argument("--loss_fn",             default="mse",
                   choices=["mse", "l1"],
                   help="Reconstruction loss: mse (default) or l1. "
                        "L1 is more robust to heavy-tailed geochemical data.")
    # ── Architecture options ──────────────────────────────────────────────────
    p.add_argument("--use_cross_attn",      action="store_true",
                   help="Use Encoder-Decoder (cross-attention) architecture. "
                        "Target cross-attends to neighbour context memory. "
                        "Borrowed from geo_dataset/new_geo_transformer.py.")
    p.add_argument("--use_mlp_pe",          action="store_true",
                   help="Use learnable MLP positional encoding instead of "
                        "sinusoidal 2D PE. Adapts to actual coordinate range. "
                        "Borrowed from geo_dataset/new_geo_transformer.py.")
    # ── Misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--out_dir",             default="outputs/pretrain_multi")
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--num_workers",         type=int,   default=4)
    p.add_argument("--resume",              action="store_true",
                   help="Resume from existing checkpoint if available.")
    return p.parse_args()


# ─── Dataset ──────────────────────────────────────────────────────────────────

class GeoDatasetMulti(Dataset):
    """
    Returns TWO independently augmented views of each sample's neighbour context.
    Used for both standard pretraining (ignore view2) and contrastive training.

    coords_query      : [N_query, 2]
    coords_ref        : [N_ref,   2]
    X_ref             : [N_ref,   C]
    neighbor_idx      : [N_query, K]
    y_all             : [N_query, P]
    predict_feat_idx  : feature column indices for predict_cols (for elem masking)
    spatial_dropout_p : prob to zero an entire neighbour row
    mask_prob         : per-cell feature zero probability
    noise_std         : Gaussian noise std
    elem_mask_prob    : per-column masking probability (across all K rows)
    two_views         : if False, view2 = view1 (no extra computation)
    """

    def __init__(self, coords_query: np.ndarray,
                 coords_ref: np.ndarray, X_ref: np.ndarray,
                 neighbor_idx: np.ndarray, y_all: np.ndarray,
                 predict_feat_idx: list = None,
                 spatial_dropout_p: float = 0.0,
                 mask_prob: float = 0.0,
                 noise_std: float = 0.0,
                 elem_mask_prob: float = 0.0,
                 two_views: bool = False):
        self.coords_query     = torch.tensor(coords_query, dtype=torch.float32)
        self.coords_ref       = torch.tensor(coords_ref,   dtype=torch.float32)
        self.X_ref            = torch.tensor(X_ref,        dtype=torch.float32)
        self.neighbor_idx     = torch.tensor(neighbor_idx, dtype=torch.long)
        self.y_all            = torch.tensor(y_all,        dtype=torch.float32)
        self.predict_feat_idx = predict_feat_idx or []
        self.spatial_dropout_p = spatial_dropout_p
        self.mask_prob        = mask_prob
        self.noise_std        = noise_std
        self.elem_mask_prob   = elem_mask_prob
        self.two_views        = two_views

    def __len__(self):
        return len(self.coords_query)

    def _augment(self, nbr_feat: torch.Tensor, dxy: torch.Tensor):
        """
        Apply all augmentations to ONE view independently.

        Augmentation pipeline (all stochastic, applied independently per call):
          1. Spatial dropout  : zero entire neighbour rows (row in [K] dimension)
          2. Per-cell masking : randomly zero individual feature cells
          3. Gaussian noise   : add noise to features
          4. Element masking  : zero an entire element column across ALL K rows

        Args
        ----
        nbr_feat : [K, C]  raw neighbour features (will be cloned)
        dxy      : [K, 2]  relative coordinates (also cloned; zeroed with row)

        Returns token tensor [K, 2+C].
        """
        nbr_feat = nbr_feat.clone()
        dxy      = dxy.clone()
        K        = nbr_feat.shape[0]

        # 1. Spatial dropout: zero entire neighbour row (position + features)
        #    Simulates missing/sparse measurements at nearby locations.
        #    Borrowed from geo_dataset/new_geo_transformer.py _mask_view().
        if self.spatial_dropout_p > 0 and K > 1:
            drop = torch.rand(K) < self.spatial_dropout_p
            if drop.all():
                drop[torch.randint(K, (1,))] = False  # keep at least one
            nbr_feat[drop] = 0.0
            dxy[drop]      = 0.0

        # 2. Per-cell random masking (element noise)
        if self.mask_prob > 0:
            cell_mask = torch.rand_like(nbr_feat) < self.mask_prob
            nbr_feat[cell_mask] = 0.0

        # 3. Gaussian noise
        if self.noise_std > 0:
            nbr_feat = nbr_feat + torch.randn_like(nbr_feat) * self.noise_std

        # 4. Per-element column masking: zero element j across ALL K neighbours.
        #    Forces the model to infer element j from other elements (cross-element).
        #    Prevents KNN-style copying of the target element from neighbours.
        if self.elem_mask_prob > 0:
            for fi in self.predict_feat_idx:
                if torch.rand(1).item() < self.elem_mask_prob:
                    nbr_feat[:, fi] = 0.0

        return torch.cat([dxy, nbr_feat], dim=-1)   # [K, 2+C]

    def __getitem__(self, idx):
        coord    = self.coords_query[idx]                  # [2]
        nbr_idx  = self.neighbor_idx[idx]                  # [K]
        nbr_xy   = self.coords_ref[nbr_idx]                # [K, 2]
        nbr_feat = self.X_ref[nbr_idx]                     # [K, C]  (read-only)
        dxy      = nbr_xy - coord.unsqueeze(0)             # [K, 2]

        view1 = self._augment(nbr_feat, dxy)               # [K, 2+C]
        view2 = self._augment(nbr_feat, dxy) if self.two_views else view1

        return view1, view2, coord, self.y_all[idx]        # [K,2+C], [K,2+C], [2], [P]


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
    tree    = KDTree(coords_ref.astype(np.float32))
    k_fetch = k + 1 if exclude_self else k
    k_fetch = min(k_fetch, len(coords_ref))
    _, ind  = tree.query(coords_query.astype(np.float32), k=k_fetch)
    if exclude_self:
        ind = ind[:, 1:]
    ind = ind[:, :k]
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


def _masked_loss(pred: torch.Tensor, target: torch.Tensor,
                 loss_fn: str = "mse") -> torch.Tensor:
    """
    Reconstruction loss over [B, P] tensors, ignoring NaN targets.
    loss_fn: 'mse' (default) or 'l1' (more robust to heavy-tailed distributions).
    """
    valid = torch.isfinite(target)
    if not valid.any():
        return pred.sum() * 0.0
    diff = pred[valid] - target[valid]
    return diff.abs().mean() if loss_fn == "l1" else (diff ** 2).mean()


# ─── Multi-area data loader ────────────────────────────────────────────────────

def _load_all_areas(csv_list: list, args, shared_names: list):
    """
    Load one or more area CSVs, apply per-area spatial train/val splits,
    fit a single CompositionalTransformer on pooled training data, and build
    area-aware KNN indices so neighbours never cross area boundaries.

    Returns a flat tuple consumed directly by main().
    """
    # ── Load each CSV ─────────────────────────────────────────────────────────
    area_dfs = []
    for csv_path in csv_list:
        print(f"[pretrain_multi] Loading {csv_path}")
        df = pd.read_csv(csv_path).dropna(subset=["X", "Y"])
        area_dfs.append(df)

    # ── Intersect numeric columns across all areas ────────────────────────────
    numeric_per_area = [
        [c for c in df.columns if c not in EXCLUDE_COLS and df[c].dtype != object]
        for df in area_dfs
    ]
    common_set = set(numeric_per_area[0])
    for nc in numeric_per_area[1:]:
        common_set &= set(nc)
    common_numeric = [c for c in numeric_per_area[0] if c in common_set]

    shared_cols = resolve_shared_cols(common_numeric, shared_names)
    print(f"[pretrain_multi] {len(shared_cols)} shared cols across {len(csv_list)} area(s): "
          f"{shared_cols}")

    # ── Resolve predict columns ───────────────────────────────────────────────
    target_str = args.target_element.strip()
    if target_str.lower() == "all":
        predict_cols = list(shared_cols)
    elif "," in target_str:
        predict_cols = [c.strip() for c in target_str.split(",")]
    else:
        predict_cols = [target_str]

    missing = [c for c in predict_cols if c not in common_numeric]
    if missing:
        raise ValueError(f"Target columns not found in all CSVs: {missing}")

    seen, feature_cols_raw = set(), []
    for c in shared_cols + predict_cols:
        if c in common_numeric and c not in seen:
            feature_cols_raw.append(c); seen.add(c)

    # ── Half-detection-limit imputation per area ──────────────────────────────
    area_dfs = [half_detection_limit(df, feature_cols_raw) for df in area_dfs]

    # ── Per-area spatial train/val split ──────────────────────────────────────
    area_train_dfs, area_val_dfs = [], []
    for df in area_dfs:
        tr_idx, va_idx = spatial_stratified_split(
            df, val_size=args.val_size, grid_n=args.val_grid_n, seed=args.seed)
        area_train_dfs.append(df.iloc[tr_idx].reset_index(drop=True))
        area_val_dfs.append(df.iloc[va_idx].reset_index(drop=True))
    for i, (p, tr, va) in enumerate(zip(csv_list, area_train_dfs, area_val_dfs)):
        print(f"[pretrain_multi]   area {i+1} ({os.path.basename(p)}): "
              f"train={len(tr)}  val={len(va)}")

    all_train_df = pd.concat(area_train_dfs, ignore_index=True)
    all_val_df   = pd.concat(area_val_dfs,   ignore_index=True)
    print(f"[pretrain_multi] pooled → train={len(all_train_df)}  val={len(all_val_df)}")

    # ── Fit transformer on POOLED training data ───────────────────────────────
    transformer = CompositionalTransformer(method=args.transform)
    X_train = transformer.fit_transform(
        all_train_df[feature_cols_raw].to_numpy(dtype=np.float32), feature_cols_raw)
    X_val   = transformer.transform(
        all_val_df[feature_cols_raw].to_numpy(dtype=np.float32))
    feature_cols = transformer.feature_cols_

    n_predict        = len(predict_cols)
    elem_to_id       = {col: i for i, col in enumerate(predict_cols)}
    elem_to_feat_idx = {col: feature_cols.index(col) for col in predict_cols}
    predict_feat_idx = [elem_to_feat_idx[c] for c in predict_cols]

    y_train_all = X_train[:, predict_feat_idx]
    y_val_all   = X_val[:,   predict_feat_idx]

    coords_train = all_train_df[["X", "Y"]].to_numpy(dtype=np.float32)
    coords_val   = all_val_df[["X", "Y"]].to_numpy(dtype=np.float32)

    # ── Per-area KNN with global index offsets ────────────────────────────────
    # Neighbours are constrained to the same area (no cross-area interpolation).
    print(f"[pretrain_multi] Precomputing per-area KNN (K={args.k_neighbors}) ...")
    K = args.k_neighbors
    train_nbr_idx = np.empty((len(coords_train), K), dtype=np.int32)
    val_nbr_idx   = np.empty((len(coords_val),   K), dtype=np.int32)

    train_sizes  = [len(tr) for tr in area_train_dfs]
    train_starts = [int(np.sum(train_sizes[:i])) for i in range(len(train_sizes))]
    val_sizes    = [len(va) for va in area_val_dfs]
    val_starts   = [int(np.sum(val_sizes[:i]))   for i in range(len(val_sizes))]

    for i, (tr_df, va_df) in enumerate(zip(area_train_dfs, area_val_dfs)):
        ts, te = train_starts[i], train_starts[i] + train_sizes[i]
        vs, ve = val_starts[i],   val_starts[i]   + val_sizes[i]

        ctr = coords_train[ts:te]
        cva = coords_val[vs:ve]

        # Train → train (self-supervised, exclude self)
        local_tr = precompute_neighbors(ctr, ctr, k=K, exclude_self=True)
        train_nbr_idx[ts:te] = local_tr + ts

        # Val → area's train (no self-exclusion needed)
        local_va = precompute_neighbors(cva, ctr, k=K, exclude_self=False)
        val_nbr_idx[vs:ve] = local_va + ts

    print("[pretrain_multi] Done precomputing.")

    in_dim = X_train.shape[1] + 2
    print(f"[pretrain_multi] {len(all_train_df)} train samples | "
          f"{len(feature_cols_raw)} feature cols | "
          f"{n_predict} predict cols: {predict_cols}")

    return (coords_train, X_train, y_train_all, train_nbr_idx,
            coords_val,   y_val_all,             val_nbr_idx,
            feature_cols, feature_cols_raw, predict_cols, predict_feat_idx,
            elem_to_id, n_predict, transformer, in_dim)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Resolve CSV list (single or multi-area) ───────────────────────────────
    csv_list = (
        [p.strip() for p in args.csv_paths.split(",")]
        if args.csv_paths else [args.csv_path]
    )

    # ── Resolve shared element names ──────────────────────────────────────────
    shared_names = (
        [s.strip() for s in args.shared_elements.split(",")]
        if args.shared_elements else SHARED_ELEMENTS
    )

    # ── Load all areas (handles both single and multi-CSV) ────────────────────
    (coords_train, X_train, y_train_all, train_nbr_idx,
     coords_val,   y_val_all,            val_nbr_idx,
     feature_cols, feature_cols_raw, predict_cols, predict_feat_idx,
     elem_to_id, n_predict, transformer, in_dim) = _load_all_areas(
         csv_list, args, shared_names)

    in_dim = X_train.shape[1] + 2   # (dx, dy) + C features
    use_contra = args.contrastive_weight > 0

    print(f"[pretrain_multi] device={DEVICE}  in_dim={in_dim}  n_predict={n_predict}")
    print(f"[pretrain_multi] loss_fn={args.loss_fn}  "
          f"use_cross_attn={args.use_cross_attn}  use_mlp_pe={args.use_mlp_pe}")
    print(f"[pretrain_multi] elem_mask_prob={args.elem_mask_prob}  "
          f"spatial_dropout_prob={args.spatial_dropout_prob}  "
          f"contrastive_weight={args.contrastive_weight}  "
          f"neighbor_mask_prob={args.neighbor_mask_prob}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    train_dataset = GeoDatasetMulti(
        coords_train, coords_train, X_train, train_nbr_idx, y_train_all,
        predict_feat_idx  = predict_feat_idx,
        spatial_dropout_p = args.spatial_dropout_prob,
        mask_prob         = args.neighbor_mask_prob,
        noise_std         = args.neighbor_noise_std,
        elem_mask_prob    = args.elem_mask_prob,
        two_views         = use_contra)            # generate 2nd view only if needed
    val_dataset = GeoDatasetMulti(
        coords_val, coords_train, X_train, val_nbr_idx, y_val_all)
    # val: no augmentation, two_views=False

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = GeoTransformerMulti(
        in_dim         = in_dim,
        n_predict      = n_predict,
        hidden_dim     = args.hidden_dim,
        n_heads        = args.n_heads,
        n_layers       = args.n_layers,
        dropout        = args.dropout,
        use_cross_attn = args.use_cross_attn,
        use_mlp_pe     = args.use_mlp_pe,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pretrain_multi] model params = {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)

    # Checkpoint name encodes key hyper-parameters
    arch_tag = ("xattn" if args.use_cross_attn else "enc") + \
               ("_mlppe" if args.use_mlp_pe else "")
    ckpt_name = (
        f"geo_transformer_multi"
        f"_K{args.k_neighbors}"
        f"_H{args.hidden_dim}"
        f"_L{args.n_layers}"
        f"_h{args.n_heads}"
        f"_B{args.batch_size}"
        f"_{args.transform}"
        f"_{arch_tag}"
        f".pt"
    )
    ckpt_path = os.path.join(args.out_dir, ckpt_name)
    print(f"[pretrain_multi] Checkpoint: {ckpt_name}")

    best_val_loss = float("inf")
    start_epoch = 0

    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt.get("epoch", -1) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"[pretrain_multi] Resumed from epoch {start_epoch}  best_val={best_val_loss:.6f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_recon, total_contra, n_steps = 0.0, 0.0, 0

        for view1_b, view2_b, coords_b, y_all_b in tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False):

            tok1  = view1_b.to(DEVICE)   # [B, K, 2+C]
            crd_t = coords_b.to(DEVICE)  # [B, 2]
            y_t   = y_all_b.to(DEVICE)   # [B, P]

            pred_all, h1 = model(crd_t, tok1)
            loss_recon   = _masked_loss(pred_all, y_t, args.loss_fn)

            if use_contra:
                tok2 = view2_b.to(DEVICE)
                _, h2      = model(crd_t, tok2)
                # Latents from two different views of the same context should agree.
                # Uses MSE (not L1) for the contrastive term regardless of loss_fn.
                loss_contra = F.mse_loss(h1, h2)
                loss        = loss_recon + args.contrastive_weight * loss_contra
                total_contra += loss_contra.item() * y_t.shape[0]
            else:
                loss = loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n_valid = torch.isfinite(y_t).sum().item()
            total_recon += loss_recon.item() * n_valid
            n_steps     += n_valid

        scheduler.step()
        train_recon = total_recon / max(n_steps, 1)
        train_contra = total_contra / max(len(train_dataset), 1)
        current_lr  = scheduler.get_last_lr()[0]

        # ── Fast val ─────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for v1, _, coords_b, y_all_b in val_loader:
                pred_all, _ = model(coords_b.to(DEVICE), v1.to(DEVICE))
                y_t = y_all_b.to(DEVICE)
                valid = torch.isfinite(y_t)
                if valid.any():
                    val_losses.extend(
                        ((pred_all[valid] - y_t[valid]) ** 2).cpu().numpy().tolist())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        log = (f"Epoch {epoch+1:3d}/{args.epochs} | lr={current_lr:.2e} | "
               f"train_{args.loss_fn}={train_recon:.5f} | val_mse={avg_val_loss:.5f}")
        if use_contra:
            log += f" | contra={train_contra:.5f}"
        print(log)

        # ── Full per-element val breakdown ────────────────────────────────────
        is_last      = (epoch + 1 == args.epochs)
        do_full_eval = ((epoch + 1) % args.eval_every == 0) or is_last
        if do_full_eval:
            all_preds = []
            model.eval()
            with torch.no_grad():
                for v1, _, coords_b, _ in val_loader:
                    pred_all, _ = model(coords_b.to(DEVICE), v1.to(DEVICE))
                    all_preds.append(pred_all.cpu().numpy())
            all_preds = np.concatenate(all_preds, axis=0)  # [N_val, P]

            print(f"  ── Per-element val (epoch {epoch+1}) {'─'*38}")
            print(f"  {'Element':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'r':>8}")
            rmses, maes, r2s, rs = [], [], [], []
            for ei, col in enumerate(predict_cols):
                rmse, mae, r2, r = _interp_metrics(
                    y_val_all[:, ei].astype(np.float64),
                    all_preds[:, ei].astype(np.float64))
                print(f"  {col:<22}{_fmt(rmse)}{_fmt(mae)}{_fmt(r2)}{_fmt(r)}")
                for lst, v in [(rmses, rmse), (maes, mae), (r2s, r2), (rs, r)]:
                    if np.isfinite(v): lst.append(v)
            avg = (np.mean(rmses) if rmses else float("nan"),
                   np.mean(maes)  if maes  else float("nan"),
                   np.mean(r2s)   if r2s   else float("nan"),
                   np.mean(rs)    if rs    else float("nan"))
            print(f"  {'[average]':<22}{_fmt(avg[0])}{_fmt(avg[1])}{_fmt(avg[2])}{_fmt(avg[3])}")
            print(f"  {'─'*54}")

        # ── Checkpoint ───────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch":                epoch,
                "best_val_loss":        best_val_loss,
                "feature_cols":         feature_cols,
                "feature_cols_raw":     feature_cols_raw,
                "predict_cols":         predict_cols,
                "elem_to_id":           elem_to_id,
                "raw_cols":             feature_cols_raw,
                "scaler":               transformer,
                "in_dim":               in_dim,
                "n_predict":            n_predict,
                "csv_paths":            csv_list,
                "args":                 vars(args),
            }, ckpt_path)
            print(f"  [ckpt] saved  → {ckpt_path}")

    print(f"\n[pretrain_multi] Done.  Best val MSE = {best_val_loss:.6f}")
    print(f"[pretrain_multi] Checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
