"""
geo_transformer.py
==================
Standalone GeoTransformer model + shared preprocessing utilities.
Used by pretrain.py, extract_latent.py, and train_anomaly.py.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

EXCLUDE_COLS = {"X", "Y", "SAMPLEID", "SAMPLETYPE", "WAMEX_A_NO", "COMPSAMPID"}


# ─── Compositional transforms ─────────────────────────────────────────────────

def _clr(X: np.ndarray) -> np.ndarray:
    """Centered Log-Ratio transform. X must be strictly positive. Returns [N, D]."""
    log_X = np.log(np.clip(X, 1e-10, None)).astype(np.float64)
    return (log_X - log_X.mean(axis=1, keepdims=True)).astype(np.float32)


def _ilr(X: np.ndarray) -> np.ndarray:
    """
    Isometric Log-Ratio transform (Helmert sequential binary partition).
    X must be strictly positive.  Maps [N, D] → [N, D-1].

    ilr_j = sqrt(j+1 / j+2) * (mean(log x_0..x_j) - log x_{j+1})
    """
    log_X = np.log(np.clip(X, 1e-10, None)).astype(np.float64)
    N, D  = log_X.shape
    out   = np.empty((N, D - 1), dtype=np.float64)
    for j in range(D - 1):
        n_left    = j + 1
        scale     = np.sqrt(n_left / (n_left + 1.0))
        out[:, j] = scale * (log_X[:, :n_left].mean(axis=1) - log_X[:, n_left])
    return out.astype(np.float32)


class CompositionalTransformer:
    """
    Unified preprocessing for geochemical element data.

    Methods
    -------
    'none' : raw (half-DL) → StandardScaler                 (D → D features)
    'clr'  : raw → CLR → StandardScaler                     (D → D features)
    'ilr'  : raw → ILR (Helmert SBP) → StandardScaler       (D → D-1 features)

    ``transform()`` always accepts the half-DL-applied element matrix [N, D]
    with columns matching those passed to ``fit_transform()``.

    For 'ilr': a separate log-StandardScaler is fitted on the target element
    via ``fit_transform_target()`` so the prediction target stays a scalar.
    """

    METHODS = ("none", "clr", "ilr")

    def __init__(self, method: str = "none"):
        if method not in self.METHODS:
            raise ValueError(f"transform must be one of {self.METHODS}, got {method!r}")
        self.method         = method
        self.scaler         = StandardScaler()
        self.raw_cols_      = None   # D input column names
        self.feature_cols_  = None   # output column names (D or D-1)
        self.target_scaler_ = None   # log-scaler for target (ilr mode only)

    def _core(self, X: np.ndarray) -> np.ndarray:
        if self.method == "clr":
            return _clr(X)
        if self.method == "ilr":
            return _ilr(X)
        return X.astype(np.float32)

    def fit_transform(self, X: np.ndarray, raw_cols: list) -> np.ndarray:
        """Fit on X and return scaled features. Sets raw_cols_ and feature_cols_."""
        self.raw_cols_ = list(raw_cols)
        X_t = self._core(X)
        self.feature_cols_ = (
            [f"ilr_{i}" for i in range(X_t.shape[1])]
            if self.method == "ilr" else list(raw_cols)
        )
        return self.scaler.fit_transform(X_t).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new raw data (same D columns as fit data)."""
        return self.scaler.transform(self._core(X)).astype(np.float32)

    def fit_transform_target(self, y_raw: np.ndarray) -> np.ndarray:
        """
        ILR mode: fit a log-StandardScaler on the raw target and return scaled
        log-targets.  Returns None for 'none'/'clr' (target is in feature_cols_).
        """
        if self.method != "ilr":
            return None
        valid = (y_raw > 0) & np.isfinite(y_raw)
        log_y = np.where(valid, np.log(y_raw.astype(np.float64)), np.nan)
        self.target_scaler_ = StandardScaler()
        self.target_scaler_.fit(log_y[valid].reshape(-1, 1))
        return np.where(
            valid,
            self.target_scaler_.transform(log_y.reshape(-1, 1)).flatten(),
            np.nan,
        ).astype(np.float32)

    def transform_target(self, y_raw: np.ndarray) -> np.ndarray:
        """Apply the fitted target-log-scaler to new data (ILR mode only)."""
        if self.method != "ilr" or self.target_scaler_ is None:
            return None
        valid = (y_raw > 0) & np.isfinite(y_raw)
        log_y = np.where(valid, np.log(y_raw.astype(np.float64)), np.nan)
        return np.where(
            valid,
            self.target_scaler_.transform(log_y.reshape(-1, 1)).flatten(),
            np.nan,
        ).astype(np.float32)


# ─── Positional Encoding ──────────────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for (longitude, latitude) coordinates."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 4 == 0, "hidden_dim must be divisible by 4"
        self.dim = dim

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords : [B, 2] or [B, N, 2]
        returns: [B, 1, dim] or [B, N, dim]
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(1)          # [B, 1, 2]
        x, y = coords[..., 0], coords[..., 1]    # each [B, N]
        B, N = x.shape

        div_term = torch.exp(
            torch.arange(0, self.dim // 2, 2, device=coords.device).float()
            * -(np.log(10000.0) / (self.dim // 2))
        )

        pe_x = torch.zeros(B, N, self.dim // 2, device=coords.device)
        pe_y = torch.zeros(B, N, self.dim // 2, device=coords.device)
        pe_x[..., 0::2] = torch.sin(x.unsqueeze(-1) * div_term)
        pe_x[..., 1::2] = torch.cos(x.unsqueeze(-1) * div_term)
        pe_y[..., 0::2] = torch.sin(y.unsqueeze(-1) * div_term)
        pe_y[..., 1::2] = torch.cos(y.unsqueeze(-1) * div_term)

        return torch.cat([pe_x, pe_y], dim=-1)   # [B, N, dim]


# ─── GeoTransformer ───────────────────────────────────────────────────────────

class GeoTransformer(nn.Module):
    """
    Sequence of tokens fed to a Transformer encoder:
        token0 : element embedding  (which element we are predicting)
        token1 : target coordinate  (query location, with 2D PE)
        token2…K+1 : K neighbor tokens  (dx,dy + all element values, with 2D PE)

    The hidden state of token1 after encoding is returned as the
    target-element-aware latent representation of the query point.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 2, num_elements: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.element_embed = nn.Embedding(num_elements, hidden_dim)
        self.input_proj    = nn.Linear(in_dim, hidden_dim)
        self.coord_proj    = nn.Linear(2, hidden_dim)
        self.pos_encoder   = PositionalEncoding2D(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden_dim * 4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.head_pred   = nn.Linear(hidden_dim, 1)

    def forward(self, element_idx: torch.Tensor,
                target_coord: torch.Tensor,
                neighbor_tokens: torch.Tensor):
        """
        element_idx    : [B]       long – element ID
        target_coord   : [B, 2]   float – (X, Y)
        neighbor_tokens: [B, K, 2+C] float – (dx, dy, feat_0 … feat_{C-1})

        Returns
        -------
        pred   : [B]    predicted target element value
        latent : [B, H] target-element-aware hidden state (token1 output)
        """
        # token0: element type
        elem_h  = self.element_embed(element_idx)                          # [B,H]
        token0  = elem_h.unsqueeze(1)                                      # [B,1,H]

        # token1: target location + element identity
        # Adding element embedding here gives the prediction head a direct
        # signal of which element to output, without relying solely on attention.
        pe_tgt  = self.pos_encoder(target_coord)                           # [B,1,H]
        token1  = (self.coord_proj(target_coord).unsqueeze(1)
                   + pe_tgt
                   + elem_h.unsqueeze(1))                                  # [B,1,H]

        # token2…: neighbors
        pe_nbr  = self.pos_encoder(neighbor_tokens[:, :, :2])            # [B,K,H]
        tokens_n = self.input_proj(neighbor_tokens) + pe_nbr             # [B,K,H]

        seq    = torch.cat([token0, token1, tokens_n], dim=1)            # [B,2+K,H]
        hidden = self.encoder(seq)                                        # [B,2+K,H]

        # Extract target-location token (index 1) as latent representation
        h_t    = torch.relu(self.head_hidden(hidden[:, 1, :]))           # [B,H]
        pred   = self.head_pred(h_t).squeeze(-1)                         # [B]
        return pred, h_t


# ─── Data utilities ───────────────────────────────────────────────────────────

def half_detection_limit(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Replace non-positive / NaN values with ½ × minimum positive per column."""
    df = df.copy()
    for c in cols:
        arr = df[c].to_numpy(dtype=float)
        bad = ~np.isfinite(arr) | (arr <= 0)
        if bad.any():
            min_pos = arr[arr > 0].min() if (arr > 0).any() else 1e-6
            arr[bad] = min_pos / 2
            df[c] = arr
    return df


def load_and_preprocess(csv_path: str, target_element: str,
                        test_frac: float = 0.1, seed: int = 42,
                        input_elements: list = None,
                        transform: str = "none"):
    """
    Load CSV, apply half-detection-limit, transform + standardise, split.

    Parameters
    ----------
    input_elements : restrict features to this subset (target always included).
    transform      : 'none' | 'clr' | 'ilr'
        'none' – StandardScaler on raw half-DL values (original behaviour)
        'clr'  – Centered Log-Ratio then StandardScaler (D → D features)
        'ilr'  – Isometric Log-Ratio then StandardScaler (D → D-1 features)

    Returns
    -------
    train_df, test_df : DataFrames with transformed feature columns added/replaced
    feature_cols      : output feature column names used by the model
    raw_cols          : original element column names (transformer input)
    transformer       : fitted CompositionalTransformer (saved as "scaler" in ckpt)
    y_col             : column in train_df/test_df holding the prediction target
    coords_train      : (n_train, 2) float32 ndarray
    tree              : KDTree built on coords_train
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["X", "Y"])

    raw_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS and df[c].dtype != object
    ]

    # ── Optionally restrict to a user-specified subset of elements ────────────
    if input_elements is not None:
        requested = set(input_elements)
        requested.add(target_element)
        available = [c for c in raw_cols if c in requested]
        missing   = requested - set(available)
        if missing:
            print(f"[warn] input_elements not found in CSV: {sorted(missing)}")
        raw_cols = available
        print(f"[load] using {len(raw_cols)} input element(s): {raw_cols}")

    df = half_detection_limit(df, raw_cols)

    np.random.seed(seed)
    n = len(df)
    test_idx  = np.random.choice(n, size=max(1, int(n * test_frac)), replace=False)
    train_idx = np.setdiff1d(np.arange(n), test_idx)

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    # ── Fit transformer on training data; apply to both splits ────────────────
    transformer = CompositionalTransformer(method=transform)
    X_tr = transformer.fit_transform(
        train_df[raw_cols].to_numpy(dtype=np.float32), raw_cols)
    X_te = transformer.transform(
        test_df[raw_cols].to_numpy(dtype=np.float32))
    feature_cols = transformer.feature_cols_

    for i, col in enumerate(feature_cols):
        train_df[col] = X_tr[:, i]
        test_df[col]  = X_te[:, i]

    # ── Prediction target column ───────────────────────────────────────────────
    if transform == "ilr":
        # target_element is absorbed into ILR coords; scale it separately via log
        y_tr = transformer.fit_transform_target(
            train_df[target_element].to_numpy(dtype=np.float32))
        y_te = transformer.transform_target(
            test_df[target_element].to_numpy(dtype=np.float32))
        train_df["__y_scaled"] = y_tr
        test_df["__y_scaled"]  = y_te
        y_col = "__y_scaled"
    else:
        y_col = target_element   # already a scaled column inside feature_cols

    coords_train = train_df[["X", "Y"]].to_numpy(dtype=np.float32)
    tree = KDTree(coords_train)

    return (train_df, test_df, feature_cols, raw_cols,
            transformer, y_col, coords_train, tree)


def build_neighbor_tokens(coord_xy: np.ndarray,
                          tree: KDTree,
                          coords_ref: np.ndarray,
                          X_ref: np.ndarray,
                          k: int,
                          exclude_self: bool = False) -> np.ndarray:
    """
    For one query point, retrieve K nearest neighbors and form token matrix.

    Returns : [k, 2+C] float32  where first two cols are (dx, dy)
    """
    n_avail = coords_ref.shape[0]
    k_fetch = min(k + 1 if exclude_self else k, n_avail)
    _, ind  = tree.query(coord_xy.reshape(1, -1), k=k_fetch)
    ind     = ind[0][1:] if exclude_self else ind[0]
    dxy     = (coords_ref[ind] - coord_xy).astype(np.float32)
    feats   = X_ref[ind].astype(np.float32)
    return np.hstack([dxy, feats])            # [k, 2+C]
