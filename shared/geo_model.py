"""
geo_model.py  –  GeoTransformer model + spatial neighbor utilities.

Copied and trimmed from geo_dataset/scripts/geo_anomaly_pretrain/geo_transformer.py.
Only the model and helper functions are kept here; data loading lives in preprocess.py.
"""

import numpy as np
import torch
import torch.nn as nn


# ── 2D Positional Encoding ────────────────────────────────────────────────────

class PositionalEncoding2D(nn.Module):
    """Sinusoidal 2D positional encoding for (x, y) coordinate pairs."""

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
            coords = coords.unsqueeze(1)
        x, y = coords[..., 0], coords[..., 1]
        B, N  = x.shape

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
        return torch.cat([pe_x, pe_y], dim=-1)


# ── GeoTransformer ────────────────────────────────────────────────────────────

class GeoTransformer(nn.Module):
    """
    Spatial Context Learning (SCL) model.

    Token sequence fed to a TransformerEncoder:
        token0  : element-type embedding (which element we predict)
        token1  : target location token (with 2D PE)
        token2… : K neighbor tokens  (Δx, Δy + element features, with 2D PE)

    The hidden state of token1 is used as the target-element-aware latent.
    The head_pred output is the predicted target element concentration (SCL task).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128,
                 n_heads: int = 4, n_layers: int = 3, num_elements: int = 1):
        super().__init__()
        self.element_embed = nn.Embedding(num_elements, hidden_dim)
        self.input_proj    = nn.Linear(in_dim, hidden_dim)
        self.coord_proj    = nn.Linear(2, hidden_dim)
        self.pos_encoder   = PositionalEncoding2D(hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden_dim * 4, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.head_pred   = nn.Linear(hidden_dim, 1)

    def forward(self,
                element_idx: torch.Tensor,
                target_coord: torch.Tensor,
                neighbor_tokens: torch.Tensor):
        """
        element_idx     : [B]         long
        target_coord    : [B, 2]      float  (x, y)
        neighbor_tokens : [B, K, 2+C] float  (Δx, Δy, feat_0 … feat_{C-1})

        Returns
        -------
        pred   : [B]    predicted target element value
        latent : [B, H] geo-context latent (hidden state of token1)
        """
        token0  = self.element_embed(element_idx).unsqueeze(1)          # [B,1,H]
        pe_tgt  = self.pos_encoder(target_coord)                         # [B,1,H]
        token1  = self.coord_proj(target_coord).unsqueeze(1) + pe_tgt   # [B,1,H]
        pe_nbr  = self.pos_encoder(neighbor_tokens[:, :, :2])           # [B,K,H]
        tokens_n = self.input_proj(neighbor_tokens) + pe_nbr            # [B,K,H]

        seq    = torch.cat([token0, token1, tokens_n], dim=1)           # [B,2+K,H]
        hidden = self.encoder(seq)                                       # [B,2+K,H]

        h_t  = torch.relu(self.head_hidden(hidden[:, 1, :]))            # [B,H]
        pred = self.head_pred(h_t).squeeze(-1)                          # [B]
        return pred, h_t


# ── Neighbor token builder ────────────────────────────────────────────────────

def build_neighbor_tokens(coord_xy: np.ndarray,
                          tree,
                          coords_ref: np.ndarray,
                          X_ref: np.ndarray,
                          k: int,
                          exclude_self: bool = False) -> np.ndarray:
    """
    Build [k, 2+C] neighbor token matrix for one query point.

    coord_xy    : (2,) query coordinate
    tree        : KDTree built on coords_ref
    coords_ref  : (N, 2) reference coordinates
    X_ref       : (N, C) scaled feature matrix matching coords_ref
    k           : number of neighbors
    exclude_self: skip the nearest neighbor (use when query is in coords_ref)

    Returns : [k, 2+C] float32  (Δx, Δy, feat_0 … feat_{C-1})
    """
    n_avail = coords_ref.shape[0]
    k_fetch = min(k + 1 if exclude_self else k, n_avail)
    _, ind  = tree.query(coord_xy.reshape(1, -1), k=k_fetch)
    ind     = ind[0][1:] if exclude_self else ind[0]
    dxy     = (coords_ref[ind] - coord_xy).astype(np.float32)
    feats   = X_ref[ind].astype(np.float32)
    return np.hstack([dxy, feats])
