"""
geo_transformer_multi.py
========================
GeoTransformerMulti: predicts ALL target-location element values in one pass.

Two architecture modes (--use_cross_attn):
───────────────────────────────────────────
  Encoder-only (default, use_cross_attn=False)
    seq = [target_token, nbr_1, …, nbr_K]   self-attention over all
    latent = hidden[:, 0, :]

  Encoder-Decoder (use_cross_attn=True)  ← borrowed from geo_dataset/new_geo_transformer
    memory  = Encoder([nbr_1, …, nbr_K])     neighbours attend each other
    latent  = Decoder([target_token], memory) target cross-attends to context
    ► cleaner information flow; target cannot "copy" neighbour values directly

Two positional encoding modes (--use_mlp_pe):
──────────────────────────────────────────────
  Sinusoidal 2D PE (default, use_mlp_pe=False)
    fixed frequency banks for (x, y) coordinates

  Learnable MLP PE (use_mlp_pe=True)  ← borrowed from geo_dataset/new_geo_transformer
    2-layer MLP maps raw (dx, dy) → hidden_dim
    adapts to the actual coordinate range of the dataset
"""

import numpy as np
import torch
import torch.nn as nn

from geo_transformer import PositionalEncoding2D


# ── Learnable MLP positional encoding ─────────────────────────────────────────

class PosEncMLP(nn.Module):
    """
    Learnable 2-layer MLP that maps relative (dx, dy) → hidden_dim.
    Adapts to actual coordinate ranges; no frequency hyper-parameters.
    Borrowed from geo_dataset/geo_transformer/new_geo_transformer.py.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, rel_xy: torch.Tensor) -> torch.Tensor:
        """rel_xy : [B, N, 2] or [B, 2]  →  [B, N, H] or [B, H]"""
        return self.net(rel_xy)


# ── GeoTransformerMulti ────────────────────────────────────────────────────────

class GeoTransformerMulti(nn.Module):
    """
    Multi-output spatial interpolation transformer.

    Parameters
    ----------
    in_dim          : 2 + C  (dx, dy, feat_0 … feat_{C-1})
    n_predict       : number of elements to predict simultaneously (P)
    hidden_dim      : transformer model dimension (H)
    n_heads         : attention heads
    n_layers        : encoder layers (decoder uses n_layers // 2, min 1)
    dropout         : dropout probability
    use_cross_attn  : if True, use Encoder-Decoder (cross-attention) architecture
    use_mlp_pe      : if True, use learnable MLP positional encoding
    """

    def __init__(self, in_dim: int, n_predict: int, hidden_dim: int = 256,
                 n_heads: int = 8, n_layers: int = 6, dropout: float = 0.0,
                 use_cross_attn: bool = False, use_mlp_pe: bool = False):
        super().__init__()
        self.n_predict      = n_predict
        self.use_cross_attn = use_cross_attn
        self.use_mlp_pe     = use_mlp_pe

        # ── Positional encoding ───────────────────────────────────────────────
        if use_mlp_pe:
            self.pos_encoder = PosEncMLP(hidden_dim)
        else:
            self.pos_encoder = PositionalEncoding2D(hidden_dim)

        # ── Input projections ─────────────────────────────────────────────────
        self.coord_proj = nn.Linear(2, hidden_dim)    # target absolute coords
        self.input_proj = nn.Linear(in_dim, hidden_dim)  # neighbour tokens

        # ── Transformer ───────────────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden_dim * 4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        if use_cross_attn:
            n_dec = max(1, n_layers // 2)
            dec_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim, nhead=n_heads, batch_first=True,
                dim_feedforward=hidden_dim * 4, dropout=dropout)
            self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_dec)

        # ── Prediction head ───────────────────────────────────────────────────
        self.head_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.head_pred   = nn.Linear(hidden_dim, n_predict)

    def _pe(self, xy: torch.Tensor) -> torch.Tensor:
        """Unified PE call for both sinusoidal and MLP variants."""
        if self.use_mlp_pe:
            # MLP PE expects [B, N, 2] or [B, 2]; handles both shapes
            if xy.dim() == 2:
                return self.pos_encoder(xy.unsqueeze(1))  # [B, 1, H]
            return self.pos_encoder(xy)                   # [B, N, H]
        else:
            return self.pos_encoder(xy)                   # sinusoidal

    def forward(self, target_coord: torch.Tensor,
                neighbor_tokens: torch.Tensor):
        """
        target_coord    : [B, 2]      float – (X, Y) of query point
        neighbor_tokens : [B, K, 2+C] float – (dx, dy, feat_0…feat_{C-1})

        Returns
        -------
        pred   : [B, P]   all-element predictions
        latent : [B, H]   target-location hidden state (for contrastive / anomaly use)
        """
        # ── Neighbour tokens ──────────────────────────────────────────────────
        pe_nbr   = self._pe(neighbor_tokens[:, :, :2])  # [B, K, H]
        tokens_n = self.input_proj(neighbor_tokens) + pe_nbr  # [B, K, H]

        # ── Target-location token ─────────────────────────────────────────────
        pe_tgt  = self._pe(target_coord)                       # [B, 1, H]
        token0  = self.coord_proj(target_coord).unsqueeze(1) + pe_tgt  # [B, 1, H]

        if self.use_cross_attn:
            # Encoder: neighbours attend each other → context memory
            memory = self.encoder(tokens_n)                    # [B, K, H]
            # Decoder: target cross-attends to neighbour memory
            h_seq  = self.decoder(token0, memory)              # [B, 1, H]
            h_t    = h_seq[:, 0, :]                            # [B, H]
        else:
            # Self-attention encoder: [target | nbr_1 … nbr_K]
            seq    = torch.cat([token0, tokens_n], dim=1)      # [B, 1+K, H]
            hidden = self.encoder(seq)                         # [B, 1+K, H]
            h_t    = hidden[:, 0, :]                           # [B, H]

        latent = torch.relu(self.head_hidden(h_t))             # [B, H]
        pred   = self.head_pred(latent)                        # [B, P]
        return pred, latent
