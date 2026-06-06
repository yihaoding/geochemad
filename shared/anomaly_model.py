"""
anomaly_model.py
================
Transformer-based autoencoder for geochemical anomaly detection.

Input per sample
----------------
  x : [B, C]  preprocessed element feature vector
  z : [B, H]  GeoTransformer latent (target-element-aware context)

Architecture
------------
  Sequence = [geo_token | elem_token_0 | … | elem_token_{C-1}]
               ↑ projected from z    ↑ element-id embed + scalar value

  TransformerEncoder → reconstruct each element value from its output token.

  Anomaly score = mean per-element squared reconstruction error → [B]
  High score indicates an out-of-distribution (anomalous) point.
"""

import torch
import torch.nn as nn


class AnomalyTransformer(nn.Module):

    def __init__(self, n_elem: int, latent_dim: int,
                 hidden: int = 128, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        """
        Parameters
        ----------
        n_elem      : number of element feature channels (C)
        latent_dim  : dimension of GeoTransformer latent vector (H)
        hidden      : Transformer hidden dimension (d)
        n_heads     : number of attention heads
        n_layers    : number of Transformer encoder layers
        """
        super().__init__()
        self.n_elem = n_elem

        # ── Element token projection ──────────────────────────────────────────
        # Each element token = concat(element_id_embed, scalar_value_proj) → hidden
        self.elem_embed = nn.Embedding(n_elem, hidden // 2)
        self.val_proj   = nn.Linear(1, hidden // 2)
        self.elem_proj  = nn.Linear(hidden, hidden)

        # ── Geo context token ─────────────────────────────────────────────────
        # Projects GeoTransformer latent z → one context token prepended to seq
        self.geo_proj = nn.Linear(latent_dim, hidden)

        # ── Transformer encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, batch_first=True,
            dim_feedforward=hidden * 4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Per-element reconstruction head ───────────────────────────────────
        self.decoder = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, z: torch.Tensor):
        """
        Parameters
        ----------
        x : [B, C]  element feature values (standardised)
        z : [B, H]  GeoTransformer latent

        Returns
        -------
        x_hat  : [B, C]       reconstructed element values
        hidden : [B, 1+C, d]  Transformer hidden states (geo token + elem tokens)
        """
        B, C = x.shape

        # Element tokens: [B, C, hidden]
        ids      = torch.arange(C, device=x.device).unsqueeze(0).expand(B, -1)
        e_emb    = self.elem_embed(ids)                      # [B, C, hidden//2]
        v_emb    = self.val_proj(x.unsqueeze(-1))            # [B, C, hidden//2]
        elem_tok = self.elem_proj(torch.cat([e_emb, v_emb], dim=-1))  # [B, C, hidden]

        # Geo context token: [B, 1, hidden]
        geo_tok = self.geo_proj(z).unsqueeze(1)

        # Full sequence: [geo | elem_0 … elem_{C-1}]
        seq    = torch.cat([geo_tok, elem_tok], dim=1)       # [B, 1+C, hidden]
        hidden = self.encoder(seq)                           # [B, 1+C, hidden]

        # Reconstruct element values from positions 1 … C
        x_hat = self.decoder(hidden[:, 1:, :]).squeeze(-1)  # [B, C]
        return x_hat, hidden

    @torch.no_grad()
    def score(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score per sample = mean squared reconstruction error.

        Returns : [B] float  (higher = more anomalous)
        """
        x_hat, _ = self.forward(x, z)
        return ((x - x_hat) ** 2).mean(dim=-1)
