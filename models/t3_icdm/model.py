"""
ICDM-version model:
  Stage 1: Backbone — single-area SSL pretrained transformer used as feature extractor
  Stage 3: AnomalyT1 — paper-T1 architecture but per-point input is augmented with
                       the frozen backbone hidden state.

Design choices (with reasoning) are listed in docstrings of each module.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Multi-scale positional encoding ─────────────────────────────────────
class MultiScalePE(nn.Module):
    """
    PE used INSIDE the backbone only. K=512 spans 100m–30km in sed1, so single
    sinusoidal PE is dominated by far neighbours. We split d_pe = d/4 each into
      • raw_offset:  Linear([Δx, Δy]) preserves high-freq local geometry
      • log_dist:    Fourier on log(‖Δ‖) covers multi-scale uniformly
      • angular:     Fourier on atan2(Δy, Δx) encodes mineralisation strike
      • density:     Linear on log(local_density) lets backbone adapt to
                     sparse (sed1) vs dense (rock2) area without retuning K
    """
    def __init__(self, d_model: int, n_bands: int = 8):
        super().__init__()
        assert d_model % 4 == 0
        self.d_q = d_model // 4
        self.raw  = nn.Linear(2,           self.d_q)
        self.den  = nn.Linear(1,           self.d_q)
        self.proj_logd = nn.Linear(2 * n_bands, self.d_q)
        self.proj_ang  = nn.Linear(2 * n_bands, self.d_q)
        self.n_bands = n_bands
        # geometric series of frequencies 2^0 .. 2^(M-1)
        self.register_buffer(
            "freq_bands",
            torch.tensor([2.0 ** k for k in range(n_bands)], dtype=torch.float32),
        )

    @staticmethod
    def _fourier(x: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        # x: [...], bands: [M] → [..., 2M]
        wx = x.unsqueeze(-1) * bands.view(*([1] * x.dim()), -1)
        return torch.cat([torch.sin(wx), torch.cos(wx)], dim=-1)

    def forward(self, rel_xy: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        # rel_xy: [B, K, 2]  density: [B, K]
        r = torch.log(rel_xy.norm(dim=-1) + 1e-6)        # [B, K]
        theta = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])
        h_raw  = self.raw(rel_xy)
        h_logd = self.proj_logd(self._fourier(r,     self.freq_bands))
        h_ang  = self.proj_ang (self._fourier(theta, self.freq_bands))
        h_den  = self.den(torch.log(density.unsqueeze(-1).clamp_min(1.0) + 1.0))
        return torch.cat([h_raw, h_logd, h_ang, h_den], dim=-1)  # [B, K, d]


# ─── 2. Backbone with multi-task SSL heads ──────────────────────────────────
class BackboneTransformer(nn.Module):
    """
    SSL pretrained backbone used purely as a feature extractor.
    After Stage 1: weights frozen, only `forward_feature` is called.

    Architecture (chosen to be small enough for tiny areas like rock2 N=224
    yet expressive for sed1 N=1392):
      • d_model = 128     same as paper T1 → clean concat downstream
      • depth   = 4       paper SCL encoder uses 6; we drop to 4 since
                           our 3-task SSL converges faster (see pretrain log)
      • heads   = 4
      • dropout = 0.20    heavy regularisation for ≤ 5k training samples
    """
    def __init__(
        self, n_elem: int, d_model: int = 128,
        n_heads: int = 4, n_layers: int = 4, dropout: float = 0.20,
    ):
        super().__init__()
        self.n_elem = n_elem
        self.d_model = d_model

        # token = elem concentrations (n_elem) + BDL flags (n_elem)
        self.elem_proj = nn.Linear(2 * n_elem, d_model)
        self.pe        = MultiScalePE(d_model)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # SSL heads
        self.mcm_head      = nn.Linear(d_model, n_elem)     # masked element regression
        self.mcm_bdl_head  = nn.Linear(d_model, n_elem)     # masked BDL classification
        self.contrast_proj = nn.Linear(d_model, d_model)    # for InfoNCE
        self.target_head   = nn.Linear(d_model, 1)          # predict query target_elem

    def _tokens(
        self,
        query_elem: torch.Tensor, query_bdl: torch.Tensor,
        nbr_elem: torch.Tensor,   nbr_bdl: torch.Tensor,
        rel_xy: torch.Tensor,     density: torch.Tensor,
    ):
        # query token (single feature vector) -------- [B, 1, d]
        q_emb = self.elem_proj(torch.cat([query_elem, query_bdl], dim=-1))
        q_tok = self.query_token.expand(q_emb.size(0), -1, -1) + q_emb.unsqueeze(1)

        # neighbour tokens -------- [B, K, d]
        nbr_emb = self.elem_proj(torch.cat([nbr_elem, nbr_bdl], dim=-1))
        pe      = self.pe(rel_xy, density)
        nbr_tok = nbr_emb + pe
        return torch.cat([q_tok, nbr_tok], dim=1)           # [B, 1+K, d]

    def forward_feature(
        self, query_elem, query_bdl, nbr_elem, nbr_bdl, rel_xy, density,
    ) -> torch.Tensor:
        """Returns query latent  h ∈ R^d  for downstream feature extraction."""
        tok = self._tokens(query_elem, query_bdl, nbr_elem, nbr_bdl, rel_xy, density)
        out = self.norm(self.encoder(tok))
        return out[:, 0]                                    # query position

    def forward_pretrain(
        self, query_elem, query_bdl, nbr_elem, nbr_bdl, rel_xy, density,
        target_y: torch.Tensor,                       # log-target for Task C
    ):
        tok = self._tokens(query_elem, query_bdl, nbr_elem, nbr_bdl, rel_xy, density)
        out = self.norm(self.encoder(tok))
        h_q = out[:, 0]
        return {
            "h_q":           h_q,
            "mcm_value":     self.mcm_head(h_q),
            "mcm_bdl_logit": self.mcm_bdl_head(h_q),
            "contrast_z":    F.normalize(self.contrast_proj(h_q), dim=-1),
            "target_pred":   self.target_head(h_q).squeeze(-1),
        }


# ─── 3. Anomaly transformer (T1 architecture, augmented input) ──────────────
class AnomalyT1(nn.Module):
    """
    Stage-3 model.  Architecture mirrors paper T1 (encoder-decoder, d_model=128,
    nhead=2, 4 enc + 4 dec layers), but each input token is augmented with the
    backbone hidden state h_i ∈ R^d_b for that point.

      per-point feature = [concentrations (C) ;  BDL_flags (C) ;  h_i (d_b)]
                          ↑↑↑ d_in = 2C + d_b
      then `in_proj` maps d_in → d_model
      then standard T1 encoder-decoder runs as in paper
    """
    def __init__(
        self, n_elem: int, d_backbone: int,
        d_model: int = 128, nhead: int = 2,
        num_enc_layers: int = 4, num_dec_layers: int = 4,
        dim_feedforward: int = 128, dropout: float = 0.1,
    ):
        super().__init__()
        d_in = 2 * n_elem + d_backbone
        self.in_proj  = nn.Linear(d_in, d_model)
        self.pos_mlp  = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                         dropout=dropout, batch_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_enc_layers)
        self.decoder = nn.TransformerDecoder(dec, num_layers=num_dec_layers)
        # paper T1 reconstructs the element vector for anomaly scoring
        self.out_proj = nn.Linear(d_model, n_elem)

    def forward(
        self,
        query_xy:   torch.Tensor,                       # [B, 2]
        query_feat: torch.Tensor,                       # [B, 2C + d_b]
        nbr_xy:     torch.Tensor,                       # [B, K, 2]
        nbr_feat:   torch.Tensor,                       # [B, K, 2C + d_b]
    ) -> torch.Tensor:
        B = query_xy.size(0)

        # neighbour tokens for encoder
        nbr_tok = self.in_proj(nbr_feat) + self.pos_mlp(nbr_xy)
        memory  = self.encoder(nbr_tok)                  # [B, K, d_model]

        # query token for decoder
        q_tok = self.in_proj(query_feat) + self.pos_mlp(query_xy)
        q_tok = (q_tok + self.query_token.expand(B, -1, -1).squeeze(1)).unsqueeze(1)

        h_q   = self.decoder(q_tok, memory).squeeze(1)   # [B, d_model]
        return self.out_proj(h_q)                        # [B, n_elem]
