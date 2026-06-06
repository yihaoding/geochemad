"""
SC-AnomalyFormer-v3 backbone.

Three concrete differences vs v2 (model.py):

 (1) ALWAYS mask the target element in every query during pretraining.
     The query never sees its own Au_ppm — the backbone has no choice but to
     learn a spatial-context-only predictor.  Resulting latent  h_q encodes
     "the expected target value given the neighbourhood", so any sample whose
     actual target differs from this expectation is, by construction, an
     anomaly candidate in latent space.  This bakes anomaly-awareness into
     the representation itself.

 (2) Hierarchical multi-scale neighbour encoder.
     The K=512 neighbours are split into three scales:
       LOCAL  = nearest k_local   (e.g. 16)
       MID    = next     k_mid    (e.g. 112)
       GLOBAL = furthest k_global (rest, ~384)
     Three transformer encoders process them separately.  The query token
     attends to its own scale of context.  Concatenated query latents → MLP
     fusion → final h_q.  Disentangles local geometry / regional baseline.

 (3) Stronger contrastive: hard augmentation + batch-level negatives.
     Two views of the SAME query differ by element dropout, BDL flag flip,
     and a different neighbourhood subsample.  Negatives come from OTHER
     queries in the batch (not the same-query other view).  Temperature
     T=0.1 with InfoNCE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Multi-scale PE (verbatim from v2; works for all 3 scales) ───────────
class MultiScalePE(nn.Module):
    def __init__(self, d_model: int, n_bands: int = 8):
        super().__init__()
        assert d_model % 4 == 0
        self.d_q = d_model // 4
        self.raw       = nn.Linear(2,                self.d_q)
        self.den       = nn.Linear(1,                self.d_q)
        self.proj_logd = nn.Linear(2 * n_bands,      self.d_q)
        self.proj_ang  = nn.Linear(2 * n_bands,      self.d_q)
        self.register_buffer(
            "freq_bands",
            torch.tensor([2.0 ** k for k in range(n_bands)], dtype=torch.float32),
        )

    @staticmethod
    def _fourier(x: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
        wx = x.unsqueeze(-1) * bands.view(*([1] * x.dim()), -1)
        return torch.cat([torch.sin(wx), torch.cos(wx)], dim=-1)

    def forward(self, rel_xy: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
        r     = torch.log(rel_xy.norm(dim=-1) + 1e-6)
        theta = torch.atan2(rel_xy[..., 1], rel_xy[..., 0])
        h_raw  = self.raw(rel_xy)
        h_logd = self.proj_logd(self._fourier(r,     self.freq_bands))
        h_ang  = self.proj_ang (self._fourier(theta, self.freq_bands))
        h_den  = self.den(torch.log(density.unsqueeze(-1).clamp_min(1.0) + 1.0))
        return torch.cat([h_raw, h_logd, h_ang, h_den], dim=-1)


# ─── 2. Hierarchical multi-scale backbone ────────────────────────────────────
class HierarchicalBackbone(nn.Module):
    SCALE_NAMES = ("local", "mid", "glob")

    def __init__(
        self,
        n_elem: int,
        d_model: int   = 128,
        n_heads: int   = 4,
        n_layers_local: int  = 4,
        n_layers_mid:   int  = 3,
        n_layers_glob:  int  = 3,
        dropout: float = 0.20,
        k_local: int   = 16,
        k_mid:   int   = 128,
        # ── Path B ablation toggles ──────────────────────────────────────
        use_scales: tuple   = ("local", "mid", "glob"),
        use_scale_tag: bool = True,
    ):
        super().__init__()
        self.n_elem  = n_elem
        self.d_model = d_model
        self.k_local = k_local
        self.k_mid   = k_mid
        # canonicalise: keep order local→mid→glob, drop unknowns
        self.use_scales = tuple(s for s in self.SCALE_NAMES if s in use_scales)
        if not self.use_scales:
            raise ValueError("use_scales must contain at least one of local/mid/glob")
        self.use_scale_tag = bool(use_scale_tag)

        self.elem_proj   = nn.Linear(2 * n_elem, d_model)
        self.pe          = MultiScalePE(d_model)
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # scale-tag: learnable when enabled, zero-frozen when ablated.
        if self.use_scale_tag:
            self.scale_tag = nn.Parameter(torch.randn(3, 1, d_model) * 0.02)
        else:
            self.register_buffer("scale_tag", torch.zeros(3, 1, d_model))

        def _stack(L: int) -> nn.TransformerEncoder:
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            return nn.TransformerEncoder(enc, num_layers=L)

        # Always build all three encoders so the state-dict shape stays stable
        # across runs that load each other's checkpoints, but unused scales
        # are simply never called in forward.
        self.enc_local = _stack(n_layers_local)
        self.enc_mid   = _stack(n_layers_mid)
        self.enc_glob  = _stack(n_layers_glob)
        self.norm_l, self.norm_m, self.norm_g = (nn.LayerNorm(d_model) for _ in range(3))

        # fusion: concat |use_scales| latents → d_model
        n_used = len(self.use_scales)
        self.fusion = nn.Sequential(
            nn.Linear(n_used * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # SSL heads
        self.target_head   = nn.Linear(d_model, 1)              # primary: target Au expectation
        self.mcm_head      = nn.Linear(d_model, n_elem)         # secondary: masked non-target elements
        self.mcm_bdl_head  = nn.Linear(d_model, n_elem)
        self.contrast_proj = nn.Linear(d_model, d_model)        # InfoNCE projection

    # ─── helpers ─────────────────────────────────────────────────────────────
    def _q_token(self, q_elem, q_bdl) -> torch.Tensor:
        emb = self.elem_proj(torch.cat([q_elem, q_bdl], dim=-1))
        return self.query_token.expand(emb.size(0), -1, -1) + emb.unsqueeze(1)

    def _nbr_tokens(self, n_elem, n_bdl, rel_xy, density, scale_idx: int) -> torch.Tensor:
        emb = self.elem_proj(torch.cat([n_elem, n_bdl], dim=-1))
        return emb + self.pe(rel_xy, density) + self.scale_tag[scale_idx]

    def _encode_scale(
        self, enc, norm,
        q_tok, n_elem, n_bdl, rel_xy, density, scale_idx,
    ) -> torch.Tensor:
        if n_elem.size(1) == 0:                                # empty scale (tiny areas)
            return q_tok.squeeze(1)
        nbr = self._nbr_tokens(n_elem, n_bdl, rel_xy, density, scale_idx)
        seq = torch.cat([q_tok, nbr], dim=1)
        out = norm(enc(seq))
        return out[:, 0]                                       # query latent at this scale

    # ─── frozen feature extraction (used by Stage 2) ────────────────────────
    def forward_feature(
        self, q_elem, q_bdl,
        n_elem_l, n_bdl_l, rel_xy_l, dens_l,
        n_elem_m, n_bdl_m, rel_xy_m, dens_m,
        n_elem_g, n_bdl_g, rel_xy_g, dens_g,
    ) -> torch.Tensor:
        q_tok = self._q_token(q_elem, q_bdl)
        # Only encode scales selected by ablation toggle.
        latents = []
        if "local" in self.use_scales:
            latents.append(self._encode_scale(
                self.enc_local, self.norm_l,
                q_tok, n_elem_l, n_bdl_l, rel_xy_l, dens_l, 0))
        if "mid" in self.use_scales:
            latents.append(self._encode_scale(
                self.enc_mid, self.norm_m,
                q_tok, n_elem_m, n_bdl_m, rel_xy_m, dens_m, 1))
        if "glob" in self.use_scales:
            latents.append(self._encode_scale(
                self.enc_glob, self.norm_g,
                q_tok, n_elem_g, n_bdl_g, rel_xy_g, dens_g, 2))
        return self.fusion(torch.cat(latents, dim=-1))

    # ─── pretrain heads ──────────────────────────────────────────────────────
    def forward_pretrain(
        self, q_elem, q_bdl,
        n_elem_l, n_bdl_l, rel_xy_l, dens_l,
        n_elem_m, n_bdl_m, rel_xy_m, dens_m,
        n_elem_g, n_bdl_g, rel_xy_g, dens_g,
    ):
        h_q = self.forward_feature(
            q_elem, q_bdl,
            n_elem_l, n_bdl_l, rel_xy_l, dens_l,
            n_elem_m, n_bdl_m, rel_xy_m, dens_m,
            n_elem_g, n_bdl_g, rel_xy_g, dens_g,
        )
        return {
            "h_q":          h_q,
            "target_pred":  self.target_head(h_q).squeeze(-1),
            "mcm_value":    self.mcm_head(h_q),
            "mcm_bdl":      self.mcm_bdl_head(h_q),
            "contrast_z":   F.normalize(self.contrast_proj(h_q), dim=-1),
        }


# ─── 3. Augmentation utilities for hard contrastive ─────────────────────────
def augment_query(
    q_elem: torch.Tensor, q_bdl: torch.Tensor,
    elem_drop: float = 0.15, bdl_flip: float = 0.05,
):
    """Augmentation for one view of a query."""
    drop = (torch.rand_like(q_elem) < elem_drop).float()
    q_elem_a = q_elem * (1.0 - drop)                           # zero out dropped
    q_bdl_a  = torch.where((torch.rand_like(q_bdl) < bdl_flip),
                           1.0 - q_bdl, q_bdl)                 # flip a few BDL flags
    return q_elem_a, q_bdl_a


def subsample_neighbours(K_total: int, K_keep: int, device) -> torch.Tensor:
    """Random K_keep indices in [0, K_total)."""
    perm = torch.randperm(K_total, device=device)
    return perm[:K_keep]
