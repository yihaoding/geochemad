"""
Path A — post-hoc attention extraction from a trained v3 hierarchical backbone.

Hooks the *last* TransformerEncoderLayer of each scale (local / mid / glob),
captures the [B, H, S, S] attention tensor, then reduces query-token (idx 0)
attention over the K neighbour tokens into 4 candidate anomaly scores per scale:

    entropy   : -sum p log p                  (uniform attention)
    max       : max_i p_i                     (focused attention)
    mean_dist : sum_i p_i * d_i_km            (attention reaches far)
    top1_dist : d at argmax_i p_i             (single farthest peak)

All 12 (= 4 × 3 scales) scores are written per area as separate CSVs so the
existing eval_anomaly_scores.py pipeline can be reused without changes.

Does NOT retrain the backbone — only loads the existing checkpoint.
"""
import argparse, os, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from data     import load_area, build_neighbour_index, collate, iter_batches
from model_v3 import HierarchicalBackbone

SCALES = ("local", "mid", "glob")
SCORES = ("entropy", "max", "mean_dist", "top1_dist")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone_ckpt", required=True)
    p.add_argument("--csv_path",      required=True)
    p.add_argument("--target_element",required=True)
    p.add_argument("--out_dir",       required=True,
                   help="Per-scale per-score CSVs written here.")
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--latlon", action="store_true",
                   help="Treat coords as (lat, lon) and use haversine for "
                        "distance; otherwise euclidean degrees * 111 km.")
    return p.parse_args()


# ─── 1. Hook plumbing ────────────────────────────────────────────────────────
# nn.TransformerEncoderLayer takes a fused fast-path in eval() mode that
# bypasses _sa_block entirely, so a `_sa_block` monkey-patch never fires.
# We replace the layer's `forward` with an explicit slow-path version that
# also returns attention weights via a side-channel.
def _slow_forward_factory(layer: nn.TransformerEncoderLayer, store: dict, key: str):
    """Return a replacement for `layer.forward` that mirrors the standard
    TransformerEncoderLayer post-norm / pre-norm logic but uses need_weights=True
    on self_attn and stashes the [B, H, S, S] attention into `store[key]`.
    """
    def forward(src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if layer.norm_first:
            sa_in = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                sa_in, sa_in, sa_in,
                attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                need_weights=True, average_attn_weights=False, is_causal=is_causal,
            )
            x = x + layer.dropout1(attn_out)
            ffn_in = layer.norm2(x)
            ffn_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(ffn_in))))
            x = x + layer.dropout2(ffn_out)
        else:
            attn_out, attn_w = layer.self_attn(
                x, x, x,
                attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                need_weights=True, average_attn_weights=False, is_causal=is_causal,
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            ffn_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ffn_out))
        store[key] = attn_w.detach()
        return x

    return forward


def install_hooks(model: HierarchicalBackbone):
    """Patch the *last* layer of each scale's encoder. Returns a dict that the
    next forward fills with [B, H, S, S] tensors keyed by 'local' / 'mid' / 'glob'.
    """
    store = {}
    for key, enc in zip(SCALES, (model.enc_local, model.enc_mid, model.enc_glob)):
        last_layer = enc.layers[-1]
        last_layer.forward = _slow_forward_factory(last_layer, store, key)
    return store


# ─── 2. Per-sample reductions ────────────────────────────────────────────────
def _km_per_deg(lat_deg: np.ndarray) -> np.ndarray:
    """Approx km per degree for euclidean fallback (uses |lat|)."""
    return 111.0  # 1 degree of lat ≈ 111 km; lon shrinks with cos(lat) but
                  # we keep this simple — distances enter only as relative scores.


def _haversine_km(lat1, lon1, lat2, lon2):
    """[..., 2] coords, returns [...] distances in km."""
    R = 6371.0
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    s = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(s, 0, 1)))


def reduce_attn(attn_BHSS: torch.Tensor, k_neigh: int,
                rel_xy_BK2: torch.Tensor, q_xy_B2: torch.Tensor,
                latlon: bool) -> dict:
    """Reduce one scale's [B, H, S, S] attention into 4 scores per sample.

    Query token is index 0 of S; neighbour tokens are indices 1..K.
    Per-head attention is averaged before scoring to avoid head-id ambiguity.
    """
    if k_neigh == 0:
        B = attn_BHSS.shape[0]
        z = torch.zeros(B, device=attn_BHSS.device)
        return {s: z.cpu().numpy() for s in SCORES}

    p = attn_BHSS[:, :, 0, 1:1 + k_neigh].mean(dim=1)          # [B, K] head-mean
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)       # renormalise

    eps = 1e-12
    ent  = -(p * (p + eps).log()).sum(dim=-1)                  # [B]
    mx,  argmx = p.max(dim=-1)                                 # [B], [B]

    if latlon:
        q_np = q_xy_B2.cpu().numpy()
        n_np = (q_xy_B2.unsqueeze(1) + rel_xy_BK2).cpu().numpy()
        d_km = _haversine_km(q_np[:, None, 0], q_np[:, None, 1],
                              n_np[:, :, 0],     n_np[:, :, 1])
        d_km = torch.tensor(d_km, device=p.device, dtype=torch.float32)
    else:
        d_km = rel_xy_BK2.norm(dim=-1) * 111.0                 # rough

    mean_d  = (p * d_km).sum(dim=-1)                           # [B]
    top1_d  = d_km.gather(1, argmx.unsqueeze(1)).squeeze(1)    # [B]

    return {
        "entropy":   ent.cpu().numpy(),
        "max":       mx.cpu().numpy(),
        "mean_dist": mean_d.cpu().numpy(),
        "top1_dist": top1_d.cpu().numpy(),
    }


# ─── 3. Main extraction loop ─────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.backbone_ckpt, map_location=device, weights_only=False)
    a    = ckpt["args"]

    ds = load_area(
        args.csv_path, args.target_element,
        transform        = a["transform"],
        input_elements   = a["input_elements"],
        bdl_strategy     = a["bdl_strategy"],
        bdl_threshold    = a["bdl_threshold"],
        paper_compatible = a.get("paper_compatible", True),
    )
    N = len(ds["coords"]); C = ds["elem"].shape[1]
    if C != ckpt["n_elem"]:
        raise ValueError(f"C mismatch: csv={C}  ckpt={ckpt['n_elem']}")

    nbr_idx = build_neighbour_index(ds["coords"], a["k_neighbors"])
    K_total, k_l, k_m = a["k_neighbors"], a["k_local"], a["k_mid"]
    k_g = K_total - k_l - k_m
    sl_l = slice(0, k_l); sl_m = slice(k_l, k_l + k_m); sl_g = slice(k_l + k_m, K_total)
    k_per_scale = {"local": k_l, "mid": k_m, "glob": k_g}
    sl_per_scale = {"local": sl_l, "mid": sl_m, "glob": sl_g}

    use_scales = tuple(s.strip() for s in
                       a.get("use_scales", "local,mid,glob").split(",")
                       if s.strip())
    use_scale_tag = not a.get("no_scale_tag", False)
    model = HierarchicalBackbone(
        n_elem=C, d_model=a["d_model"], n_heads=a["n_heads"],
        n_layers_local=a["n_layers_local"],
        n_layers_mid  =a["n_layers_mid"],
        n_layers_glob =a["n_layers_glob"],
        dropout=a["dropout"], k_local=k_l, k_mid=k_m,
        use_scales=use_scales,
        use_scale_tag=use_scale_tag,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    store = install_hooks(model)

    # buffers: scores[scale][score_name] = float[N]
    scores = {s: {sc: np.zeros(N, dtype=np.float32) for sc in SCORES} for s in SCALES}
    target_idx = ds["target_idx"]

    with torch.no_grad():
        for q_ids in iter_batches(np.arange(N), args.batch_size, shuffle=False):
            batch = collate(ds, nbr_idx, q_ids, device)
            q_elem = batch["q_elem"].clone()
            q_bdl  = batch["q_bdl" ].clone()
            q_elem[:, target_idx] = 0.0
            q_bdl [:, target_idx] = 0.0

            # forward — hooks fill `store` with last-layer attn per scale
            _ = model.forward_feature(
                q_elem, q_bdl,
                batch["n_elem"   ][:, sl_l], batch["n_bdl"     ][:, sl_l],
                batch["rel_xy"   ][:, sl_l], batch["n_density" ][:, sl_l],
                batch["n_elem"   ][:, sl_m], batch["n_bdl"     ][:, sl_m],
                batch["rel_xy"   ][:, sl_m], batch["n_density" ][:, sl_m],
                batch["n_elem"   ][:, sl_g], batch["n_bdl"     ][:, sl_g],
                batch["rel_xy"   ][:, sl_g], batch["n_density" ][:, sl_g],
            )

            for s in SCALES:
                if store.get(s) is None:                       # empty scale
                    continue
                red = reduce_attn(
                    store[s], k_per_scale[s],
                    batch["rel_xy"][:, sl_per_scale[s]],
                    batch["q_xy"], args.latlon,
                )
                for sc in SCORES:
                    scores[s][sc][q_ids] = red[sc]

            store.clear()

    # write per-scale-per-score CSVs (with x, y, anomaly_score)
    coords = ds["coords"]
    import pandas as pd
    written = []
    for s in SCALES:
        for sc in SCORES:
            df = pd.DataFrame({
                "x": coords[:, 0],
                "y": coords[:, 1],
                "anomaly_score": scores[s][sc],
            })
            out_csv = Path(args.out_dir) / f"attn_{s}_{sc}.csv"
            df.to_csv(out_csv, index=False)
            written.append(str(out_csv))

    meta = {
        "N": N, "n_elem": C,
        "k_local": k_l, "k_mid": k_m, "k_glob": k_g,
        "n_heads": a["n_heads"], "d_model": a["d_model"],
        "backbone_ckpt": str(args.backbone_ckpt),
        "scores_written": written,
    }
    with open(Path(args.out_dir) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[attn-extract] {len(written)} CSVs → {args.out_dir}")


if __name__ == "__main__":
    main()
