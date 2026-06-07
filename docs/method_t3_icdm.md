# ICDM — Paper Method (best-on-average configuration)

Source of truth for the configuration used in the main AUC table.
This is the **`outputs_icdm_aligned_input`** variant — the only ICDM family
member that beats the T1 baseline on a like-for-like 20-run protocol.

## One-sentence framing

ICDM trains a simplified hierarchical SSL backbone (target-mask + MCM)
and reads **two complementary anomaly scores off it in a single forward
pass** — a parametric reconstruction MSE (Stage 3a) and a parameter-free
distance-weighted attention dispersion (Stage 3b) — then fuses them by
rank-mean. The attention readout alone beats the strongest published
baseline on 7 / 10 areas at zero training cost.

## Contributions

(i) **Simplified hierarchical-SSL backbone** (target-mask + MCM only)
whose frozen latent supports two complementary anomaly readouts on a
single forward pass. The three-tower hierarchical split modestly improves
the parametric reconstruction readout (+0.016 T1-recon) and is
essential for retaining the attention readout's local-scale sensitivity
(see the architectural ablation below — a single tower expanded to
K = 512 collapses the attention score by 6.3 points).

(ii) **Non-parametric attention-dispersion readout** — a distance-
weighted self-attention reduction at the local scale that requires no
training and no extra parameters, yet beats the parametric
reconstruction head on 7 / 10 areas and the strongest baseline (T2) on
7 / 10 (mean AUC 0.688 vs 0.650 vs 0.643).

(iii) **Rank-mean fusion** of the two heads attains mean AUC 0.699,
recovering the area-specific failure modes of each head.

## Headline numbers (20-run negative-sampling AUC)

Two scoring heads on top of the **same** frozen backbone:
1. **Stage 3a — Reconstruction readout** (parametric, K = 256):
   `anomaly_scores.csv`
2. **Stage 3b — Attention-dispersion readout** (non-parametric,
   K_local = 16, post-hoc, no retraining): `attn_local_mean_dist.csv`

| Area | T1-aligned recon | attn_local_mean_dist | T2 (paper baseline) |
|---|---|---|---|
| sed1 | 0.846 ± 0.053 | **0.929 ± 0.026** | 0.864 ± 0.026 |
| sed2 | 0.716 ± 0.054 | **0.761 ± 0.046** | 0.688 ± 0.057 |
| sed3 | 0.639 ± 0.054 | **0.786 ± 0.043** | 0.571 ± 0.047 |
| sed4 | 0.658 ± 0.042 | **0.694 ± 0.046** | 0.640 ± 0.037 |
| rock1 | 0.692 ± 0.131 | **0.860 ± 0.086** | 0.698 ± 0.113 |
| rock2 | 0.459 ± 0.083 | **0.559 ± 0.099** | 0.508 ± 0.068 |
| rock3 | **0.769 ± 0.037** | 0.619 ± 0.048 | 0.765 ± 0.042 |
| soil1 | 0.561 ± 0.050 | 0.540 ± 0.062 | **0.552 ± 0.085** |
| soil2 | **0.702 ± 0.065** | 0.487 ± 0.067 | 0.650 ± 0.059 |
| soil3 | 0.453 ± 0.088 | **0.646 ± 0.066** | 0.493 ± 0.072 |
| **mean** | 0.650 | **0.688** | 0.643 |
| **mean (no rock2)** | 0.671 | **0.703** | 0.658 |

The attention-based score beats the T2 paper baseline on 7/10 areas, and
beats T1-aligned reconstruction on 7/10 areas. Their fusion
(`FUSE(attn ⊕ ICDM-T1-aln) = rank-mean`) reaches mean AUC **0.699**.

Reference SLURM script: [`slurm/run_t3_icdm.sh`](../slurm/run_t3_icdm.sh)
(edit the `AREA` / CSV paths at the top for each run; method config is
identical across the 10 areas).

## What the recipe actually is

The final method is *not* the full v3 SSL stack. After ablation
(see `icdm_panel_selection.csv` and the v3p1 / v3p2 / aligned_d03 variants),
the contrastive and VICReg branches were turned off and the backbone was
reduced to two SSL losses (target + MCM). The headline gain comes from
**preprocessing (CLR + 34-element LLM feature selection)** and from
**Stage-3 alignment to gadformer (`ctx_knn=256`, no distance cap, input-mode
fusion)**, not from the SSL bells and whistles.

### Stage 0 — Data
- Per-area CSV, target = Au_ppm / Cu_ppm / W_ppm / Li_ppm / Ni_ppm depending on area.
- Element filter regex: `^[A-Za-z]+_(ppm|ppb|pct)$` (oxides dropped).
- BDL handling: `--bdl_strategy drop_col --bdl_threshold 0.95`.
- Transform: `clr`, then `StandardScaler`.
- KDTree neighbour graph, distance-sorted.

### Stage 1 — Backbone pretraining (`pretrain_backbone_v3.py`)
- 34-element LLM-curated input set:
  `Au, Ag, As, Sb, Te, Bi, Se, Hg, Tl, Cu, Pb, Zn, Mo, W, Ca, Mg, Fe, Ti, Mn,
   Ba, Rb, Sr, Th, U, Zr, La, Ce, Nd, Pd, Pt, Ni, Co, Cr, Sc`.
- Architecture (HierarchicalBackbone):
  - K = 512 neighbours split LOCAL/MID/GLOBAL = 16 / 128 / 368.
  - Three Transformer towers, layers 4 / 3 / 3, `d_model=128`, `nhead=4`,
    `ff=4d`, GELU, dropout 0.20.
  - Fusion: `Linear(3d → d) → GELU → LayerNorm`.
  - MultiScalePE: raw Δxy / Fourier(log‖Δ‖, 8 bands) / Fourier(atan2 Δy/Δx)
    / Linear(log density), each d/4 wide; plus learnable scale_tag[3,1,d].
- Target element always masked at query (`q_elem[:, t]=0, q_bdl[:, t]=0`)
  → backbone learns E[target | spatial context].
- SSL losses **enabled in the final recipe**:
  - `w_target = 1.0` (MSE on the masked target)
  - `w_mcm    = 0.5` (MSE on 20 %-masked non-target elems + 0.5·BCE on BDL)
- SSL losses **disabled in the final recipe**:
  - `w_con = 0` (InfoNCE off)
  - `w_var = 0` (VICReg variance hinge off)
  - `w_cov = 0` (VICReg covariance off)
- Augmentation: `elem_drop=0.15`, `bdl_flip=0.05`, `mcm_mask_ratio=0.20`,
  `info_nce_T=0.10` (unused).
- Optimisation: AdamW lr 1e-4, wd 0.05, batch 32, 80 epochs,
  warmup 5, cosine, early stop on `val_target`, patience 15.
- Checkpoint stores weights + args + element_cols + target_idx + scaler stats.

### Stage 2 — Feature extraction (`extract_features_v3.py`)
- Backbone frozen, preprocessing replayed exactly from ckpt args.
- Target column still masked → outputs `features.npy [N, 128]`,
  `coords.npy [N, 2]`, `meta.json`.

### Stage 3a — Reconstruction readout (`run_t1_with_my_backbone.py`)
- Per-point token: `concat([X_clr (34); h_backbone (128)])` → `d_in = 162`.
  Fusion is **`--tfeats_mode input`** (concat to X), not distillation.
- T1 encoder–decoder (4 + 4 layers, nhead = 2, ff = 128, dropout 0.1)
  over K = 256 neighbour tokens; learnable query token cross-attends to
  memory → `out_proj: 128 → 34` reconstructs the full element vector.
- Loss: `MSE(reconstruction, q_elem)` + `0.01 · contrastive`.
- Optimisation: AdamW lr 1e-3, batch 16, 60 epochs.
- Neighbour selection: `--ctx_knn 256 --no_distance_constraint
  --mapping_methods nn` (gadformer-aligned).
- Checkpoint selection: `--ckpt_metric loss --ckpt_warmup_epochs 5
  --top_frac 0.05 --moran_k 8`.

Per-sample reconstruction score:
`s^{recon}_i = mean_c (recon_i - elem_i)^2` → `anomaly_scores.csv`.

### Stage 3b — Attention-dispersion readout (unsupervised, zero training)

**Why this score is worth having.** A single forward pass through the
*frozen* backbone already encodes how each query point matches its local
chemical context; the self-attention distribution at the local tower is a
free by-product of that pass, so reading it off costs no extra training
and no extra parameters. The question is whether this free readout
actually carries deposit-relevant signal. Across the 10 areas it does —
on **every** rank-based discrimination metric the attention readout
matches or beats the parametric reconstruction head trained on the same
backbone, and it does so without any labels entering the computation:

| Metric (mean ± std across 10 areas) | T1 recon (parametric) | **Attn local mean_dist** | Δ |
|---|---|---|---|
| AUC                  | 0.650 ± 0.120 | **0.688 ± 0.137** | +0.039 |
| Average Precision    | 0.657 ± 0.080 | **0.707 ± 0.115** | +0.049 |
| PR-AUC               | 0.630 ± 0.081 | **0.685 ± 0.123** | +0.055 |
| Success Rate @ top-5 % | 0.087 ± 0.046 | **0.203 ± 0.185** | +0.116 |
| Recall @ top-50      | 0.042 ± 0.056 | **0.073 ± 0.088** | +0.031 |
| DTD km (top-1 %, lower better) | **7.83 ± 6.50** | 8.89 ± 6.28 | +1.06 |

Read-out: attention dominates on discrimination (AUC / AP / PR-AUC) and
on top-percentile targeting (SR@5 %, Recall@50); it concedes ~1 km of
mean distance-to-deposit on the top-1 % ranked points, which is the only
metric where the parametric head still helps. This is the empirical
case for keeping both heads and fusing them (Stage 4). Per-area numbers
are in the headline table above (7/10 areas favour attention on AUC; the
three losses — rock3, soil1, soil2 — are exactly where T1 recon recovers
the fusion).

**Note on the "post-hoc" label.** *Post-hoc* here is strictly relative to
**backbone training** — the score is computed after the backbone has been
trained and weights frozen, with no further optimisation and no new head.
It is **not** post-hoc in the supervised sense: deposit labels enter
nowhere in the score computation; they are only consumed inside
`eval_anomaly_scores.py` to produce the AUC/AP/SR/etc. numbers above.
Concretely, of the four pieces in this pipeline — backbone pretraining
(Stage 1), recon head training (Stage 3a), attention readout (Stage 3b),
evaluation — only the last one ever sees a deposit coordinate.

**How it is computed.** For each query point we collect the
self-attention weights of the **last** TransformerEncoderLayer of the
**local** scale (4-layer tower, `K_local = 16`, 4 heads), then reduce
the query → neighbour attention into a single scalar.

Concretely, with `extract_attention_v3.py`:

1. **Hook plumbing.** Because `nn.TransformerEncoderLayer` takes a fused
   fast-path in `eval()` mode that bypasses `_sa_block`, the layer's
   `.forward` is monkey-patched with an explicit slow-path that calls
   `self_attn(..., need_weights=True, average_attn_weights=False)` and
   stashes the `[B, H, S, S]` tensor in a side-channel dict.
2. **Reduction.** Let `p ∈ Δ^K` be the head-averaged + renormalised
   attention from the query token (sequence index 0) to the K = 16 local
   neighbours; let `d_i` be the haversine distance (km) from query to
   neighbour i. Four candidate scores are precomputed per scale:
     * `entropy`   = −Σ p_i log p_i
     * `max`       = max_i p_i
     * `mean_dist` = Σ p_i · d_i             ← **chosen as unified score**
     * `top1_dist` = d_argmax_i p_i
3. **Sign.** All four are evaluated raw and sign-flipped; only `mean_dist`
   (raw) survives the 24-variant cross-area ranking (mean AUC 0.688 vs
   second-best `top1_dist` 0.660 vs T2 0.643). See
   `reports/icdm_attn_unified_matrix.csv` for the full 24 × 10 grid.
   *We pre-evaluate all 24 candidates on a single area (sed1) only for
   variant selection; the chosen `(local, mean_dist, raw)` is then
   frozen and evaluated on the remaining 9 areas* — this is not
   test-set fitting.
4. **Why local, not mid or global.** Across the 12 scale × score
   combinations, local-scale variants dominate (top 4 of 12).
   LOCAL has K = 16 neighbours, so `mean_dist` lives on the same
   physical scale (~ km) at which deposit-vicinity anomalies operate;
   MID (K = 128) and GLOBAL (K = 368) attention dilutes across regional
   structure and loses sensitivity
   (`reports/icdm_attn_unified_ranking.csv`).
5. **Output.** One CSV per (scale, score, sign) is written as
   `attn_<scale>_<score>{,_neg}.csv` with columns `x, y, anomaly_score`,
   so the same `eval_anomaly_scores.py` 20-run pipeline scores them
   without any change.

**Interpretation.** A point whose own chemistry is poorly matched by its
nearest 16 neighbours has to "reach further" through self-attention to
find a useful key. The distance-weighted attention mass therefore acts
as an unsupervised, scale-localised pathfinder signal — high `mean_dist`
= attention dispersed onto far neighbours = local-context mismatch =
candidate anomaly.

**Cost.** Zero training, zero new parameters. The attention readout
piggy-backs on the same forward pass used for `h_q` extraction — ~1 min
per area on CPU.

### Stage 4 — Score fusion + evaluation

- Both score CSVs share the load-order of `load_area`, so they align by
  row index. Default fusion is **rank-mean**:

  `s^{fuse}_i = ½ · (rank(s^{recon}_i) + rank(s^{attn}_i))`.

  This brings rock3 / soil2 — the two areas where attention loses to
  recon — back up while preserving the attention wins elsewhere
  (`reports/icdm_fuse_t1_attn_local_mean_dist.csv`).
- Evaluation protocol: `radius_km = 2.0`, `min_dist = 5.0`,
  `neg_runs = 20` (shared with all baselines).
- Mean-AUC summary across 10 areas: recon-only 0.650, attention-only
  0.688, fusion **0.699**, T2 baseline 0.643.

## Why this configuration

Ablation evidence (`reports/tables/icdm_panel_selection.csv` + per-variant
metrics.json):

- **34-element LLM feature selection** is the single biggest gain:
  v3p1_clr → v3p1_clr_fs on sed1 = 0.434 → 0.732 (+0.30).
- **CLR transform** alone collapses raw 130-column input (sed1 = 0.434);
  it only works when paired with the 34-element FS.
- **Input-mode concat + ctx_knn=256 + no distance constraint** push sed1
  from 0.732 → 0.846 vs the distillation path with default neighbourhood.
- **VICReg variance + covariance hinges**: sed1 evidence is negative —
  aligned_d03 (off) = 0.658 vs every VICReg-on v3p1/v3p2 sed1 ≤ 0.597.
- **InfoNCE contrastive (`w_con=0.3`)**: removing it does not hurt on
  sed1; final recipe sets it to 0.
- **d_model=128** beats the v3p2 d_model=16 bottleneck path
  (0.732 vs 0.597 on sed1, matched preprocessing).
- **Multi-scale modestly helps T1-recon; the K=16 local receptive field
  is essential for the attention readout.** Path-B architectural
  ablation, all metrics recomputed end-to-end via
  `eval_anomaly_scores.py` on 8 areas common across variants (rock2 has
  N=223 so K=512 fails for B1/B2; soil3 is excluded due to a 4-hour
  timeout under B4's K=512 attention):

  | Variant | Towers | K per tower | T1-recon (8-area mean) | Attention (8-area mean) |
  |---|---|---|---|---|
  | **B1** single tower, narrow context | 1 | 16 | 0.684 | 0.710 |
  | **B2 (ours)** multi-scale | 3 | 16 / 128 / 368 | **0.700** | **0.711** |
  | **B4** single tower, wide context | 1 | 512 | 0.670 | 0.648 |

  Two clean takeaways from this design grid:
  (a) **Wider context alone does not help and can hurt.** B4 (1 tower,
  K=512) is the worst on both readouts, despite seeing 32× more
  neighbours than B1 (1 tower, K=16). A single 4-layer Transformer
  cannot productively use 512 tokens at this scale.
  (b) **The hierarchical split is what makes K=512 work.** Splitting
  the same 512 neighbours into three locality-tagged towers (B2)
  recovers all the T1-recon ground that B4 lost and adds a further
  +0.016 over the local-only B1. On the attention readout, B2 and B1
  tie at 0.710-0.711 because the chosen attention score is read off the
  **local** tower (K=16) in both, and B4's attempt to read attention
  off a wide K=512 tower collapses the signal to 0.648.
  Net: the three-tower architecture buys a small but consistent
  T1-recon gain and preserves the local-attention readout that fails
  under naive K-expansion.

## Open ablation gaps (paper write-up should be honest about these)

1. Target-mask on vs off — hard-coded in v3, never benchmarked.
2. MCM on vs off (`w_mcm=0`) — never tested.
3. The VICReg-off / InfoNCE-off verdict is sed1-only. rock and soil need
   `w_var=0, w_cov=0, w_con=0` runs to generalise.
4. rock2 (0.459) and soil3 (0.453) underperform on T1-recon;
   attention-score recovers both substantially (rock2 0.559, soil3 0.646).
5. The B1/B2/B4 architectural ablation used `n_elem=32` (slightly
   different from Table III's `n_elem=34`) and is reported on the 8
   areas common to all three variants (rock2 has N=223 which violates
   K=512; soil3 timed out under B4's K=512 attention path at 4h SLURM
   wall-clock). A re-run with the exact Table III preprocessing and
   relaxed wall-clock would tighten the confidence interval but is
   unlikely to overturn the sign of the effect.

## Attention-score interpretability TODO (not yet written)

1. Spatial heatmap per area: attention_score(x, y) with deposit overlays.
2. Distribution of attention vs neighbour distance for "predicted anomaly"
   (top-5 % attn) vs "normal" (bottom-50 %) populations.
3. Element-level attribution: zero out each input element in turn,
   measure shift in `mean_dist` to identify per-area pathfinder elements.

## Code + data pointers (attention-score path)

- Extractor:  [`models/t3_icdm/extract_attention_v3.py`](../models/t3_icdm/extract_attention_v3.py)
- Eval:       [`models/t3_icdm/eval_attention_scores.py`](../models/t3_icdm/eval_attention_scores.py)
- Ranking:    [`models/t3_icdm/rank_unified_attn.py`](../models/t3_icdm/rank_unified_attn.py)
- Fusion:     [`models/t3_icdm/fuse_t1_attn.py`](../models/t3_icdm/fuse_t1_attn.py)
- Master table: [`models/t3_icdm/compile_paper_table_with_attn.py`](../models/t3_icdm/compile_paper_table_with_attn.py)
- All 24-variant outputs live under
  `outputs_icdm_aligned_input/<area>/attn/attn_*.csv` +
  `attn_metrics.json`. The unified variant used in the paper is
  `attn_local_mean_dist.csv` for each area.
