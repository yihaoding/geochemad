# Configuration & Protocol

This document is the single reference for **how data is preprocessed, how models
are configured, and how anomaly scores are evaluated**. All of it is driven by
`shared/area_config.py` and the shared modules in `shared/`.

---

## Preprocessing

Implemented in [`shared/preprocess.py`](../shared/preprocess.py) — the
`load_area()` entry point is shared by **all** baselines and T1–T3 models, so
every method sees identical inputs.

1. **Element selection.** Keep columns matching `^[A-Za-z]+_(ppm|ppb|pct)$`;
   drop oxide forms (redundant). Per-area overrides live in `area_config.py`
   (`base_elements` = all; `llm_elements` = curated pathfinder subset).
2. **BDL / missing values (sentinel `-9999`).** Two strategies:
   - `half_dl` — replace `<=0` / non-finite with **half the column's minimum
     positive value** (default for dense matrices).
   - `drop_col` — drop any element column whose BDL fraction exceeds
     `--bdl_threshold` (e.g. 0.95), used by the T3 recipe.
3. **Compositional transform.** `clr` (centred-log-ratio) is the default for the
   paper recipe; `log1p` and `ilr` are also supported. CLR removes the
   constant-sum (closure) artefact of compositional geochemistry.
4. **Scaling.** `StandardScaler` fit on the **training split only**, applied to
   all samples.
5. **Train/val split.** 5 % random validation, `seed=42`; neural models pick the
   best checkpoint by minimum validation reconstruction loss.
6. **Spatial graph.** KD-tree nearest-neighbour index, distance-sorted, built on
   `(X, Y)`; K is per-model (see below).

### Feature selection

[`shared/feature_select.py`](../shared/feature_select.py) implements three modes
(suffixes on output dirs): `none` → all elements; `llm` → the curated
pathfinder list in [`shared/llm_feature_selection.json`](../shared/llm_feature_selection.json)
(`_fs`); `pca` → top-k vs the target (`_pca`). The pathfinder rationale is in
[`shared/pathfinder_features.md`](../shared/pathfinder_features.md). The
**34-element LLM selection is the single biggest quality lever** in the T3
recipe (see [method_t3_icdm.md](method_t3_icdm.md)).

### Interpolation (for grid/patch models)

[`preprocess/`](../preprocess/) holds the interpolation utilities used to fill
element grids before the patch-based VAE baseline:
- `kriging_interp.py` — ordinary kriging (`pykrige`).
- `interpolation.py` — the VAE patch interpolation driver.
- `ml_fill.py`, `ml_pca_fill.py` — ML / PCA-based gap filling (T3 inputs).

---

## Evaluation protocol

Implemented in [`shared/evaluate.py`](../shared/evaluate.py); every method writes
`anomaly_scores.csv` with columns `x, y, anomaly_score` for **all N samples**,
then is scored identically:

- **Positive labels:** known deposit sites (`data/gswa/site/*.csv`).
- A sample is a positive hit if within `radius_km` of a deposit.
- **Negative sampling:** `neg_runs` random negative draws (default **20**),
  averaged, to control for the spatial base rate.
- Default parameters (`run_evaluate` in `preprocess.py`):
  `radius_km = 2.0`, `min_dist = 0.5`, `neg_runs = 20`.
  *(The T3 paper table uses `min_dist = 5.0`.)*
- **Primary metric:** mean ROC-AUC across the negative-sampling runs.
  Secondary: Average Precision, PR-AUC, Success-Rate@top-5 %, Recall@top-50,
  mean distance-to-deposit (DTD).

> Reproducibility note: a single best-epoch AUC (as in some earlier
> GeoChemFormer tables) is **not** comparable to the 20-run negative-sampling
> AUC used here. Always compare within the same protocol.

---

## Models

### Baselines — `baselines/`
`run_baselines.py` scores each sample with classic detectors on the shared
preprocessed matrix: **z-score**, **Mahalanobis**, **kNN distance** (k=5),
**One-Class SVM**, **Isolation Forest** (200 trees). The **VAE family**
(`baselines/vae/`) is the deep baseline: patch interpolation → VAE / VAE-GAN
reconstruction error, with unsupervised checkpoint selection.

### T1 — `models/t1_cl/` (T1-CL)
`run_t1_cl.py`: an AnomalyTransformer reconstruction model with **SCL
(supervised-contrastive) pretraining** keeping the latent target-element aware.
(`run_t1.py` is the no-pretraining ablation.) Anomaly score = per-sample
reconstruction MSE.

### T2 — `models/t2_geochemformer/` (pretrain + T1)
A two-part GeoChemFormer: a **GeoTransformer** spatial encoder pretrained to
predict masked element values (`pretrain_multi.py`), whose latent feeds an
**AnomalyTransformer** reconstruction head (`run_t2.py`, end-to-end variant).
Auxiliary contrastive + masked-element-modelling losses prevent latent collapse.

### T3 — `models/t3_icdm/` (hierarchical backbone + dual scoring heads)
The full recipe and ablations are in [method_t3_icdm.md](method_t3_icdm.md).
In brief — one frozen hierarchical SSL backbone (target-mask + MCM,
local/mid/global towers, K=512 split 16/128/368) read out two ways:
- **Stage 3a — reconstruction head** (`run_t1_with_my_backbone.py`): parametric
  MSE over `concat(X_clr, h_backbone)`.
- **Stage 3b — attention-dispersion head** (`extract_attention_v3.py`): a
  parameter-free, zero-training distance-weighted attention reduction
  (`attn_local_mean_dist`).
- **Stage 4 — fusion** (`fuse_t1_attn.py`): rank-mean of the two heads.

Headline 10-area mean AUC (20-run protocol): recon 0.650 · attention **0.688** ·
fusion **0.699** vs the T2 baseline 0.643.

> `models/t3_icdm/_vendor/` holds three vendored legacy dependencies
> (`evaluate.py`, `geochem_preprocessor.py`, `new_geo_transformer.py`,
> `our_model/feature_extractor.py`) so the T3 scripts run self-contained. They
> are imported via file-relative paths — no machine-specific paths remain.

---

## Path & output overrides

| Variable | Meaning | Default |
|----------|---------|---------|
| `GEOCHEM_DATA_ROOT` | location of `data/gswa` | in-repo `data/gswa/` |
| `GEOCHEM_OUT_ROOT`  | where T3 post-processing scripts read/write results | `./outputs` |

Training/eval scripts take explicit `--data`, `--deposits`, `--out_dir`
arguments; nothing writes outside the path you pass.
