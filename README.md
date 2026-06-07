# OpenGADBench — Benchmarking Unsupervised Geochemical Anomaly Detection
 
<!-- badges -->
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Data: GSWA](https://img.shields.io/badge/Data-GSWA%20Open-orange)](https://dasc.dmirs.wa.gov.au/)

---
 
Geochemical anomaly detection (GAD) is central to mineral exploration — deviations from regional baselines may indicate mineralization. Existing work suffers from three key limitations: (1) proprietary datasets that prevent reproduction; (2) single-region, single-medium, or single-element evaluation; and (3) reconstruction-only scoring that conflates generic outliers with genuine mineralisation signals.
 
**We introduce OpenGADBench**, the first open, multi-region, multi-medium benchmark for unsupervised GAD, compiled from authoritative government geological surveys in Australia and Canada. We further propose **SAGAD**, a self-supervised attention-guided framework with a dual-view anomaly readout — combining reconstruction error with a parameter-free attention-dispersion score — that outperforms all baselines across 10 diverse survey tasks.
 
---
 
## Highlights
 
- **10 benchmark tasks** across three sampling media (sediment, rock-chip, soil), four target commodities (Au, Cu, Ni, W), and two countries (Australia, Canada)
- **12 baselines** systematically reproduced and evaluated: statistical (Z-score, Mahalanobis, kNN), classical ML (IsolationForest, OCSVM), deep generative (AE, DAE, VAE, VAE-GAN, VAE-cascade-GAN, VAE-Diffusion), and transformer (T1 vanilla)
- **SAGAD (T3)** achieves mean AUC **0.696** across 10 tasks, outperforming all baselines — driven by a non-parametric attention-dispersion score that requires zero additional training
- **Practical guidance** on preprocessing: feature selection, CLR/ILR transforms, interpolation, and label-free checkpoint selection
- Fully reproducible: shared 20-run negative-sampling AUC protocol, open data, MIT-licensed code
---

---

## Repository layout

```
geochemad/
├── data/gswa/              16 tasks — geochemical/ samples + site/ deposit labels
├── shared/                 the spine: area_config, preprocess, evaluate, models
│   ├── area_config.py      ← single source of truth for all 16 tasks
│   ├── preprocess.py       CLR/scaling, BDL(-9999) handling, load_area(), eval
│   ├── feature_select.py   none / llm / pca feature-selection modes
│   ├── evaluate.py         20-run negative-sampling AUC protocol
│   ├── geo_model.py · anomaly_model.py
│   ├── llm_feature_selection.json · pathfinder_features.md
├── preprocess/             interpolation & gap-filling
│   ├── kriging_interp.py · interpolation.py · ml_fill.py · ml_pca_fill.py
│   └── _vendor/preprocessing/   patch-interpolation library (kriging+CLR/ILR)
├── baselines/
│   ├── run_baselines.py     z-score · Mahalanobis · kNN · OCSVM · IsolationForest
│   └── vae/                 deep baseline: VAE / VAE-GAN over interpolated patches
├── models/
│   ├── t1_cl/               T1 — AnomalyTransformer + SCL contrastive pretraining
│   ├── t2_geochemformer/    T2 — GeoChemFormer (pretrained encoder + recon head)
│   └── t3_icdm/             T3 — hierarchical backbone + dual scoring heads (+_vendor/)
├── slurm/                   ready-to-edit SLURM templates
├── docs/                    datasets.md · configuration.md · method_t3_icdm.md
├── requirements.txt · LICENSE · .gitignore
```

## The three models

| Tier | Name | Idea | Entry point |
|------|------|------|-------------|
| **T1** | T1-CL | AnomalyTransformer reconstruction with supervised-contrastive pretraining | `models/t1_cl/run_t1_cl.py` |
| **T2** | GeoChemFormer | Pretrained spatial GeoTransformer encoder feeding an anomaly reconstruction head | `models/t2_geochemformer/run_t2.py` |
| **T3** | ICDM | One frozen hierarchical SSL backbone, read out as **(a)** a reconstruction score and **(b)** a zero-training attention-dispersion score, then rank-fused | `models/t3_icdm/` |

T3 is the headline method. Full recipe, fusion details, and ablations:
[docs/method_t3_icdm.md](docs/method_t3_icdm.md).

## Main results

10-area mean ROC-AUC under the shared **20-run negative-sampling protocol**
(`radius_km = 2.0`, `min_dist = 5.0`). Two scoring heads read off the same
frozen ICDM backbone; rank-mean fusion of both heads reaches mean AUC **0.699**.

| Area | ICDM recon (T1-aligned) | ICDM attention | T2 baseline |
|------|-------------------------|----------------|-------------|
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

Attention beats T2 on 7/10 areas; rank-mean fusion (**0.699** mean) recovers
area-specific failure modes of each head. Per-area breakdowns, architectural
ablations, and the full 16-area baseline grid are in
[docs/method_t3_icdm.md](docs/method_t3_icdm.md).

---
 
## Dataset — OpenGADBench
 
Data is sourced from the [Geological Survey of Western Australia (GSWA)](https://dasc.dmirs.wa.gov.au/) and the Geological Survey of Canada. All datasets are publicly available.
 
| Region | Source | ID | #Samples | Area (km²) | Dist. (km) | Target | Anom. Pts | #Elem. |
|--------|--------|----|----------|-----------|-----------|--------|-----------|--------|
| WA | Sediment | sed1 | 1,392 | ~8,523 | 2.48 | Au | 32 | 124 |
| WA | Sediment | sed2 | 2,994 | ~6,671 | 1.49 | Cu | 21 | 124 |
| WA | Rock-chip | rock1 | 3,790 | ~3,177 | 0.91 | W | 7 | 124 |
| WA | Rock-chip | rock2 | 224 | ~2 | 0.09 | Au | 12 | 124 |
| WA | Rock-chip | rock3 | 9,624 | ~7,423 | 0.88 | Cu | 21 | 124 |
| WA | Soil | soil1 | 2,734 | ~5.7 | 0.05 | Au | 14 | 126 |
| WA | Soil | soil2 | 5,163 | ~57 | 0.11 | Au | 17 | 126 |
| WA | Soil | soil3 | 21,040 | ~2,018 | 0.31 | Ni | 13 | 126 |
| Canada | Sediment | sed3 | 10,086 | ~3,538 | 0.59 | Au | 36 | 92 |
| Canada | Sediment | sed4 | 10,086 | ~3,538 | 0.59 | Cu | 34 | 92 |
 
Data is bundled under `data/gswa/`. Each task provides:
- `geochemical/<area>.csv` — sample coordinates + element concentrations (including abnormal values such as -9999 which require preprocessing)
- `site/<area>_site.csv` — verified deposit locations (used **only** for evaluation, never for training)
For full provenance, element lists, and data licence terms see [docs/datasets.md](docs/datasets.md).

---

## Quickstart

```bash
# 1. Environment
pip install -r requirements.txt
export PYTHON=$(which python)          # used by the SLURM templates

# 2. Data is bundled under data/gswa/. To use an external copy instead:
#    export GEOCHEM_DATA_ROOT=/path/to/data/gswa

# 3. Classic baselines on all 16 areas (CPU)
cd baselines
python run_baselines.py --method all --out_root ../outputs/baselines

# 4. T3 full pipeline for one area (GPU) — edit AREA at top of the script
sbatch slurm/run_t3_icdm.sh
```

Every method writes `anomaly_scores.csv` (`x, y, anomaly_score` for all N
samples) and is scored by the identical protocol in `shared/evaluate.py`
(see [docs/configuration.md](docs/configuration.md)).

## Reproducibility
 
- All 10 tasks, element lists, and per-area hyperparameters are defined once in [`shared/area_config.py`](shared/area_config.py). Change configs there, not in model scripts.
- **No machine-specific paths in the code.** Data resolves from `$GEOCHEM_DATA_ROOT` (default: bundled `data/gswa/`); outputs resolve from `$GEOCHEM_OUT_ROOT` (default: `./outputs`).
- `models/t3_icdm/_vendor/` and `preprocess/_vendor/` hold vendored copies of legacy dependencies so the pipeline runs self-contained.
- No model checkpoints or experiment outputs are committed (see `.gitignore`).
- **Compare methods only within the same evaluation protocol** (20-run negative-sampling AUC with `radius_km=2.0`, `min_dist=5.0`). Single-best-epoch or different sampling numbers are not comparable.
- All experiments were run on a single NVIDIA RTX A6000 (48 GB) GPU. CPU-only runs are supported for baselines and T3 attention extraction.

---

## Citing

If you use this benchmark, cite this repository and the original government
surveys the data derives from (see [LICENSE](LICENSE) and
[docs/datasets.md](docs/datasets.md)). The code is MIT-licensed; the data
retains its original providers' terms.

---
 
## Licence
 
Code: MIT — see [LICENSE](LICENSE).  
Data: retains the original providers' terms — see [docs/datasets.md](docs/datasets.md).