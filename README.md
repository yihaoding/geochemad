# GeoChemAD — Geochemical Anomaly Detection Benchmark

A reproducible benchmark for **unsupervised mineral-prospectivity anomaly
detection** from multi-element geochemical surveys. It bundles 16 real survey
tasks, a shared preprocessing/evaluation pipeline, classic and deep baselines,
and three progressively stronger transformer models (T1 → T3).

Anomaly detection here is **label-free**: models score every sample from its
chemistry and spatial context; known mineral-deposit sites are used *only* to
evaluate how well high scores concentrate near deposits.

---

## Repository layout

```
geochem-ad-benchmark/
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

T3 is the headline method: 10-area mean ROC-AUC **0.688** (attention) / **0.699**
(fusion) vs the strongest baseline 0.643. Full recipe + ablations:
[docs/method_t3_icdm.md](docs/method_t3_icdm.md).

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

## Reproducibility notes

- All 16 tasks, element lists, and per-area hyper-parameters are defined once in
  [`shared/area_config.py`](shared/area_config.py) — change configs there, not
  in the model scripts.
- **No machine-specific paths remain in the code.** Data resolves from
  `$GEOCHEM_DATA_ROOT` (default: bundled `data/gswa/`); T3 post-processing
  output resolves from `$GEOCHEM_OUT_ROOT` (default: `./outputs`).
- `models/t3_icdm/_vendor/` and `preprocess/_vendor/` hold small vendored copies
  of legacy dependencies so the pipeline runs self-contained. No model
  checkpoints or experiment outputs are committed (see `.gitignore`).
- Compare methods **only within the same evaluation protocol** (20-run
  negative-sampling AUC); single-best-epoch numbers are not comparable.

## Citing

If you use this benchmark, cite this repository and the original government
surveys the data derives from (see [LICENSE](LICENSE) and
[docs/datasets.md](docs/datasets.md)). The code is MIT-licensed; the data
retains its original providers' terms.
