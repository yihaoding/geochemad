#!/bin/bash
# =============================================================================
# T3 (ICDM) — full 4-stage pipeline for ONE area, with the paper recipe.
# Edit AREA / target / CSV names below, or drive from shared/area_config.py.
# Paths are repo-relative; override data location with $GEOCHEM_DATA_ROOT.
# =============================================================================
#SBATCH --job-name=t3_icdm
#SBATCH --partition=gpu            # GPU partition (CPU-only jobs: use `work`)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/t3_icdm_%j.out
#SBATCH --error=logs/t3_icdm_%j.err

set -euo pipefail

# ---- resolve repo root (this script lives in <repo>/slurm/) -----------------
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"                       # point at your env's python
DATA_ROOT="${GEOCHEM_DATA_ROOT:-${REPO}/data/gswa}"
export MPLBACKEND=Agg
mkdir -p "${REPO}/logs"

# ---- area definition (example: area1 sed1, Au) ------------------------------
AREA=sed1
TARGET=Au_ppm
DATA="${DATA_ROOT}/geochemical/area1_sediment_au.csv"
SITE="${DATA_ROOT}/site/area1_sediment_au_site.csv"

# 34-element LLM-curated pathfinder set (see shared/llm_feature_selection.json)
LLM_FULL="Au_ppm,Ag_ppm,As_ppm,Sb_ppm,Te_ppm,Bi_ppm,Se_ppm,Hg_ppm,Tl_ppm,Cu_ppm,Pb_ppm,Zn_ppm,Mo_ppm,W_ppm,Ca_ppm,Mg_ppm,Fe_ppm,Ti_ppm,Mn_ppm,Ba_ppm,Rb_ppm,Sr_ppm,Th_ppm,U_ppm,Zr_ppm,La_ppm,Ce_ppm,Nd_ppm,Pd_ppb,Pt_ppb,Ni_ppm,Co_ppm,Cr_ppm,Sc_ppm"
LLM_SHORT="Au,Ag,As,Sb,Te,Bi,Se,Hg,Tl,Cu,Pb,Zn,Mo,W,Ca,Mg,Fe,Ti,Mn,Ba,Rb,Sr,Th,U,Zr,La,Ce,Nd,Pd,Pt,Ni,Co,Cr,Sc"

OUT="${GEOCHEM_OUT_ROOT:-${REPO}/outputs}/icdm/${AREA}"
PRETRAIN="${OUT}/backbone"; FEATURES="${OUT}/features"; T1="${OUT}/t1"
mkdir -p "${PRETRAIN}" "${FEATURES}" "${T1}"

cd "${REPO}/models/t3_icdm"

echo "═══ Stage 1: hierarchical backbone (target + MCM, d=128) ═══"
${PYTHON} pretrain_backbone_v3.py \
    --csv_path "${DATA}" --target_element ${TARGET} \
    --transform clr --input_elements "${LLM_FULL}" \
    --bdl_strategy drop_col --bdl_threshold 0.95 \
    --k_neighbors 512 --k_local 16 --k_mid 128 \
    --d_model 128 --n_heads 4 \
    --n_layers_local 4 --n_layers_mid 3 --n_layers_glob 3 \
    --dropout 0.20 --epochs 80 --early_patience 15 --warmup 5 \
    --lr 1e-4 --weight_decay 0.05 --batch_size 32 \
    --mcm_mask_ratio 0.20 --elem_drop 0.15 --bdl_flip 0.05 --info_nce_T 0.10 \
    --w_target 1.0 --w_mcm 0.5 --w_con 0.0 --w_var 0.0 --w_cov 0.0 \
    --val_frac 0.10 --out_dir "${PRETRAIN}"

echo "═══ Stage 2: feature extraction (frozen backbone) ═══"
${PYTHON} extract_features_v3.py \
    --backbone_ckpt "${PRETRAIN}/backbone.pt" \
    --csv_path "${DATA}" --target_element ${TARGET} \
    --out_dir "${FEATURES}" --batch_size 32

echo "═══ Stage 3a: reconstruction head (concat X_clr + h_backbone) ═══"
${PYTHON} run_t1_with_my_backbone.py \
    --data "${DATA}" --deposits "${SITE}" \
    --target_elements "${LLM_SHORT}" \
    --epochs 60 --batch 16 --lr 1e-3 --contrastive_weight 0.01 \
    --checkpoint "${T1}/best.pt" --output "${T1}/anomaly_scores.csv" \
    --mode both --transform clr \
    --radius_km 2.0 --min_dist 5.0 --neg_runs 20 \
    --mapping_methods nn --no_distance_constraint --ctx_knn 256 \
    --tfeats_npy "${FEATURES}/features.npy" \
    --coords_npy "${FEATURES}/coords.npy" --tfeats_mode input \
    --ckpt_metric loss --ckpt_warmup_epochs 5 --top_frac 0.05 --moran_k 8

echo "═══ Stage 3b: attention-dispersion head (zero training) ═══"
${PYTHON} extract_attention_v3.py \
    --backbone_ckpt "${PRETRAIN}/backbone.pt" \
    --csv_path "${DATA}" --target_element ${TARGET} \
    --out_dir "${OUT}/attn"

echo "✅ T3 done → ${OUT}"
echo "   Stage 4 fusion + eval: run models/t3_icdm/fuse_t1_attn.py (see docs)."
