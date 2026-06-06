#!/bin/bash
# =============================================================================
# Classic baselines (z-score / Mahalanobis / kNN / OCSVM / IsolationForest)
# across all areas. CPU-only — uses the shared preprocessing in shared/.
# =============================================================================
#SBATCH --job-name=baselines
#SBATCH --partition=work           # CPU partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
export GEOCHEM_DATA_ROOT="${GEOCHEM_DATA_ROOT:-${REPO}/data/gswa}"
mkdir -p "${REPO}/logs" "${REPO}/outputs/baselines"

cd "${REPO}/baselines"
${PYTHON} run_baselines.py \
    --method all \
    --out_root "${REPO}/outputs/baselines"

echo "✅ baselines done → ${REPO}/outputs/baselines"
