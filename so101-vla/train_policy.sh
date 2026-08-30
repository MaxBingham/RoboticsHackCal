#!/usr/bin/env bash
# ==============================================================================
# train_policy.sh
# Fine-tune a SmolVLA or ACT policy on recorded SO-101 demonstration episodes.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_DIR="${SCRIPT_DIR}/../lerobot"

# Activate environment
if [ -f "${LEROBOT_DIR}/.venv/bin/activate" ]; then
    source "${LEROBOT_DIR}/.venv/bin/activate"
elif [ -n "${CONDA_DEFAULT_ENV:-}" ]; then
    echo "Using active conda environment: ${CONDA_DEFAULT_ENV}"
fi

DATASET_REPO="${1:-local/so101_pick_place}"
OUTPUT_DIR="${2:-outputs/train/smolvla_so101}"
POLICY_TYPE="${POLICY_TYPE:-smolvla}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-20000}"

echo "=========================================================="
echo " SO-101 Policy Training"
echo "=========================================================="
echo " Policy Type:  ${POLICY_TYPE}"
echo " Dataset:      ${DATASET_REPO}"
echo " Output Dir:   ${OUTPUT_DIR}"
echo " Device:       ${DEVICE}"
echo " Batch Size:   ${BATCH_SIZE}"
echo " Train Steps:  ${STEPS}"
echo "=========================================================="

python -m lerobot.scripts.train \
    --policy.type="${POLICY_TYPE}" \
    --dataset.repo_id="${DATASET_REPO}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --device="${DEVICE}" \
    --output_dir="${OUTPUT_DIR}"

