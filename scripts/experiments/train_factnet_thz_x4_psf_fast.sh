#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-options/train/THZ/train_FACTNet_THZ_SRx4_x4_psf_fast.yml}"
META_INFO="options/train/THZ/meta_info_THZ_SR_train_x4_psf.txt"

if [[ ! -f "$META_INFO" ]]; then
  echo "[FACTNet] Meta info not found: $META_INFO"
  echo "[FACTNet] Generating meta info from paired folders..."
  python scripts/experiments/generate_meta_info_paired.py \
    --gt-root datasets/THZ_SR/train/transform_db_method5_wash_data_no_repeat \
    --lq-root datasets/THZ_SR/train/x4_psf \
    --out "$META_INFO"
fi

python factnet/train.py -opt "$CONFIG" --launcher none

