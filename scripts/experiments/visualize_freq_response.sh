#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

WEIGHT_PATH="${1:-experiments/pretrained_models/FACTNet/FACTNet_SRx4.pth}"
INPUT_PATH="${2:-datasets/THZ_SR/vail/split_6_x4_psf}"
OUT_DIR="${3:-freq_vis}"

python inference/visualize_freq_response.py \
  --weight "$WEIGHT_PATH" \
  --input "$INPUT_PATH" \
  --outdir "$OUT_DIR"

