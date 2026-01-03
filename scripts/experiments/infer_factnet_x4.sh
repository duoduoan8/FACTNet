#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DIR="${1:-datasets/THZ_SR/vail/split_1_x4_psf}"
OUTPUT_DIR="${2:-results/FACTNet/split_1_x4_psf}"
WEIGHT_PATH="${3:-experiments/pretrained_models/FACTNet/FACTNet_SRx4.pth}"

python inference/inference_factnet.py \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --task FACTNet \
  --scale 4 \
  --model_path "$WEIGHT_PATH"

