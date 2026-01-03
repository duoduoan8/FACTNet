#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python scripts/experiments/eval_classical_baselines_thz.py \
  --val-root datasets/THZ_SR/vail \
  --scale 4 \
  --splits all \
  --lr-suffix _psf \
  --out-dir outputs

