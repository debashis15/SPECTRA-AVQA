#!/usr/bin/env bash
# scripts/test.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-spectra_avqa}"
SPLIT="${SPLIT:-test}"

python -u main_test.py \
    --config      ./configs/config.yaml \
    --checkpoint  "$CHECKPOINT" \
    --gpu         "$GPU" \
    --split       "$SPLIT" \
    --save_preds \
    "$@"
