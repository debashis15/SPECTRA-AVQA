#!/usr/bin/env bash
# scripts/train.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
CHECKPOINT="${CHECKPOINT:-spectra_avqa}"

python -u main_train.py \
    --config      ./configs/config.yaml \
    --checkpoint  "$CHECKPOINT" \
    --gpu         "$GPU" \
    "$@"
