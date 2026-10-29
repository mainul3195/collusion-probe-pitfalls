#!/bin/bash
# Paper 1 A4: detector scores on the steered episodes (re-read by the unsteered model). CPU only.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"
cd "$THESIS/code"
python -u steer_scores.py --v2 "$R"/e2v2_*_c?_act --v1-controls "$R"/e2sys_a_act "$R"/e2sys_b_act "$R"/e2ben_a_act "$R"/e2ben_b_act \
  --steer "$R"/steer/base_col_act "$R"/steer/abl_col_act "$R"/steer/ablr_col_act --out "$THESIS/results/steer/scores"
