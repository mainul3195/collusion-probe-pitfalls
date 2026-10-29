#!/bin/bash
# Paper 1 A3: fine-tuned ModernBERT text baseline scores (25 leave-scenario-and-template-out cells). 1 GPU.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"
cd "$THESIS/code"
python -u text_encoder_scores.py --v2 "$R"/e2v2_*_c?_act \
  --v1-controls "$R"/e2sys_a_act "$R"/e2sys_b_act "$R"/e2ben_a_act "$R"/e2ben_b_act --out "$R/encoder_scores"
