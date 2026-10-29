#!/bin/bash
# Paper 1 A3: fine-tuned encoder text baseline, scored with the same aggregators and calibration (judge mode reads fixed
# per-row scores). CPU only; needs runs/qwen3_14b/encoder_scores from text_encoder_scores.py.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/paper1"; mkdir -p "$O"
cd "$THESIS/code"
python -u e2v2_analysis.py --signal judge --judge-dir "$R/encoder_scores" --core none --v2 "$R"/e2v2_*_c?_act \
  --v1-controls "$R"/e2sys_a_act "$R"/e2sys_b_act "$R"/e2ben_a_act "$R"/e2ben_b_act --workers 5 \
  --out "$O/encoder.json" > "$O/encoder.log" 2>&1
L=$(grep -m1 -A6 "PRE-VOTE, positives = ALL" "$O/encoder.log" | grep -E "peak\|last|ca_t\|last|peak\|cummean|ca_t\|cummean" | tr -s ' ' | cut -c1-80 | tr '\n' ';')
