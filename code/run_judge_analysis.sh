#!/bin/bash
# Judge analysis (CPU): same cells, aggregators and calibration as the probe. JUDGE_NAME = dir under runs/qwen3_14b,
# OUT_NAME = results/e2v2b/<OUT_NAME>.{json,log}.
set -euo pipefail
source "${THESIS_ROOT:?set THESIS_ROOT}/env.sh"
: "${JUDGE_NAME:?}" "${OUT_NAME:?}"
R="$THESIS/runs/qwen3_14b"; O="$THESIS/results/e2v2b"; mkdir -p "$O"
cd "$THESIS/code"
python -u e2v2_analysis.py --signal judge --judge-dir "$R/$JUDGE_NAME" --core "$R/activations/core" --v2 "$R"/e2v2_*_c?_act \
  --v1-controls "$R"/e2{sys,ben}_{a,b}_act --workers 5 --out "$O/$OUT_NAME.json" > "$O/$OUT_NAME.log" 2>&1
echo "judge analysis COMPLETE $(date -Is)"
